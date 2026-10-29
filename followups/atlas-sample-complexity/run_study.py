"""Stored-rollout sample complexity and fresh micro-atlas validation (design 28564651).

Run the CPU gate with python --smoke. Paid execution needs the project owner's
separate authorization. The only requested accelerator is A10G.
"""
import ast
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
STUDY = RESEARCH / '2026-10-26-persona-neurons'
sys.path[:0] = [str(RESEARCH), str(STUDY)]
import hf_cache

MODELS = {
    '0.8b': ('Qwen/Qwen3.5-0.8B', '2fc06364715b967f1860aea9cf38778875588b17'),
    '2b': ('Qwen/Qwen3.5-2B', '15852e8c16360a2fea060d615a32b45270f8a8fc'),
    '4b': ('Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
    '9b': ('Qwen/Qwen3.5-9B', 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'),
}
PILE = ('NeelNanda/pile-10k', '127bfedcd5047750df5ccf3a12979a47bfa0bafa')
HORIZONS = ['0', '0_user', '1', '4', '16', '64', 'full']
ALL_LAYER = {'0', '4', '16', '64', 'full'}
OUT = modal.Volume.from_name('personakit-out', create_if_missing=True)
ATLAS = modal.Volume.from_name('personakit-atlas', create_if_missing=True)
image = hf_cache.image('flash-linear-attention==0.6.0').add_local_python_source('persona', 'prompts', 'judge')
app = modal.App('personakit-atlas-sample-complexity')


def out_dir(tag):
    if not tag or Path(tag).name != tag:
        raise ValueError('tag must be a single directory name')
    root = Path("out") if modal.is_local() else Path('/out')
    path = root / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def call(fn, *args):
    if os.environ.get('PERSONAKIT_RESEARCH_CHECK'):
        return fn.remote(*args)
    fc = fn.spawn(*args)
    while True:
        try:
            return fc.get()
        except modal.exception.ConnectionError:
            print('[connection] reconnecting to existing call', flush=True)


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False))
    tmp.replace(path)
    if not modal.is_local():
        OUT.commit()


def unit(x):
    import torch
    norms = x.norm(dim=-1, keepdim=True)
    if not torch.isfinite(x).all() or (norms < 1e-10).any():
        raise ValueError('nonfinite or zero direction')
    return x / norms


def stats(x):
    import torch
    x = torch.as_tensor(x).detach().float().cpu()
    if not x.numel() or not torch.isfinite(x).all():
        raise ValueError('invalid bootstrap statistics')
    return dict(mean=float(x.mean()), sd=float(x.std(unbiased=False)),
                p2_5=float(x.quantile(.025)), p97_5=float(x.quantile(.975)))


def read_s1(key, smoke):
    import torch
    suffix = '-smoke' if smoke else ''
    src = out_dir(f's1a-{key}{suffix}')
    records = [json.loads(s) for s in (src / 'texts.jsonl').read_text().splitlines()]
    labels = [json.loads(s) for s in (src / 'labels.jsonl').read_text().splitlines()]
    full = torch.load(src / 'acts.pt', map_location='cpu', weights_only=True, mmap=True)
    axis = torch.load(out_dir(f's1b-{key}{suffix}') / 'axis.pt', map_location='cpu', weights_only=True)
    calib = torch.load(src / 'calib.pt', map_location='cpu', weights_only=True)
    if len(records) != len(labels) or len(records) != len(full):
        raise ValueError('S1 records, labels and activations do not align')
    primary = {(r['role'], r['q']): i for i, r in enumerate(records)
               if r['kind'] == 'role' and r['system'] == 0}
    questions = {q for _, q in primary}
    selected = []
    for i, r in enumerate(records):
        kind = r['kind']
        keep = kind == 'concept' or (kind == 'default' and r['q'] in questions)
        if kind == 'role':
            p = primary.get((r['role'], r['q']))
            keep = r['system'] == 0 or (r['system'] == 1 and p is not None
                    and labels[p].get('role') != 3 and labels[i].get('role') == 3)
        if keep:
            selected.append(i)
    recs = [{**records[i], 'source_index': i, 'role_label': labels[i].get('role')} for i in selected]
    full_sel = full[selected].clone()
    for i, r in enumerate(recs):
        if not r.get('reply') or not str(r.get('reply', '')).strip() or r.get('n_tokens', 1) == 0:
            full_sel[i] = float('nan')
    if not smoke:
        counts = {k: sum(r['kind'] == k for r in recs) for k in ('role', 'default', 'concept')}
        if len(primary) != 3200 or counts['default'] != 200 or counts['concept'] != 100:
            raise ValueError(f'incomplete canonical S1 input: {counts}, primary={len(primary)}')
    return recs, full_sel, axis, calib['pile']['norm_q10_50_90'][:, 1].float()


def prompt_ids(tok, rec):
    import persona
    if rec.get('prompt'):
        return tok.encode(rec['prompt'], add_special_tokens=False)
    return persona.chat_ids(tok, rec['question'], rec.get('system_text', ''))


def extract(model, tok, prompts, replies, horizons, batch, all_layers=True):
    """Reduce each decoder output immediately; retain only CPU prefix summaries.

    Empty reply spans are NaN, never silently replaced by prefill representations.
    Those records are explicitly excluded by each horizon's sampling population.
    """
    import torch
    import persona
    layers = len(persona.blocks(model))
    mid = layers // 2
    width = model.config.hidden_size
    output = {t: torch.full((len(prompts), layers if all_layers and t in ALL_LAYER else 1, width),
                            float('nan'), dtype=torch.float16) for t in horizons}
    order = sorted(range(len(prompts)), key=lambda j: len(prompts[j]) + len(replies[j]))
    for offset in range(0, len(order), batch):
        js = order[offset:offset + batch]
        seqs = [prompts[j] + replies[j] for j in js]
        ids, mask = persona.pad_left(seqs, tok.pad_token_id, model.device)
        weights = {}
        for t in horizons:
            w = torch.zeros_like(mask, dtype=torch.float32)
            for b, j in enumerate(js):
                pad = ids.shape[1] - len(seqs[b])
                start = pad + len(prompts[j])
                if t == '0':
                    lo, hi = start - 1, start
                elif t == '0_user':
                    lo, hi = pad + 1, start
                else:
                    n = len(replies[j]) if t == 'full' else min(int(t), len(replies[j]))
                    lo, hi = start, start + n
                if hi > lo:
                    w[b, lo:hi] = 1 / (hi - lo)
            weights[t] = w
        handles = []
        def hook_at(layer):
            def hook(module, args, out):
                h = out[0] if isinstance(out, tuple) else out
                for t, w in weights.items():
                    every = all_layers and t in ALL_LAYER
                    if every or layer == mid:
                        reduced = torch.einsum('bt,btd->bd', w, h.float())
                        reduced[w.sum(-1) == 0] = float('nan')
                        output[t][js, layer if every else 0] = reduced.cpu().half()
            return hook
        try:
            for layer, block in enumerate(persona.blocks(model)):
                handles.append(block.register_forward_hook(hook_at(layer)))
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, logits_to_keep=1, use_cache=False)
        finally:
            for h in handles:
                h.remove()
    return output


def pile_norms(model, tok, docs, batch):
    """Per-passage token norms, preserving token weighting for the S1 median."""
    import torch
    import persona
    result = torch.full((len(docs), len(persona.blocks(model)), max(map(len, docs)) - 1), float('nan'))
    for s in range(0, len(docs), batch):
        chunk = docs[s:s + batch]
        ids, mask = persona.pad_left(chunk, tok.pad_token_id, model.device)
        handles = []
        def hook_at(layer):
            def hook(module, args, out):
                h = out[0] if isinstance(out, tuple) else out
                norms = h.float().norm(dim=-1).cpu()
                for b, doc in enumerate(chunk):
                    result[s + b, layer, :len(doc) - 1] = norms[b, ids.shape[1] - len(doc) + 1:]
            return hook
        try:
            for layer, block in enumerate(persona.blocks(model)):
                handles.append(block.register_forward_hook(hook_at(layer)))
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, logits_to_keep=1, use_cache=False)
        finally:
            for h in handles:
                h.remove()
    return result


def pile_docs(tok, smoke):
    dataset = hf_cache.dataset(*PILE, split='train')
    docs = []
    for r in dataset:
        ids = tok.encode(r['text'], add_special_tokens=False)[:128]
        if len(ids) == 128:
            docs.append(ids)
        if len(docs) == (4 if smoke else 256):
            return docs
    raise ValueError('insufficient Pile passages')


def medoids(vectors, k):
    """Deterministic farthest-point initialization and best-improving PAM swaps."""
    import torch
    dist = (1 - unit(vectors) @ unit(vectors).T).clamp(0, 2).cpu()
    n = len(dist)
    if k >= n:
        return list(range(n))
    selected = [int(dist.sum(1).argmin())]
    while len(selected) < k:
        near = dist[:, selected].min(1).values
        near[selected] = -1
        selected.append(int(near.argmax()))
    while True:
        current = float(dist[:, selected].min(1).values.sum())
        best, best_cost = None, current
        for pos in range(k):
            other = selected[:pos] + selected[pos + 1:]
            nearest = dist[:, other].min(1).values if other else torch.full((n,), float('inf'))
            costs = torch.minimum(nearest[:, None], dist).sum(0)
            costs[selected] = float('inf')
            candidate = int(costs.argmin())
            if float(costs[candidate]) < best_cost - 1e-6:
                best, best_cost = (pos, candidate), float(costs[candidate])
        if best is None:
            return selected
        selected[best[0]] = best[1]


def role_population(recs, filtered, valid=None, kept=None):
    result = {}
    for i, r in enumerate(recs):
        if r['kind'] != 'role' or (valid is not None and not valid[i]):
            continue
        if filtered:
            if r['role_label'] != 3 or (kept is not None and r['role'] not in kept):
                continue
        elif r['system'] != 0:
            continue
        result.setdefault(r['role'], []).append(i)
    return result


def role_lists(recs, full, axis, extra_counts=()):
    import torch
    mid = full.shape[1] // 2
    valid = torch.isfinite(full[:, mid].float()).all(-1).tolist()
    defaults = full[[i for i, r in enumerate(recs) if r['kind'] == 'default' and valid[i]], mid].float().mean(0)
    lists = {}
    for filtered in (False, True):
        pop = role_population(recs, filtered, valid=valid, kept=axis['roles'])
        names = sorted(pop)
        if not names:
            lists[str(filtered)] = {}
            continue
        vectors = torch.stack([full[pop[n], mid].float().mean(0) - defaults for n in names])
        lists[str(filtered)] = {str(k): [names[j] for j in medoids(vectors, k)]
                                for k in sorted(set([2, 4, 8, 16, 40, len(names), *extra_counts])) if 0 < k <= len(names)}
    return lists


def sparse_weights(draws, n, device):
    """Each replicate is a weighted mean, with equal mass per selected role."""
    import torch
    rows, cols, vals = [], [], []
    for b, groups in enumerate(draws):
        for group in groups:
            rows.extend([b] * len(group))
            cols.extend(group)
            vals.extend([1 / (len(groups) * len(group))] * len(group))
    return torch.sparse_coo_tensor([rows, cols], vals, (len(draws), n), device=device).coalesce()


def sample_weights(recs, pop, names, nr, nq, strategy, B, device):
    roles = sorted(pop)
    rd = []
    for seed in range(B):
        rng = random.Random(seed)
        chosen = rng.sample(roles, nr) if strategy == 'random' else names
        rd.append([rng.sample(pop[name], min(nq, len(pop[name]))) for name in chosen])
    return sparse_weights(rd, len(recs), device)


def stage1_sweep(key, cache, lists, smoke, device):
    import torch
    recs, features, axis, reference_m = cache['recs'], cache['features'], cache['axis'], cache['m']
    mid = axis['unit'].shape[0] // 2
    B = 20 if smoke else 200
    source = '0.8b' if smoke else ('4b' if key == '2b' else '2b')
    result = {'model': key, 'B': B, 'cross_source': source, 'cross_smoke_surrogate': smoke,
              'axis': [], 'concept': [], 'pile': [], 'unavailable': [], 'medoids': lists[key],
              'canonical_records': len(recs), 'source_indices': [r['source_index'] for r in recs]}
    concept_indices = {c: [i for i, r in enumerate(recs) if r.get('concept') == c]
                       for c in axis['concept_dirs']}
    concepts = list(concept_indices)
    total_role_qs = len({r['q'] for r in recs if r['kind'] == 'role'})
    # A concept curve is conditional on its exact role/question/horizon axis draw.
    # Save all cross-model candidate curves and the standard 8x4 anchor, not an
    # unrelated full-reference axis masquerading as a micro-atlas projection.
    for t in HORIZONS:
        data = features[t]
        x = data[:, mid if data.shape[1] > 1 else 0].float().to(device)
        valid = torch.isfinite(x).all(-1).cpu().tolist()
        x = x.nan_to_num()
        by_q = {}
        for i, r in enumerate(recs):
            if r['kind'] == 'default' and valid[i]:
                by_q.setdefault(r['q'], []).append(i)
        default_fixed_vec = x[[i for i, r in enumerate(recs) if r['kind'] == 'default' and valid[i]]].mean(0).expand(B, -1)
        default_draws = {}
        for nq in [1, 2, 4, 10, 40]:
            nd = min(len(by_q), max(2, nq))
            dd = [[[i for q in random.Random(10000 + seed).sample(sorted(by_q), nd) for i in by_q[q]]] for seed in range(B)]
            dw = sparse_weights(dd, len(recs), device)
            default_draws[nq] = (dw, torch.sparse.mm(dw, x), nd)
        concept_means_by_nc = {}
        for nc in [1, 2, 4, 6, 8, 10]:
            if all(sum(valid[i] for i in ids) >= nc for ids in concept_indices.values()):
                means = []
                for c in concepts:
                    ids = [i for i in concept_indices[c] if valid[i]]
                    draws = [[random.Random(seed).sample(ids, nc)] for seed in range(B)]
                    means.append(torch.sparse.mm(sparse_weights(draws, len(recs), device), x))
                concept_means_by_nc[nc] = torch.stack(means, 1)
        for filtered in (False, True):
            pop = role_population(recs, filtered, valid, axis['roles'])
            role_grid = sorted(set([2, 4, 8, 16, 40, len(pop)]))
            for nr in role_grid:
                if not nr or nr > len(pop):
                    continue
                for nq in [1, 2, 4, 10, 40]:
                    for strategy in ['random', 'diverse_kmedoids', 'diverse_cross']:
                        lookup = lists[source if strategy == 'diverse_cross' else key][str(filtered)]
                        names = lookup.get(str(nr), [])
                        meta = dict(filtered=filtered, n_roles=nr, n_questions=nq, T=t, strategy=strategy)
                        if strategy != 'random' and (len(names) != nr or any(n not in pop for n in names)):
                            result['unavailable'].append({**meta, 'reason': 'source medoids unavailable in target population'})
                            continue
                        if nq > total_role_qs:
                            result['unavailable'].append({**meta, 'reason': 'nq exceeds role extraction questions in dataset'})
                            continue
                        eligible = {n: ids for n, ids in pop.items() if (len(ids) > 0 if filtered else len(ids) >= nq)}
                        if len(eligible) < nr or (strategy != 'random' and any(n not in eligible for n in names)):
                            result['unavailable'].append({**meta, 'reason': 'insufficient distinct passing questions'})
                            continue
                        rw = sample_weights(recs, eligible, names, nr, nq, strategy, B, device)
                        dw, default_sub, nd = default_draws[nq]
                        role = torch.sparse.mm(rw, x)
                        for fixed in (False, True):
                            default = default_fixed_vec if fixed else default_sub
                            axes = unit(default - role)
                            cos = axes @ unit(axis['unit'][mid].float()).to(device)
                            cell = {**meta, 'default_fixed': fixed, 'n_default_q': nd,
                                    'roles': names if strategy != 'random' else None, 'cos_mid': stats(cos)}
                            key_cell = (nr == min(8, len(pop)) and nq == min(4, total_role_qs)) or (nr == len(pop) and nq == total_role_qs)
                            if key_cell and t in ALL_LAYER and not fixed:
                                cs = []
                                for layer in range(data.shape[1]):
                                    xl = data[:, layer].float().to(device).nan_to_num()
                                    al = unit(torch.sparse.mm(dw, xl) - torch.sparse.mm(rw, xl))
                                    cs.append((al * axis['unit'][layer].to(device)).sum(-1))
                                stack = torch.stack(cs, 1)
                                cell['cos_all_layers_min'] = stats(stack.min(1).values)
                                cell['cos_all_layers_mean'] = stats(stack.mean(1))
                            result['axis'].append(cell)
                            if not fixed and strategy == 'diverse_cross' and nr <= 8 and nq <= 4:
                                for nc, cm in concept_means_by_nc.items():
                                    for ci, c in enumerate(concepts):
                                        v = cm[:, ci] - (cm.sum(1) - cm[:, ci]) / (len(concepts) - 1)
                                        v -= (v * axes).sum(-1, keepdim=True) * axes
                                        cc = unit(v) @ unit(axis['concept_dirs'][c][mid].float()).to(device)
                                        result['concept'].append({**meta, 'n_concept_q': nc, 'concept': c, 'cos_mid': stats(cc)})
        print(f'[stage1] {key} T={t}: {len(result["axis"])} axis cells', flush=True)
    norms = cache['pile']
    for n in ([2, 4] if smoke else [8, 16, 32, 64, 128, 256]):
        # Resample passage clusters WITH replacement; take the token median, not
        # the median of passage medians. Seeds are paired across dose sizes.
        estimates = torch.stack([norms[torch.randint(len(norms), (n,), generator=torch.Generator().manual_seed(b))]
                                 .permute(1, 0, 2).flatten(1).nanquantile(.5, dim=1) for b in range(B)])
        error = (estimates - reference_m).abs() / reference_m
        result['pile'].append({'n_pile': n, 'relative_error_mid': stats(error[:, mid]),
                               'relative_error_all_layers': [stats(error[:, l]) for l in range(error.shape[1])]})
    return result


def choose_config(results, smoke):
    """Require every reference concept and every model to pass, then minimize tokens."""
    fallback = dict(n_roles=8, n_questions=4, n_default_q=4, n_concept_q=4, T='16', n_pile=32)
    eligible_sets = []
    for result in results.values():
        pile_n = 4 if smoke else 32
        dose_ok = any(p['n_pile'] == pile_n and p['relative_error_mid']['mean'] <= .03 for p in result['pile'])
        passing = set()
        if dose_ok:
            for a in result['axis']:
                if (a['strategy'] != 'diverse_cross' or a['filtered'] or a['default_fixed'] or
                    a['n_roles'] > 8 or a['n_questions'] > 4 or a['T'] not in ['1', '4', '16'] or
                    a['cos_mid']['mean'] < .95):
                    continue
                for nc in [1, 2, 4, 6, 8, 10]:
                    cs = [c for c in result['concept'] if not c['filtered'] and c['n_roles'] == a['n_roles']
                          and c['n_questions'] == a['n_questions'] and c['T'] == a['T'] and c['n_concept_q'] == nc]
                    if len(cs) == 10 and min(c['cos_mid']['mean'] for c in cs) >= .95:
                        passing.add((a['n_roles'], a['n_questions'], a['n_default_q'], nc, a['T']))
        eligible_sets.append(passing)
    common = set.intersection(*eligible_sets)
    if not common:
        return {**fallback, 'selection': 'preregistered_fallback', 'thresholds_met': False}
    cost = lambda c: (c[0] * c[1] + 5 * c[2] + 10 * c[3]) * int(c[4])
    best = sorted(common, key=lambda c: (cost(c), c))
    if len(best) > 1 and cost(best[0]) == cost(best[1]):
        return {**fallback, 'selection': 'preregistered_tie_break', 'thresholds_met': False}
    nr, nq, nd, nc, t = best[0]
    return dict(n_roles=nr, n_questions=nq, n_default_q=nd, n_concept_q=nc, T=t,
                n_pile=32, selection='minimum_token_budget', thresholds_met=True)


def synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_atlas(model, tok, recs, roles, config, docs, batch):
    """Fresh rollouts; timing excludes resident-model load and includes all build work."""
    import torch
    import persona
    import prompts as P
    synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pn = pile_norms(model, tok, docs[:config['n_pile']], batch)
    m = pn.permute(1, 0, 2).flatten(1).nanquantile(.5, dim=1)
    synchronize()
    pile_s = time.perf_counter() - started
    gen_start = time.perf_counter()
    primary = [r for r in recs if r['kind'] == 'role' and r['system'] == 0]
    qs = list(dict.fromkeys(r['q'] for r in primary))
    records = [r for r in primary if r['role'] in roles and r['q'] in qs[:config['n_questions']]]
    records += [r for r in recs if r['kind'] == 'default' and r['q'] in qs[:config['n_default_q']]]
    records += [dict(kind='concept', concept=c, question=q.format(x=c))
                for c in P.CONCEPTS_DEV + P.CONCEPTS_TEST for q in P.CONCEPT_QUESTIONS[:config['n_concept_q']]]
    ids = [prompt_ids(tok, r) for r in records]
    replies = [[] for _ in records]
    tokens = 0
    t = config['T']
    if t != '0':
        horizon = int(t)
        for sampled in (False, True):
            order = sorted([j for j, r in enumerate(records) if (r['kind'] != 'concept') == sampled], key=lambda j: len(ids[j]))
            for offset in range(0, len(order), batch):
                js = order[offset:offset + batch]
                outs = persona.generate(model, tok, [ids[j] for j in js], horizon, sample=sampled, seed=offset // batch)
                for j, reply in zip(js, outs):
                    replies[j] = reply
                    tokens += len(reply)
    synchronize()
    generation_s = time.perf_counter() - gen_start
    extract_start = time.perf_counter()
    # A fresh full-response extraction at this protocol's token budget.
    feature_t = '0' if t == '0' else 'full'
    feats = extract(model, tok, ids, replies, [feature_t], batch)[feature_t].float()
    valid = torch.isfinite(feats).all(dim=(1, 2))
    def mean(kind, name=None):
        idx = [i for i, r in enumerate(records) if valid[i] and r['kind'] == kind and
               (name is None or r.get('role', r.get('concept')) == name)]
        if not idx:
            raise ValueError(f'no valid fresh {kind} rows for {name}')
        return feats[idx].mean(0)
    default = mean('default')
    axis = unit(default - torch.stack([mean('role', r) for r in roles]).mean(0))
    concepts = P.CONCEPTS_DEV + P.CONCEPTS_TEST
    cm = torch.stack([mean('concept', c) for c in concepts])
    dirs = {}
    for ci, c in enumerate(concepts):
        v = cm[ci] - (cm.sum(0) - cm[ci]) / (len(cm) - 1)
        dirs[c] = v - (v * axis).sum(-1, keepdim=True) * axis
    atlas = dict(unit=axis, m=m, concept_means=cm, concepts=concepts, concept_dirs=dirs, config=config, roles=roles)
    synchronize()
    timing = dict(pile_s=pile_s, gen_s=generation_s, extract_s=time.perf_counter() - extract_start,
                  total_s=time.perf_counter() - started,
                  peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                  generated_tokens=tokens, rollouts=len(records), empty_replies=int((~valid).sum()),
                  includes_model_load=False, includes_data_load=False)
    return atlas, timing


def protocol_wizard(model, tok, atlas, batch, smoke):
    import torch
    import persona
    import prompts as P
    # All ten question forms are generated under each atlas's own horizon. For
    # micro protocols the preregistered N_concept_q prefix defines the direction.
    ids = [persona.chat_ids(tok, q.format(x='wizards')) for q in P.CONCEPT_QUESTIONS]
    t = atlas['config']['T']
    replies = []
    for s in range(0, len(ids), batch):
        replies.extend([[] for _ in ids[s:s + batch]] if t == '0' else
                       persona.generate(model, tok, ids[s:s + batch], int(t)))
    feature_t = '0' if t == '0' else 'full'
    features = extract(model, tok, ids, replies, [feature_t], batch)[feature_t].float()
    selected = features[:atlas['config']['n_concept_q']]
    valid = torch.isfinite(selected).all(dim=(1, 2))
    if not valid.any():
        raise ValueError('no valid wizard concept responses')
    v = selected[valid].mean(0) - atlas['concept_means'].float().mean(0)
    u = atlas['unit']
    atlas['concept_dirs']['wizards'] = v - (v * u).sum(-1, keepdim=True) * u
    return [{'question': q.format(x='wizards'), 'reply': tok.decode(r, skip_special_tokens=True),
             'used': i < atlas['config']['n_concept_q']} for i, (q, r) in enumerate(zip(P.CONCEPT_QUESTIONS, replies))]


def normalize_atlas(raw, config):
    import torch
    out = {'unit': unit(raw['unit'].float()), 'm': raw.get('m', raw.get('median_resid')).float(), 'config': config}
    out['concepts'] = raw.get('concepts', list(raw['concept_dirs']) if 'concept_dirs' in raw else [])
    out['concept_means'] = raw['concept_means'].float()
    out['concept_dirs'] = {}
    for i, c in enumerate(out['concepts']):
        v = out['concept_means'][i] - torch.cat([out['concept_means'][:i], out['concept_means'][i + 1:]]).mean(0)
        out['concept_dirs'][c] = (raw['concept_dirs'][c].float() if 'concept_dirs' in raw else
                                  v - (v * out['unit']).sum(-1, keepdim=True) * out['unit'])
    return out


def throughput(model, tok, recs, t, batch):
    import persona
    selected = [r for r in recs if r['kind'] == 'role' and r['system'] == 0][:batch]
    ids = [prompt_ids(tok, r) for r in selected]
    synchronize()
    started = time.perf_counter()
    replies = persona.generate(model, tok, ids, t, sample=True, seed=0)
    synchronize()
    gen_s = time.perf_counter() - started
    extract(model, tok, ids, replies, ['full'], batch)
    synchronize()
    seconds = time.perf_counter() - started
    return dict(max_new=t, n_prompts=len(ids), generated_tokens=sum(map(len, replies)), gen_s=gen_s,
                total_s=seconds, tokens_per_s=sum(map(len, replies)) / gen_s)


def evaluate(model, tok, atlases, concepts, smoke, batch, root):
    import torch
    import persona
    import prompts as P
    mid = len(persona.blocks(model)) // 2
    width = model.config.hidden_size
    zero_vec = torch.zeros(width, dtype=torch.float32)
    questions = P.IDENTITY_TEST[:1] if smoke else P.IDENTITY_TEST
    everyday = P.EVERYDAY_TEST[:1] if smoke else P.EVERYDAY_TEST
    mmlu = persona.mmlu_rows(4 if smoke else 300)
    saved = json.loads((root / 'generated.json').read_text()) if (root / 'generated.json').exists() else {}
    rows, capability = saved.get('rows', []), saved.get('mmlu', [])
    configs = [(a, f'{mode}_0.5_{da}', mode == 'both', da) for a in atlases
               for mode in ('recipe', 'both') for da in (.5, .25)]
    configs += [('random_dir', f'{mode}_0.5_{da}', mode == 'both', da)
                for mode in ('recipe', 'both') for da in (.5, .25)]
    configs += [(b, b, b == 'prompt_only', 0) for b in ['prompt_only', 'neither']]
    max_new = 4 if smoke else 96
    gen_batch = 8 if smoke else 96
    mmlu_batch = 8 if smoke else 60

    def steering_vec(atlas_name, da, ci, concept):
        if atlas_name in ('prompt_only', 'neither') or da == 0:
            return zero_vec
        atlas = atlases['full_s1'] if atlas_name == 'random_dir' else atlases[atlas_name]
        u = atlas['unit'][mid].float()
        v = unit(atlas['concept_dirs'][concept][mid].float())
        vec = atlas['m'][mid].float() * (.5 * v - da * u)
        if atlas_name == 'random_dir':
            # Match the complete recipe norm, orthogonal to both the
            # assistant axis and the intended concept direction.
            g = torch.Generator().manual_seed(1000 + ci)
            r = torch.randn(len(u), generator=g)
            r = r - (r @ u) * u - (r @ v) * v
            vec = unit(r) * vec.norm()
        return vec

    def mmlu_seq(r, system=''):
        q = (f"The following are multiple choice questions (with answers) about "
             f"{r['subject'].replace('_', ' ')}.\n\n{r['question'].strip()}\n"
             + "".join(f"{c}. {t}\n" for c, t in zip('ABCD', r['choices'])))
        return persona.chat_ids(tok, q, system, reply='Answer:')

    with persona.Hooks(model) as hooks:
        def run_jobs(jobs):
            replies = [''] * len(jobs)
            order = sorted(range(len(jobs)), key=lambda i: len(jobs[i]['prompt_ids']))
            for offset in range(0, len(order), gen_batch):
                js = order[offset:offset + gen_batch]
                V = torch.stack([jobs[j]['vec'] for j in js])
                hooks.clear()
                if V.abs().sum() > 0:
                    hooks.add(mid, V)
                outs = persona.generate(model, tok, [jobs[j]['prompt_ids'] for j in js], max_new)
                for j, o in zip(js, outs):
                    replies[j] = tok.decode(o, skip_special_tokens=True)
            hooks.clear()
            return replies

        if not rows:
            t1_jobs, ev_jobs = [], []
            for atlas_name, condition, prompted, da in configs:
                concept_list = [(0, None)] if atlas_name == 'neither' else list(enumerate(concepts))
                for ci, concept in concept_list:
                    vec = steering_vec(atlas_name, da, ci, concept)
                    system = P.PERSONA_PROMPT.format(name=P.PERSONA_NAMES[concept]) if prompted else ''
                    for qi, q in enumerate(questions):
                        t1_jobs.append(dict(atlas=atlas_name, condition=condition, prompted=prompted, da=da,
                                            ci=ci, concept=concept, item=qi, question=q, system=system,
                                            vec=vec, prompt_ids=persona.chat_ids(tok, q, system)))
                    if not prompted or atlas_name == 'prompt_only':
                        for qi, q in enumerate(everyday):
                            ev_jobs.append(dict(atlas=atlas_name, condition=condition, concept=concept,
                                                item=qi, question=q, vec=vec,
                                                prompt_ids=persona.chat_ids(tok, q, system)))

            t1_replies = run_jobs(t1_jobs)
            print(f'[stage2] generated Turn-1: {len(t1_replies)} replies', flush=True)

            t2_jobs = []
            for j, reply in zip(t1_jobs, t1_replies):
                for c in (concepts if j['concept'] is None else [j['concept']]):
                    rows.append(dict(atlas=j['atlas'], condition=j['condition'], concept=c,
                                     endpoint='turn1', item=j['item'], question=j['question'], reply=reply))
                if j['condition'] in ('recipe_0.5_0.5', 'both_0.5_0.5', 'both_0.5_0.25', 'prompt_only', 'neither'):
                    history = [(j['question'], reply)]
                    t2_jobs.append({**j, 't1_reply': reply,
                                    'prompt_ids': persona.chat_ids(tok, P.CHALLENGE, j['system'], history=history)})

            t2_replies = run_jobs(t2_jobs)
            for j, reply in zip(t2_jobs, t2_replies):
                for c in (concepts if j['concept'] is None else [j['concept']]):
                    rows.append(dict(atlas=j['atlas'], condition=j['condition'], concept=c, endpoint='turn2',
                                     item=j['item'], question=P.CHALLENGE, reply=reply,
                                     history=[dict(question=j['question'], reply=j['t1_reply'])]))
            print(f'[stage2] generated Turn-2: {len(t2_replies)} replies', flush=True)

            ev_replies = run_jobs(ev_jobs)
            for j, reply in zip(ev_jobs, ev_replies):
                for c in (concepts if j['concept'] is None else [j['concept']]):
                    rows.append(dict(atlas=j['atlas'], condition=j['condition'], concept=c,
                                     endpoint='everyday', item=j['item'], question=j['question'], reply=reply))
            print(f'[stage2] generated Everyday: {len(ev_replies)} replies (total rows={len(rows)})', flush=True)
            save_json(root / 'generated.json', {'rows': rows, 'mmlu': capability})
        else:
            print(f'[stage2] resumed {len(rows)} generated rows from checkpoint', flush=True)

        if not capability:
            persona.free()
            letters = [tok.encode(f' {c}', add_special_tokens=False)[0] for c in 'ABCD']
            unprompted = [('neither', 'neither', None, zero_vec)] + [
                (atlas_name, condition, concept, steering_vec(atlas_name, da, ci, concept))
                for atlas_name, condition, prompted, da in configs
                if not prompted and atlas_name != 'neither'
                for ci, concept in enumerate(concepts)
            ]
            V_unp = torch.stack([v for _, _, _, v in unprompted])
            unp_right = [[] for _ in unprompted]
            hooks.clear()
            hooks.add(mid, V_unp)
            with torch.inference_mode():
                for r in mmlu:
                    seq = mmlu_seq(r, '')
                    ids = torch.tensor([seq] * len(unprompted), device=model.device)
                    mask = torch.ones_like(ids)
                    preds = model(input_ids=ids, attention_mask=mask, logits_to_keep=1,
                                  use_cache=False).logits[:, -1, letters].argmax(-1).tolist()
                    for k, p in enumerate(preds):
                        unp_right[k].append(int(int(p) == r['answer']))
            hooks.clear()
            for c in concepts:
                capability.append(dict(atlas='neither', condition='neither', concept=c, correct=unp_right[0]))
            for (atlas_name, condition, concept, _), right in zip(unprompted[1:], unp_right[1:]):
                capability.append(dict(atlas=atlas_name, condition=condition, concept=concept, correct=right))
            for concept in concepts:
                system = P.PERSONA_PROMPT.format(name=P.PERSONA_NAMES[concept])
                seqs = [mmlu_seq(r, system) for r in mmlu]
                right = []
                with torch.inference_mode():
                    for s in range(0, len(seqs), mmlu_batch):
                        ids, mask = persona.pad_left(seqs[s:s + mmlu_batch], tok.pad_token_id, model.device)
                        preds = model(input_ids=ids, attention_mask=mask, logits_to_keep=1,
                                      use_cache=False).logits[:, -1, letters].argmax(-1).tolist()
                        right.extend(int(int(p) == r['answer']) for p, r in zip(preds, mmlu[s:s + mmlu_batch]))
                capability.append(dict(atlas='prompt_only', condition='prompt_only', concept=concept, correct=right))
            save_json(root / 'generated.json', {'rows': rows, 'mmlu': capability})
            print(f'[stage2] completed MMLU: {len(capability)} concept-condition evaluations', flush=True)
        else:
            print(f'[stage2] resumed {len(capability)} MMLU evaluations from checkpoint', flush=True)
    return rows, capability, mmlu


def score_replies(model, tok, rows, batch=32):
    """Read shared rubric literals without executing the legacy module's imports."""
    import torch
    import persona
    import prompts as P
    import importlib.util
    source = Path(importlib.util.find_spec('judge').origin).read_text()
    rubrics = {n.targets[0].id.lower(): ast.literal_eval(n.value) for n in ast.parse(source).body
               if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id in ('CONCEPT', 'IDENTITY', 'COHERENT')}
    digits = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(2)]
    labels = [{} for _ in rows]
    for name, template in rubrics.items():
        prompts = [persona.chat_ids(tok, template.format(question=r['question'], answer=r['reply'][:3000],
                    concept=P.PERSONA_NAMES[r['concept']])) for r in rows]
        groups = {}
        for j, p in enumerate(prompts):
            groups.setdefault(tuple(p), []).append(j)
        unique_keys = sorted(groups, key=len)
        for offset in range(0, len(unique_keys), batch):
            ks = unique_keys[offset:offset + batch]
            ids, mask = persona.pad_left([list(k) for k in ks], tok.pad_token_id, model.device)
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=1, use_cache=False).logits[:, -1].float()
            probs = logits[:, digits].softmax(-1).cpu()
            mass = logits.softmax(-1)[:, digits].sum(-1).cpu()
            for k, pr, mm in zip(ks, probs, mass):
                valid = bool(torch.isfinite(pr).all() and torch.isfinite(mm))
                for j in groups[k]:
                    labels[j][name] = int(pr.argmax()) if valid else None
                    labels[j][name + '_p'] = pr.tolist() if valid else None
                    labels[j][name + '_mass'] = float(mm) if valid else None
        print(f'[judge] scored {name}: {len(rows)} ({len(unique_keys)} unique)', flush=True)
    return labels


def paired_ci(a, b):
    import torch
    if len(a) != len(b) or not a:
        raise ValueError('unpaired or empty comparison')
    diff = torch.tensor(a, dtype=torch.float32) - torch.tensor(b, dtype=torch.float32)
    draws = torch.randint(len(a), (2000, len(a)), generator=torch.Generator().manual_seed(42))
    boot = diff[draws].mean(1)
    return dict(n=len(a), difference=float(diff.mean()), ci95=[float(boot.quantile(.025)), float(boot.quantile(.975))], reps=2000)


def summarize(rows, labels, capability):
    import persona
    import prompts as P
    groups = {}
    for row, lab in zip(rows, labels):
        for scope in [row['concept'], 'pooled']:
            key = (row['atlas'], row['condition'], row['endpoint'], scope)
            values = groups.setdefault(key, {})
            pair = (row['concept'], row['item'])
            valid = all(lab.get(k) in (0, 1) for k in ['concept', 'identity', 'coherent'])
            values[pair] = ({'CI': lab['concept'], 'IB': 1 - lab['identity'], 'coherent': lab['coherent'],
                             'coherent_CI': lab['concept'] * lab['coherent'], 'CM': int(P.mentions(row['reply'], row['concept']))}
                            if valid else None)
    rates, differences = [], []
    for key, values in groups.items():
        atlas, condition, endpoint, scope = key
        for metric in ['CI', 'IB', 'coherent', 'coherent_CI', 'CM']:
            vals = [v[metric] for v in values.values() if v is not None]
            n, k = len(vals), sum(vals)
            rates.append(dict(atlas=atlas, condition=condition, endpoint=endpoint, scope=scope, metric=metric,
                              n=n, invalid=len(values) - n, successes=k, rate=k / n if n else None,
                              ci95=persona.wilson(k, n)))
            for reference in ['full_s1', 'k1_atlas', 'prompt_only']:
                ref_condition = 'prompt_only' if reference == 'prompt_only' else condition
                other = groups.get((reference, ref_condition, endpoint, scope))
                if other is None or atlas == reference:
                    continue
                keys = sorted(p for p, v in values.items() if v is not None and other.get(p) is not None)
                if keys:
                    differences.append(dict(atlas=atlas, condition=condition, endpoint=endpoint, scope=scope,
                                            metric=metric, reference=reference,
                                            **paired_ci([values[p][metric] for p in keys], [other[p][metric] for p in keys])))
    mgroups = {}
    for row in capability:
        mgroups.setdefault((row['atlas'], row['condition']), {})[row['concept']] = row['correct']
    msummary, mdiff = [], []
    for (atlas, condition), concepts in mgroups.items():
        # Average across concepts BEFORE resampling paired question IDs.
        n = len(next(iter(concepts.values())))
        item_means = [sum(v[i] for v in concepts.values()) / len(concepts) for i in range(n)]
        msummary.append(dict(atlas=atlas, condition=condition, accuracy=sum(item_means) / n,
                             n_items=n, n_concepts=len(concepts), per_concept={c: sum(v) / n for c, v in concepts.items()}))
        for reference in ['full_s1', 'k1_atlas', 'prompt_only']:
            other = mgroups.get((reference, 'prompt_only' if reference == 'prompt_only' else condition))
            if other is None or atlas == reference:
                continue
            if set(other) != set(concepts):
                raise ValueError('MMLU concepts do not pair')
            ref = [sum(v[i] for v in other.values()) / len(other) for i in range(n)]
            mdiff.append(dict(atlas=atlas, condition=condition, reference=reference, **paired_ci(item_means, ref)))
    return dict(rates=rates, differences=differences, mmlu=msummary, mmlu_differences=mdiff)


@app.function(image=image, gpu='A10G', memory=32768, timeout=2400,
              volumes={**hf_cache.volumes, '/out': OUT, '/atlas': ATLAS})
def run(smoke: bool, tag: str, historical_build: dict, provenance: dict):
    import torch
    import persona
    import prompts as P
    torch.set_num_threads(4)
    P.PERSONA_NAMES['wizards'] = 'a wizard'
    P.FAMILIES['wizards'] = ['wizard', 'spell', 'magic', 'wand', 'arcane', 'sorcer', 'enchant', 'mage', 'staff', 'potion']
    if not modal.is_local():
        OUT.reload()
        ATLAS.reload()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if smoke != bool(os.environ.get('PERSONAKIT_RESEARCH_CHECK')):
        raise ValueError('CPU checker is required for smoke, and only smoke is allowed on CPU')
    root = out_dir(tag)
    existing = json.loads((root / 'results.json').read_text()) if (root / 'results.json').exists() else None
    if existing and existing.get('status') == 'complete':
        raise FileExistsError(f'use a fresh tag: {root}')
    started = time.perf_counter()
    results = existing or dict(schema=1, status='running', smoke=smoke, device=device, provenance=provenance,
                               design_commit='28564651', stage1={}, stage2={}, deviations=[],
                               timing_scope='resident model and cached datasets; model/data loads reported separately')
    results['status'] = 'running'
    results['provenance'] = provenance
    if smoke and not results.get('deviations'):
        results['deviations'] = ['0.8b toy S1 input; 20 Stage 1 replicates; self-model diverse_cross surrogate',
                                 'micro/K1 sizes capped to stored toy support; 4 Pile documents',
                                 '1 identity/everyday prompt, cookies and wizards, 4-token evaluation replies, MMLU-4',
                                 'CPU timing and toy judge are execution checks, not scientific evidence']
    save_json(root / 'results.json', results)
    keys = ['0.8b'] if smoke else ['2b', '4b', '9b']
    lists = {}
    try:
        # Include the other models' R_max values, not only this model's own
        # grid, so a transferable k-medoids list exists whenever k is feasible.
        counts = []
        for key in keys:
            ax = torch.load(out_dir(f's1b-{key}' + ('-smoke' if smoke else '')) / 'axis.pt',
                            map_location='cpu', weights_only=True)
            counts.append(len(ax['roles']))
            del ax
        for key in keys:
            if (root / f'stage1-{key}.pt').exists():
                if key in results['stage1'] and 'medoids' in results['stage1'][key]:
                    lists[key] = results['stage1'][key]['medoids']
                else:
                    cache = torch.load(root / f'stage1-{key}.pt', map_location='cpu', weights_only=True, mmap=True)
                    lists[key] = role_lists(cache['recs'], cache['features']['full'], cache['axis'], counts)
                    del cache
                print(f'[stage1] reusing cached {key}', flush=True)
                continue
            print(f'[stage1] extracting {key}', flush=True)
            recs, full, axis, m = read_s1(key, smoke)
            model, tok = persona.load(hf_cache.model(*MODELS[key]), device)
            batch = 4 if smoke else (16 if key == '9b' else 32)
            ids = [prompt_ids(tok, r) for r in recs]
            replies = [tok.encode(r['reply'], add_special_tokens=False)[:64] for r in recs]
            for i, reply in enumerate(replies):
                if not reply:
                    full[i] = float('nan')
            lists[key] = role_lists(recs, full, axis, counts)
            valid_full = torch.isfinite(full[:, full.shape[1] // 2].float()).all(-1).tolist()
            reference_cm = torch.stack([full[[i for i, r in enumerate(recs) if r.get('concept') == c and valid_full[i]]].float().mean(0)
                                        for c in P.CONCEPTS_DEV + P.CONCEPTS_TEST])
            features = extract(model, tok, ids, replies, [t for t in HORIZONS if t != 'full'], batch)
            features['full'] = full
            docs = pile_docs(tok, smoke)
            norms = pile_norms(model, tok, docs, batch)
            cache = dict(recs=recs, features=features, axis=axis, m=m, pile=norms, reference_cm=reference_cm)
            torch.save(cache, root / f'stage1-{key}.pt')
            del model, tok, cache, features, full, norms, axis, reference_cm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Models are gone before any cross-model sweep. Only one model's CPU
        # summary cache is resident at a time; transfer middle-layer summaries.
        for key in keys:
            if key in results['stage1']:
                print(f'[stage1] reusing sweep {key}', flush=True)
                continue
            cache = torch.load(root / f'stage1-{key}.pt', map_location='cpu', weights_only=True, mmap=True)
            results['stage1'][key] = stage1_sweep(key, cache, lists, smoke, device)
            save_json(root / 'results.json', results)
            del cache
            persona.free()
        config = choose_config(results['stage1'], smoke)
        results['stage2']['selection'] = config
        key = '0.8b' if smoke else '4b'
        batch = 8 if smoke else 32
        if (root / 'stage2-atlases.pt').exists() and 'wizard_extraction' in results['stage2'] and 'timings' in results['stage2']:
            print('[stage2] reusing stage2-atlases.pt and timing summary from checkpoint', flush=True)
            atlases = torch.load(root / 'stage2-atlases.pt', map_location='cpu', weights_only=True)
            timings = results['stage2']['timings']
            model, tok = persona.load(hf_cache.model(*MODELS[key]), device)
        else:
            cache = torch.load(root / f'stage1-{key}.pt', map_location='cpu', weights_only=True, mmap=True)
            recs = cache['recs']
            raw_s1 = {**cache['axis'], 'concept_means': cache['reference_cm'], 'concepts': P.CONCEPTS_DEV + P.CONCEPTS_TEST}
            atlases = {'full_s1': normalize_atlas(raw_s1, dict(T='16' if smoke else '160', n_concept_q=1 if smoke else 10))}
            del raw_s1, cache
            source = key if smoke else '2b'
            effective = {**config}
            if smoke:
                nr = len({r['role'] for r in recs if r['kind'] == 'role'})
                nq = len({r['q'] for r in recs if r['kind'] == 'role'})
                effective.update(n_roles=min(config['n_roles'], nr), n_questions=min(config['n_questions'], nq),
                                 n_default_q=min(config['n_default_q'], nq), n_concept_q=1, n_pile=4)
            roles = lists[source]['False'][str(effective['n_roles'])]
            results['stage2']['effective_config'] = effective
            results['stage2']['role_source'] = source
            results['stage2']['roles'] = roles
            model, tok = persona.load(hf_cache.model(*MODELS[key]), device)
            docs = pile_docs(tok, smoke)
            # Warm up prefill and decoding, outside timed builds.
            persona.generate(model, tok, [persona.chat_ids(tok, 'Say hello.')], 2)
            synchronize()
            timings = {}
            for name, cfg in [('micro_rollout', effective), ('micro_prefill', {**effective, 'T': '0'})]:
                timings[name] = []
                for repeat in range(2 if smoke else 3):
                    atlas, timing = build_atlas(model, tok, recs, roles, cfg, docs, batch)
                    timings[name].append(timing)
                    if repeat == 0:
                        atlases[name] = atlas
                    print(f'[timing] {name} repeat {repeat + 1}: {timing["total_s"]:.2f}s', flush=True)
                del atlas
            if smoke:
                kcfg = {**effective, 'T': '16'}
                atlases['k1_atlas'], toy_timing = build_atlas(model, tok, recs, sorted(roles), kcfg, docs, batch)
                timings['k1_toy'] = [toy_timing]
            else:
                raw = torch.load('/atlas/4b.pt', map_location='cpu', weights_only=True)
                if raw['model'] != key or raw['revision'] != MODELS[key][1] or len(raw['roles']) != 40:
                    raise ValueError('K1 model/revision/role count mismatch')
                atlases['k1_atlas'] = normalize_atlas(raw, dict(T=str(raw['sizes']['max_new']), n_concept_q=raw['sizes']['concept_q']))
                del raw
            timings['throughput'] = {'micro': throughput(model, tok, recs, int(effective['T']), batch),
                                     'k1': throughput(model, tok, recs, 16 if smoke else 128, batch)}
            timings['historical_k1_build'] = historical_build
            timings['comparison'] = 'historical full K1 build plus same-container matched-prompt throughput; not a fresh full K1 build'
            timings['summary'] = {name: {field: stats([r[field] for r in reps]) for field in
                                        ['total_s', 'pile_s', 'gen_s', 'extract_s', 'peak_gpu_gb']}
                                  for name, reps in timings.items() if isinstance(reps, list)}
            results['stage2']['timings'] = timings
            results['stage2']['atlas_similarity'] = {}
            mid = len(persona.blocks(model)) // 2
            for name, atlas in atlases.items():
                results['stage2']['atlas_similarity'][name] = dict(
                    cos_axis_mid=float(atlas['unit'][mid] @ atlases['full_s1']['unit'][mid]),
                    dose_relative_error_mid=float((atlas['m'][mid] - atlases['full_s1']['m'][mid]).abs() / atlases['full_s1']['m'][mid]))
            wizard_records = {}
            for name, atlas in atlases.items():
                wizard_records[name] = protocol_wizard(model, tok, atlas, batch, smoke)
            results['stage2']['wizard_extraction'] = wizard_records
            torch.save(atlases, root / 'stage2-atlases.pt')
            save_json(root / 'results.json', results)
        concepts = ['cookies', 'wizards'] if smoke else P.CONCEPTS_TEST + ['wizards']
        rows, capability, mmlu = evaluate(model, tok, atlases, concepts, smoke, batch, root)
        del model, tok, atlases
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        judge_key = '0.8b' if smoke else '9b'
        jbatch = 8 if smoke else 64
        if (root / 'labels.json').exists():
            labels = json.loads((root / 'labels.json').read_text())
        else:
            judge_model, jtok = persona.load(hf_cache.model(*MODELS[judge_key]), device)
            labels = score_replies(judge_model, jtok, rows, batch=jbatch)
            del judge_model, jtok
            persona.free()
            save_json(root / 'labels.json', labels)
        results['stage2'].update(summarize(rows, labels, capability))
        results['stage2'].update(rows=rows, labels=labels, mmlu_raw=capability,
                                 mmlu_items=mmlu, judge=judge_key, judge_batch=jbatch)
        results['loads'] = hf_cache.loads
        results['elapsed_s'] = results.get('elapsed_s', 0.0) + (time.perf_counter() - started)
        results['status'] = 'complete'
        results['invalid_labels'] = sum(any(v.get(k) is None for k in ['concept', 'identity', 'coherent']) for v in labels)
        save_json(root / 'results.json', results)
        flat = dict(tag=tag, smoke=smoke, status='complete', stage1_models=len(keys),
                    stage1_axis_cells=sum(len(v['axis']) for v in results['stage1'].values()),
                    replies=len(rows), invalid_labels=results['invalid_labels'], elapsed_s=results['elapsed_s'])
        for name in ['micro_rollout', 'micro_prefill']:
            flat[name + '_seconds'] = timings['summary'][name]['total_s']['mean']
        for row in results['stage2']['rates']:
            if row['scope'] == 'pooled':
                print('PERSONAKIT_RESULT', json.dumps({'tag': tag, 'smoke': smoke, **row}, allow_nan=False), flush=True)
        print('PERSONAKIT_RESULT', json.dumps(flat, allow_nan=False), flush=True)
        return flat
    except Exception as exc:
        results['status'] = 'failed'
        results['error'] = f'{type(exc).__name__}: {exc}'
        save_json(root / 'results.json', results)
        raise


@app.local_entrypoint()
def main(smoke: bool = False, tag: str = ''):
    if smoke and not os.environ.get('PERSONAKIT_RESEARCH_CHECK'):
        raise ValueError('use python for CPU smoke')
    tag = tag or 'atlas-sample-complexity-' + ('smoke-' if smoke else '') + time.strftime('%Y%m%dT%H%M%S')
    history_path = Path.home() / 'workspace/persona-neurons/k1/build.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    if not smoke and not history:
        raise FileNotFoundError(f'required historical K1 timing: {history_path}')
    provenance = {'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'models': MODELS, 'pile': PILE, 'historical_build_path': str(history_path),
                  'bootstrap_seeds': '0..19' if smoke else '0..199', 'paired_bootstrap_seed': 42}
    result = call(run, smoke, tag, history, provenance)
    print('PERSONAKIT_RESULT', json.dumps(result, allow_nan=False), flush=True)
