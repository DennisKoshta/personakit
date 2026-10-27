"""personakit's command line. Each command runs one function of personakit.remote on Modal (on your account) and
prints what it found; -o writes the full result as JSON (for make and port, the recipe that `personakit.steer` reads).

    personakit explain 22 2742 --sign + --model 4b --measure --concept "an alien"
    personakit make "a pirate" --about pirates --model 4b -o pirate.json
    personakit port pirate.json --to 9b -o pirate-9b.json
    personakit twins 22 2742 --sign + --from 4b --to 9b
    personakit build --model 2b

--local runs the same code on this machine instead, on the 0.8B at toy sizes (a check that the kit works, not a
result).
"""
import argparse
import json
import sys

from . import data as D

# The study behind the kit (README.md): 406 Qwen3.5-4B neurons pushed at dose 0.75, judged on 10 identity prompts.
H4_PRIOR = ("in the study, 3 of 406 Qwen3.5-4B neurons broke the model's identity when pushed at dose 0.75 (0.7%), "
            "and neither the axis cosine (AUROC 0.47) nor the logit lens's peakiness (0.77, interval 0.40-0.97) "
            "predicted which: measure it (--measure)")


def call(fn: str, key: str, args: argparse.Namespace, *params, **kw):
    from . import remote as R
    f = getattr(R, fn)
    if args.local:
        return f.local(*params, size="toy", **kw)
    import modal
    with modal.enable_output(), R.app.run():
        return R.wait(f.with_options(gpu=D.gpu_for(key)).spawn(*params, **kw))


def sign_of(s: str) -> int:
    if s not in ("+", "-"):
        raise argparse.ArgumentTypeError("the sign is + or -")
    return 1 if s == "+" else -1


def show_rates(r: dict) -> str:
    parts = [f"{k.replace('_', ' ')} {r[k]:.2f} [{r[k + '_ci95'][0]:.2f}, {r[k + '_ci95'][1]:.2f}]"
             for k in ("persona", "identity_break", "coherent") if k in r]
    return f"{', '.join(parts)} (n {r['n']})"


def print_explain(r: dict):
    print(f"\nL{r['layer']} N{r['neuron']}{'+' if r['sign'] > 0 else '-'} on {r['model']}")
    print(f"  logit lens, up:   {' | '.join(repr(t) for t in r['lens_top'])}")
    print(f"  logit lens, down: {' | '.join(repr(t) for t in r['lens_bottom'])}")
    print(f"  peakiness: the top token sits {r['lens_peakiness']} standard deviations above the vocabulary's mean")
    a = r["axis"]
    print(f"  Assistant Axis: cosine with 'away from the Assistant' {a['cos_away']:+.3f} "
          f"({a['pct'] * 100:.1f}th percentile of its layer)")
    print("  concepts: " + ", ".join(f"{c['concept']} {c['cos']:+.3f} ({c['pct'] * 100:.1f}th pct)" for c in r["concepts"][:4]))
    print("  nearest roles: " + ", ".join(f"{x['role']} {x['cos']:+.3f}" for x in r["roles"]))
    n = r["natural"]
    print(f"  natural range: Pile {n['pile_lo']} to {n['pile_hi']}, chat {n['chat_lo']} to {n['chat_hi']}; "
          f"its largest natural activation this way equals a push of dose {n['dose_of_extreme']}")
    print(f"  identity break: {H4_PRIOR}")
    for m in r.get("measured", []):
        print(f"  dose {m['dose']:<5} {show_rates(m)}")
        if m["dose"]:
            print(f"      e.g. {m['sample'][:160]!r}")


def print_recipe(r: dict):
    if not r["made"]:
        print(f"\nNo coherent cell made {r['model']} speak as {r['name']}. The grid:")
        for g in r["grid"]:
            print(f"  dc {g['dc']:<5} da {g['da']:<5} {show_rates(g)}; everyday coherent {g['everyday_coherent']:.2f}")
        return
    t, c = r["test"], r["cost"]
    print(f"\n{r['name']} on {r['model']}: layer {r['layer']}, concept dose {r['dc']}, axis dose {r['da']} "
          f"(vector norm {sum(x * x for x in r['vector']) ** 0.5:.1f})")
    if "at_source_doses" in r:
        s = r["at_source_doses"]
        print(f"  at {r['ported_from']['model']}'s doses ({r['ported_from']['dc']}, {r['ported_from']['da']}): "
              + (f"{show_rates(s)}; everyday coherent {s['everyday_coherent']:.2f}" if s else "not on the grid"))
    print(f"  development cell: {show_rates(r['dev'])}; everyday coherent {r['dev']['everyday_coherent']:.2f}")
    print(f"  held-out identity prompts, steered:   {show_rates(t['steered'])}")
    print(f"  held-out identity prompts, unsteered: {show_rates(t['unsteered'])}")
    print(f"  everyday prompts, steered:            {show_rates(t['everyday_steered'])}")
    print(f"  capability: MMLU {c['mmlu_unsteered']:.3f} -> {c['mmlu_steered']:.3f} "
          f"(difference interval [{c['mmlu_diff_ci95'][0]:+.3f}, {c['mmlu_diff_ci95'][1]:+.3f}] over {c['mmlu_items']} items); "
          f"KL from the unsteered model {c['kl_per_token']} nats per token on the Pile")
    for s in r["samples"]:
        print(f"  e.g. {s[:160]!r}")


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(prog="personakit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local", action="store_true", help="run on this machine: the 0.8B at toy sizes, as a check")
    sub = p.add_subparsers(dest="cmd", required=True)
    models = sorted(D.MODELS)

    b = sub.add_parser("build", help="build a model's atlas (made on first use otherwise)")
    b.add_argument("--model", default="4b", choices=models)

    e = sub.add_parser("explain", help="what a neuron points at, and with --measure what pushing it does")
    e.add_argument("layer", type=int)
    e.add_argument("neuron", type=int)
    e.add_argument("--sign", type=sign_of, default=1, help="+ or - (the direction pushed)")
    e.add_argument("--model", default="4b", choices=models)
    e.add_argument("--measure", action="store_true", help="push it at doses 0.25-1.5 and judge 10 replies each")
    e.add_argument("--concept", default="", help='with --measure, also judge whether it speaks as this ("an alien")')

    m = sub.add_parser("make", help="a persona recipe: a concept direction plus a push off the Assistant Axis")
    m.add_argument("name", help='the persona, as the judge should read it ("a pirate")')
    m.add_argument("--about", default="", help="the topic its concept direction is read from (default: the name)")
    m.add_argument("--model", default="4b", choices=models)

    t = sub.add_parser("port", help="re-derive a recipe in another size")
    t.add_argument("recipe", help="a recipe JSON from make")
    t.add_argument("--to", required=True, choices=models)

    w = sub.add_parser("twins", help="a neuron's twins in another size, by its direct effect on the shared vocabulary")
    w.add_argument("layer", type=int)
    w.add_argument("neuron", type=int)
    w.add_argument("--sign", type=sign_of, default=1)
    w.add_argument("--from", dest="src", default="4b", choices=models)
    w.add_argument("--to", required=True, choices=models)

    for s in (b, e, m, t, w):
        s.add_argument("-o", "--out", help="write the full result here as JSON")
    a = p.parse_args(argv)
    key = "0.8b" if a.local else getattr(a, "model", None) or getattr(a, "to", None)

    if a.cmd == "build":
        r = call("build", key, a, key)
        print(json.dumps(r, indent=1))
    elif a.cmd == "explain":
        r = call("explain", key, a, key, a.layer, a.neuron, a.sign, a.measure, a.concept)
        print_explain(r)
    elif a.cmd == "make":
        r = call("make", key, a, key, a.name, a.about or a.name)
        print_recipe(r)
    elif a.cmd == "port":
        recipe = json.load(open(a.recipe))
        if not recipe.get("made"):
            sys.exit(f"{a.recipe} holds no recipe: make found no coherent cell for {recipe.get('name')}")
        r = call("port", key, a, recipe, key)
        print_recipe(r)
    else:
        src = "0.8b" if a.local else a.src
        r = call("twins", key, a, src, a.layer, a.neuron, a.sign, key)
        print(f"\nL{a.layer} N{a.neuron}{'+' if a.sign > 0 else '-'} on {src}: {' | '.join(repr(x) for x in r['lens_top'])}")
        for x in r["twins"]:
            print(f"  {key} L{x['layer']} N{x['neuron']}{'+' if x['sign'] > 0 else '-'}: Spearman {x['spearman']:.3f} "
                  f"(Pearson {x['pearson']:.3f}); {' | '.join(repr(y) for y in x['lens_top'])}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
        print(f"\nwrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
