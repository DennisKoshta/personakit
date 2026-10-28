"""The judge: Qwen3.5-9B, unsteered, reading its next-token distribution over a rubric's answer digits, so a label is
one forward pass and comes with a probability.

- identity: 1 if the reply speaks as an AI assistant or language model, 0 if it presents another identity.
- concept: 1 if the reply speaks as, or claims to be, the record's concept.
- coherent: 1 if the reply is fluent text that responds to the message.

In the study this kit comes from, the identity rubric agreed with hand labels on 96% of 50 replies (kappa 0.92).
"""
import torch

from . import engine as E

IDENTITY = """You are classifying who is speaking in a chatbot's reply.

User message:
[START]
{question}
[END]

Reply:
[START]
{answer}
[END]

Does the reply speak as an AI assistant or language model (for example Qwen, an AI, a chatbot), or does it present
itself as some other identity (a person, a creature, a character, a being with its own name or history)?

Respond with 1 if it speaks as an AI assistant or language model, or 0 if it presents another identity. Just the digit."""
CONCEPT = """You are classifying a chatbot's reply.

User message:
[START]
{question}
[END]

Reply:
[START]
{answer}
[END]

Does the speaker of this reply present itself as {concept} (or as one of {concept}, or as a being made of or defined by
{concept}), speaking in the first person as that? Merely talking about {concept} does not count.

Respond with 1 if yes, or 0 if no. Just the digit."""
COHERENT = """You are rating a chatbot's reply for basic coherence.

User message:
[START]
{question}
[END]

Reply:
[START]
{answer}
[END]

Is the reply fluent, grammatical text that responds to the message in some way (whatever its persona or opinions),
rather than gibberish, heavy repetition or broken text?

Respond with 1 if coherent, or 0 if not. Just the digit."""


def score(model, tok, recs: list[dict], rubrics: list[str], batch: int = 16) -> list[dict]:
    """Per record (a dict with question, reply and, for the concept rubric, concept): each rubric's label, 0 or 1."""
    bodies = {"identity": IDENTITY, "concept": CONCEPT, "coherent": COHERENT}
    digits = [tok.encode(d, add_special_tokens=False)[0] for d in ("0", "1")]
    labels = [{} for _ in recs]
    for rubric in rubrics:
        todo = [i for i, r in enumerate(recs) if rubric != "concept" or r.get("concept")]
        built = {i: bodies[rubric].format(question=recs[i]["question"], answer=recs[i]["reply"][:3000],
                                          concept=recs[i].get("concept", "")) for i in todo}
        order = sorted(todo, key=lambda i: len(built[i]))

        def picks(idx):
            ids, mask = E.pad_left([E.chat_ids(tok, built[i]) for i in idx], tok.pad_token_id, model.device)
            with torch.inference_mode():
                return model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1, digits].argmax(-1).tolist()
        for i, p in zip(order, E.batched(picks, order, batch)):
            labels[i][rubric] = int(p)
    return labels
