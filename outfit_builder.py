"""
Builds a complete outfit around the uploaded item, rather than returning
one complementary piece at a time.

The matching engine answers "what goes with this?" - a ranked list of
individual items. This answers the question a customer actually has:
"what do I wear with this?", which usually needs more than one piece. If
they upload jeans, a shirt alone isn't an outfit; a shirt plus a layer
is.

Deliberately built on top of the existing ranked candidates rather than a
second retrieval pass: the ranking work is already done, so this is a
selection problem, not a search problem. The only new idea is cohesion -
each additional piece is scored against the pieces already chosen, not
just against the uploaded item, so the outfit holds together as a whole
instead of being three things that each happen to match the jeans.
"""

from pairing_rules import (
    BOTTOMS,
    COLOR_WEIGHT,
    STYLE_WEIGHT,
    TOPS,
    color_score,
    silhouette_score,
)

LAYER = "blazer"


def outfit_slots_for(category):
    """
    Which roles still need filling to make a complete look, in the order
    they should be chosen (the essential piece first, the optional layer
    last). A dress is already a complete outfit, so it only takes a layer.
    """
    if category == "dress":
        return [LAYER]
    if category in TOPS:
        return ["bottom", LAYER]
    if category in BOTTOMS:
        return ["top", LAYER]
    if category == LAYER:
        return ["top", "bottom"]
    return []


def _fills_slot(item, slot):
    if slot == "top":
        return item["category"] in TOPS
    if slot == "bottom":
        return item["category"] in BOTTOMS
    return item["category"] == slot


def cohesion_score(item, already_chosen):
    """
    How well an item sits with the pieces already in the outfit, on the
    same 0-1 scale as the matching engine's own signals. Zero when nothing
    has been chosen yet, so the first pick is decided purely by how well
    it matches the uploaded item.
    """
    if not already_chosen:
        return 0.0

    total = 0.0
    for chosen in already_chosen:
        total += COLOR_WEIGHT * (color_score(item["color"], chosen["color"]) / 2)
        total += STYLE_WEIGHT * (silhouette_score(item["style"], chosen["style"]) / 2)
    return total / len(already_chosen)


def build_outfit(uploaded_category, candidates):
    """
    Picks at most one item per outfit slot from the already-ranked
    candidates. Greedy on purpose: with a handful of slots and a small
    catalog, searching every combination would cost more to explain than
    it would gain, and a greedy pick is easy to justify to a customer.

    Returns a list of chosen items, each with the slot it fills and the
    scores behind it. Slots with no available candidate are skipped
    rather than filled badly - an incomplete outfit is better than a
    wrong one.
    """
    chosen = []
    used_ids = set()

    for slot in outfit_slots_for(uploaded_category):
        pool = [
            item for item in candidates
            if _fills_slot(item, slot) and item["id"] not in used_ids
        ]
        if not pool:
            continue

        best = max(pool, key=lambda item: item["final_score"] + cohesion_score(item, chosen))
        picked = {
            **best,
            "slot": slot,
            "cohesion": cohesion_score(best, chosen),
            "outfit_score": best["final_score"] + cohesion_score(best, chosen),
        }
        chosen.append(picked)
        used_ids.add(best["id"])

    return chosen
