"""
Tests for building a whole outfit rather than one pairing at a time.
"""

import pytest

from outfit_builder import build_outfit, cohesion_score, outfit_slots_for
from pairing_rules import BOTTOMS, TOPS


def item(item_id, category, color="black", style="casual", final_score=0.5):
    return {
        "id": item_id,
        "name": f"{color} {category}",
        "category": category,
        "color": color,
        "style": style,
        "final_score": final_score,
    }


CANDIDATES = [
    item("c1", "shirt", color="white", final_score=0.60),
    item("c2", "top", color="red", final_score=0.55),
    item("c3", "jeans", color="navy", final_score=0.58),
    item("c4", "skirt", color="olive", final_score=0.52),
    item("c5", "blazer", color="black", final_score=0.50),
]


def test_a_bottom_needs_a_top_and_a_layer():
    assert outfit_slots_for("jeans") == ["bottom", "blazer"][::-1] or outfit_slots_for("jeans") == ["top", "blazer"]
    assert outfit_slots_for("jeans")[0] == "top"


def test_a_top_needs_a_bottom():
    assert outfit_slots_for("shirt")[0] == "bottom"


def test_a_dress_only_takes_a_layer():
    """A dress is already a complete outfit - adding a skirt to it would
    be the same mistake as pairing jeans with more jeans."""
    assert outfit_slots_for("dress") == ["blazer"]


def test_unknown_category_builds_nothing_rather_than_crashing():
    assert outfit_slots_for("spacesuit") == []
    assert build_outfit("spacesuit", CANDIDATES) == []


def test_outfit_fills_each_slot_with_the_right_kind_of_garment():
    outfit = build_outfit("jeans", CANDIDATES)
    slots = [piece["slot"] for piece in outfit]
    assert slots == ["top", "blazer"]
    assert outfit[0]["category"] in TOPS
    assert outfit[1]["category"] == "blazer"


def test_an_outfit_never_repeats_the_same_item():
    outfit = build_outfit("blazer", CANDIDATES)
    ids = [piece["id"] for piece in outfit]
    assert len(ids) == len(set(ids))


def test_a_slot_with_no_candidates_is_skipped_not_filled_badly():
    """Better to return an incomplete outfit than to put a skirt in the
    blazer slot just to fill it."""
    no_layers = [c for c in CANDIDATES if c["category"] != "blazer"]
    outfit = build_outfit("jeans", no_layers)
    assert [piece["slot"] for piece in outfit] == ["top"]


def test_cohesion_is_zero_for_the_first_pick():
    """Nothing has been chosen yet, so the first piece is decided purely
    on how well it matches the uploaded item."""
    assert cohesion_score(CANDIDATES[0], []) == 0.0


def test_cohesion_rewards_pieces_that_work_with_what_is_already_chosen():
    chosen = [item("x", "shirt", color="white", style="fitted")]
    goes_with = item("y", "blazer", color="navy", style="loose")   # neutral + balanced
    clashes = item("z", "blazer", color="purple", style="fitted")  # no relationship
    assert cohesion_score(goes_with, chosen) > cohesion_score(clashes, chosen)


def test_cohesion_changes_which_item_gets_picked():
    """The whole point of scoring against the outfit rather than only the
    upload: a slightly lower-ranked piece should win if it holds the
    outfit together better."""
    chosen_top = item("t1", "shirt", color="white", style="fitted", final_score=0.60)
    candidates = [
        chosen_top,
        item("b_high", "blazer", color="purple", style="fitted", final_score=0.50),
        item("b_cohesive", "blazer", color="navy", style="loose", final_score=0.49),
    ]
    outfit = build_outfit("jeans", candidates)
    layer = [piece for piece in outfit if piece["slot"] == "blazer"][0]
    assert layer["id"] == "b_cohesive"


@pytest.mark.parametrize("category", sorted(TOPS | BOTTOMS))
def test_every_wearable_category_produces_a_plan(category):
    assert outfit_slots_for(category), f"{category} has no outfit plan"
