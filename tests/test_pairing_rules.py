"""
Tests for the pairing rules - the part of the project that encodes the
actual idea ("items that look similar are not the same as items that go
together"), so it's the part most worth protecting with tests.
"""

import pytest

from pairing_rules import (
    describe_category,
    BOTTOMS,
    CATEGORY_LABELS,
    NEUTRAL_COLORS,
    TOPS,
    color_score,
    explain_color_relationship,
    explain_silhouette_relationship,
    get_paired_categories,
    silhouette_score,
)


@pytest.mark.parametrize("bottom", sorted(BOTTOMS))
def test_a_bottom_never_pairs_with_another_bottom(bottom):
    """The whole point of the filter step - jeans should not be suggested
    alongside more jeans, however visually similar they are."""
    paired = get_paired_categories(bottom)
    assert paired & BOTTOMS == set()
    assert TOPS <= paired


@pytest.mark.parametrize("top", sorted(TOPS))
def test_a_top_never_pairs_with_another_top(top):
    paired = get_paired_categories(top)
    assert paired & TOPS == set()
    assert BOTTOMS <= paired


def test_a_dress_is_already_a_complete_outfit():
    """A dress only takes a layer over it, not a separate top or bottom."""
    assert get_paired_categories("dress") == {"blazer"}


def test_every_category_label_is_a_real_catalog_category():
    """CATEGORY_LABELS doubles as the CLIP prompt list and the category
    vocabulary, so a prompt with no matching category would silently
    produce an unusable classification."""
    for category in CATEGORY_LABELS:
        assert category.islower()
        assert get_paired_categories(category), f"{category} pairs with nothing"


def test_unknown_category_pairs_with_nothing_rather_than_crashing():
    assert get_paired_categories("spacesuit") == set()


def test_colour_scoring_is_symmetric():
    """Pairing red with green must score the same as green with red - an
    asymmetry here would make rankings depend on upload order."""
    colors = ["red", "green", "blue", "navy", "cream", "olive", "pink", "yellow"]
    for a in colors:
        for b in colors:
            assert color_score(a, b) == color_score(b, a)
            assert explain_color_relationship(a, b) == explain_color_relationship(b, a)


def test_complementary_beats_unrelated():
    assert color_score("blue", "yellow") > color_score("purple", "teal")


@pytest.mark.parametrize("neutral", sorted(NEUTRAL_COLORS))
def test_neutrals_pair_safely_with_anything(neutral):
    for other in ["red", "green", "orange", "purple", "teal"]:
        assert color_score(neutral, other) >= 1


def test_fitted_and_loose_balance_each_other():
    assert silhouette_score("fitted", "loose") > silhouette_score("fitted", "fitted")
    assert explain_silhouette_relationship("fitted", "loose") == "balanced"
    assert explain_silhouette_relationship("casual", "casual") == "matching"


def test_scores_stay_within_the_range_the_reranker_assumes():
    """matching_engine divides both scores by 2 to normalise them, which
    is only correct while 2 really is the maximum."""
    values = ["red", "green", "blue", "navy", "cream", "olive", "pink", "yellow", "black"]
    assert max(color_score(a, b) for a in values for b in values) == 2
    styles = ["fitted", "loose", "casual", "formal", "edgy"]
    assert max(silhouette_score(a, b) for a in styles for b in styles) == 2


@pytest.mark.parametrize("category,expected", [
    ("jeans", "jeans"), ("pants", "pants"), ("shorts", "shorts"),
    ("skirt", "a skirt"), ("dress", "a dress"), ("top", "a top"),
])
def test_category_descriptions_use_the_right_article(category, expected):
    """Copy that says "a jeans" undermines an otherwise polished app."""
    assert describe_category(category) == expected
