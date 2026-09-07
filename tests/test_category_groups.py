"""
Tests for the coarse category groups. The app falls back to these when it
isn't confident about the exact garment type, so a category that belongs
to no group would leave the app with nothing to say.
"""

import pytest

from pairing_rules import BOTTOMS, CATEGORY_GROUPS, CATEGORY_LABELS, TOPS, category_group


@pytest.mark.parametrize("category", sorted(CATEGORY_LABELS))
def test_every_category_belongs_to_a_group(category):
    assert category_group(category) is not None, f"{category} has no coarse group"


@pytest.mark.parametrize("category", sorted(TOPS))
def test_tops_group_as_top(category):
    assert category_group(category) == "top"


@pytest.mark.parametrize("category", sorted(BOTTOMS))
def test_bottoms_group_as_bottom(category):
    assert category_group(category) == "bottom"


def test_a_dress_is_its_own_group():
    assert category_group("dress") == "dress"


def test_a_blazer_groups_with_tops():
    """It's outerwear, but it goes on the upper body - and the groups only
    exist to tell the customer roughly what kind of garment this is."""
    assert category_group("blazer") == "top"


def test_unknown_category_has_no_group_rather_than_guessing():
    assert category_group("spacesuit") is None


def test_groups_do_not_overlap():
    """A category in two groups would make the summed group confidence
    exceed 100%, which would be visible nonsense in the UI."""
    seen = set()
    for members in CATEGORY_GROUPS.values():
        assert not (seen & members), f"category in two groups: {seen & members}"
        seen |= members
