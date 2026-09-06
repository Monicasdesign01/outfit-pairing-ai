"""
Tests for colour naming. These deliberately cover the exact failure that
took the live app down once: closest_color_name() can return "beige",
which was not a key in either reference dict, so a dropdown built from
those dicts crashed with a ValueError the first time a real upload was
detected as beige.
"""

import colorsys

import pytest

from color_detector import (
    ALL_COLOR_NAMES,
    CHROMATIC_FAMILIES,
    NEUTRAL_REFERENCE,
    NEUTRAL_SATURATION_THRESHOLD,
    closest_color_name,
)


def test_every_reference_colour_names_itself():
    """A reference swatch should be named as its own family - if it isn't,
    some other family is stealing it and the palette needs rebalancing."""
    for name, anchors in CHROMATIC_FAMILIES.items():
        for anchor in anchors:
            assert closest_color_name(anchor) == name, f"{anchor} should be {name}"


def test_obvious_neutrals():
    assert closest_color_name((0, 0, 0)) == "black"
    assert closest_color_name((255, 255, 255)) == "white"
    assert closest_color_name((150, 150, 150)) == "gray"


def test_beige_is_reachable_and_listed():
    """The regression test for the production crash: beige is a real
    return value, so it must also appear in ALL_COLOR_NAMES."""
    beige = closest_color_name((200, 195, 185))  # desaturated but genuinely warm
    assert beige == "beige"
    assert "beige" in ALL_COLOR_NAMES


def test_a_pure_grey_is_never_called_beige():
    """Beige is a warm tone. A neutral grey has no warmth at all, but it
    reports hue 0 (which looks warm numerically) - that mistake labelled
    every mid-tone grey as beige until a test caught it."""
    for level in (100, 130, 150, 180, 200):
        assert closest_color_name((level, level, level)) in {"gray", "white"}


def test_all_color_names_covers_every_possible_return_value():
    """Exhaustive-ish guarantee that nothing can be returned that a caller
    building a dropdown from ALL_COLOR_NAMES wouldn't know about."""
    import random

    random.seed(0)
    for _ in range(20000):
        rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        assert closest_color_name(rgb) in ALL_COLOR_NAMES


def test_saturated_colours_never_fall_back_to_a_neutral():
    """The bug that made almost everything 'gray': muted colours sit near
    the centre of RGB space, so plain distance matching pulled them to the
    gray reference. Anything with real hue must resolve to a hue family."""
    saturated_samples = [
        (200, 30, 40),
        (50, 100, 200),
        (40, 130, 60),
        (230, 126, 34),
        (120, 60, 140),
    ]
    for rgb in saturated_samples:
        _, s, v = colorsys.rgb_to_hsv(*[c / 255 for c in rgb])
        assert s >= NEUTRAL_SATURATION_THRESHOLD and v >= 0.15, "sample is not actually saturated"
        assert closest_color_name(rgb) not in NEUTRAL_REFERENCE


@pytest.mark.parametrize("rgb", [(0, 0, 0), (255, 255, 255), (128, 0, 255), (7, 200, 3)])
def test_always_returns_a_string(rgb):
    assert isinstance(closest_color_name(rgb), str)
