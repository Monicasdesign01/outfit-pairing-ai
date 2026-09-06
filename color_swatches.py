"""
A representative RGB swatch for every colour name closest_color_name()
can return (color_detector.ALL_COLOR_NAMES) - used only for rendering
(the 3D mannequin preview, Step 9), never for detection. Reuses the same
reference RGB values color_detector.py already tuned for detection, so
the mannequin's colours stay consistent with what the app actually
detected/confirmed - no separate palette to drift out of sync.
"""

from color_detector import NEUTRAL_REFERENCE, CHROMATIC_FAMILIES, ALL_COLOR_NAMES

# "beige" has no dict entry in color_detector.py (it's a bare string
# returned inline - see ALL_COLOR_NAMES's own comment on why), so it
# needs its own swatch value here, picked to visually match its name.
BEIGE_SWATCH = (222, 196, 160)

COLOR_SWATCHES = {
    **NEUTRAL_REFERENCE,
    **{name: anchors[0] for name, anchors in CHROMATIC_FAMILIES.items()},
    "beige": BEIGE_SWATCH,
}

# Defensive, not decorative: if a new colour is ever added to
# ALL_COLOR_NAMES without a matching swatch here, fail loudly at import
# time rather than crashing later mid-render with a confusing KeyError.
assert set(COLOR_SWATCHES) == set(ALL_COLOR_NAMES), (
    f"color_swatches.py is out of sync with color_detector.ALL_COLOR_NAMES: "
    f"missing {set(ALL_COLOR_NAMES) - set(COLOR_SWATCHES)}, "
    f"extra {set(COLOR_SWATCHES) - set(ALL_COLOR_NAMES)}"
)


def to_hex(color_name):
    r, g, b = COLOR_SWATCHES[color_name]
    return f"#{r:02x}{g:02x}{b:02x}"
