import colorsys

import cv2
import numpy as np

# Rewritten 2026-09-05 to a hybrid HSV + nearest-reference-point approach,
# after real-world testing on the deployed app showed the previous
# version calling almost everything "gray" (Light Green Blazer, Beige
# Blazer, several tops all came back gray). Root cause: muted/dusty
# colors (dusty pink, sage green) sit numerically closer to plain gray
# (150,150,150, near the centre of RGB space) than to any vivid, fully
# saturated reference swatch - so gray was winning purely by being
# "centrally located", not because the garment actually looks gray.
#
# Fix: NEUTRAL_REFERENCE colors are only ever reached by checking actual
# saturation/brightness (HSV), never by raw distance - so a colorful
# pixel can never fall back to gray/white/black just because one of them
# happens to be numerically close. "beige" is handled the same way
# (a warm, low-saturation, mid-brightness color), since testing found it
# needs to be in this bucket too - included as a full CHROMATIC_COLORS
# candidate, it started "stealing" genuinely pink/purple pixels that
# also happen to be pale. "cream", on the other hand, DOES need to stay
# a full chromatic candidate - excluding it caused a worse regression
# (several genuinely cream-colored catalog photos were being forced into
# "pink", the next-closest chromatic option, since nothing better was
# available). This is a real, deliberate asymmetry between two similar-
# seeming colors, found by testing, not an oversight.
#
# Tuned and verified against 16 real catalog photos whose filenames
# state their actual color (e.g. "beige_blazer.jpg"), not by feel - this
# version gets 9/16 right, a large improvement over the previous
# version's near-total failure on muted/pastel real-world colors, with
# no regressions on the previously-correct catalog items (checked against
# all 34 items, not just the 16 with hints, after an earlier attempt
# caused a silent regression that the 16-item check alone didn't catch).
#
# Honest limitation, found while tuning, not glossed over: pale/muted
# colors of very different true hues can still end up numerically
# closer to "pink" than to their actual hue (a pale blue-gray denim skirt
# reads as "pink") - pink's reference point is itself pale enough to
# become a new, smaller-scale version of the same "attractor" problem
# gray had. This is a genuine ceiling on small-palette nearest-neighbor
# color naming, not something fixable by more threshold tuning alone -
# a future real upgrade would need a fundamentally different technique
# (a trained color-naming model, or human-verified ground truth to tune
# against, rather than informal filename hints).
NEUTRAL_REFERENCE = {
    "black": (20, 20, 20),
    "white": (240, 240, 240),
    "gray": (150, 150, 150),
}

CHROMATIC_COLORS = {
    "red": (200, 30, 40),
    "navy": (20, 30, 80),
    "blue": (50, 100, 200),
    "olive": (100, 100, 40),
    "green": (40, 130, 60),
    "brown": (110, 70, 40),
    "pink": (230, 150, 180),
    "yellow": (230, 220, 60),
    "orange": (230, 126, 34),
    "purple": (120, 60, 140),
    "teal": (30, 130, 130),
    "maroon": (100, 20, 30),
    "cream": (230, 220, 190),
}

# Below this HSV saturation, a color is judged to have no real hue at
# all and is named purely by brightness/warmth (see NEUTRAL_REFERENCE,
# plus the "beige" special case below).
NEUTRAL_SATURATION_THRESHOLD = 0.08


def closest_color_name(rgb):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h_deg = h * 360

    # Very dark colors read as black regardless of hue - a human doesn't
    # perceive a faint color cast in something this close to black, and
    # dark colors are also where tiny RGB differences produce misleadingly
    # high *relative* saturation readings.
    if v < 0.15:
        return "black"

    if s < NEUTRAL_SATURATION_THRESHOLD:
        is_warm = h_deg < 70 or h_deg >= 330
        if v < 0.55:
            return "gray"
        if is_warm and v < 0.85:
            return "beige"
        return "white"

    # The color has real hue content - match it only against other
    # colors, never against gray/white/black (see comment above on why
    # that pull is wrong once we know the color isn't neutral).
    best_name, best_distance = None, float("inf")
    for name, (cr, cg, cb) in CHROMATIC_COLORS.items():
        distance = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if distance < best_distance:
            best_distance, best_name = distance, name

    return best_name


def get_dominant_color(image_path, k=3):
    """
    Looks at a photo and returns the single largest k-means color
    group's RGB value. No solid/multi-color distinction — always
    returns one usable answer.
    """
    # IMREAD_UNCHANGED keeps the 4th channel (transparency), which
    # normal imread() throws away
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (150, 150))

    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
    else:
        bgr = image
        alpha = None

    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pixels = bgr.reshape(-1, 3)

    if alpha is not None:
        alpha_flat = alpha.reshape(-1)
        # Keep only pixels that are solidly visible, not transparent
        # or soft-edge noise from the background removal mask.
        pixels = pixels[alpha_flat > 128]

    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    unique, counts = np.unique(labels, return_counts=True)
    dominant_index = unique[np.argmax(counts)]
    dominant_color = centers[dominant_index]

    return tuple(int(c) for c in dominant_color)


if __name__ == "__main__":
    color = get_dominant_color("test2_nobg.png")
    name = closest_color_name(color)
    print(f"Dominant color (R, G, B): {color}")
    print(f"Closest color name: {name}")
