import cv2
import numpy as np

# Reference colors we compare against to turn raw RGB numbers into
# human-readable names like "red" or "olive". "orange" added 2026-09-05 -
# it was missing entirely, so any genuinely orange garment was forced
# into whichever unrelated color happened to be numerically closest
# (measured: orange RGB (219,119,102) was being called "pink").
COLOR_NAMES = {
    "red": (200, 30, 40),
    "navy": (20, 30, 80),
    "blue": (50, 100, 200),
    "olive": (100, 100, 40),
    "green": (40, 130, 60),
    "black": (20, 20, 20),
    "white": (240, 240, 240),
    "gray": (150, 150, 150),
    "cream": (230, 220, 190),
    "brown": (110, 70, 40),
    "pink": (230, 150, 180),
    "yellow": (230, 220, 60),
    "orange": (230, 126, 34),
}

# How "colorful" (vs. neutral gray/black/white) an RGB value is, using
# the gap between its brightest and darkest channel - a genuinely neutral
# color has R, G, and B all close together, regardless of how bright it
# is. Below this, we skip hue-matching entirely and go by brightness -
# fixes a real bug found 2026-09-05: a plain neutral gray, RGB (81,81,81),
# was being called "olive" because "gray" only has one reference point
# (150,150,150, a fairly light gray), so a darker true-gray ended up
# numerically closer to an unrelated hue like olive purely by chance of
# where that one point sits in 3D color space.
NEUTRAL_CHANNEL_SPREAD = 15


def closest_color_name(rgb):
    r, g, b = rgb

    if max(r, g, b) - min(r, g, b) < NEUTRAL_CHANNEL_SPREAD:
        brightness = (r + g + b) / 3
        if brightness < 60:
            return "black"
        if brightness < 200:
            return "gray"
        return "white"

    # Finds whichever reference color above is numerically closest to
    # the given RGB value, using simple distance math (like distance
    # between two points on a map, just in 3D using R, G, B).
    best_name = None
    best_distance = float("inf")

    for name, (cr, cg, cb) in COLOR_NAMES.items():
        distance = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if distance < best_distance:
            best_distance = distance
            best_name = name

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
