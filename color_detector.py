import cv2
import numpy as np

# A small reference list of common color names and their RGB values.
# This is what we compare against to name any color we detect.
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
}

def closest_color_name(rgb):
    r, g, b = rgb
    best_name = None
    best_distance = float("inf")

    for name, (cr, cg, cb) in COLOR_NAMES.items():
        # Simple distance formula: how far apart are two colors in 3D space?
        # Same idea as finding distance between two points on a map.
        distance = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if distance < best_distance:
            best_distance = distance
            best_name = name

    return best_name


def get_dominant_color(image_path, k=3):
    # IMREAD_UNCHANGED keeps the 4th channel (transparency), which
    # normal imread() throws away
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (150, 150))

    # If this image has 4 channels (Red, Green, Blue, Alpha/transparency),
    # split off the alpha channel so we can use it to filter pixels
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
        # Keep only pixels where alpha > 0 — meaning "actually visible,
        # not transparent." This throws out every background pixel rembg
        # already removed for us.
        pixels = pixels[alpha_flat > 0]

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