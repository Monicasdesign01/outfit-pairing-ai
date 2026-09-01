# This script automatically finds every test photo in the test_images
# folder and runs THREE checks on each one: color, garment type, and style.
# No filenames are typed by hand anywhere — it discovers them itself.

import os
from classify_garment import classify_garment
from color_detector import get_dominant_color, closest_color_name

# The list of possible "types" CLIP will choose between for every photo
TYPE_LABELS = [
    "a photo of jeans",
    "a photo of a dress",
    "a photo of a jacket",
    "a photo of a shirt",
    "a photo of a t-shirt",
    "a photo of a skirt",
    "a photo of a kurta",
    "a photo of a hooded zip-up sweatshirt",
]

# The list of possible "styles" CLIP will choose between for every photo
STYLE_LABELS = [
    "a photo of fitted clothing",
    "a photo of loose clothing",
    "a photo of casual clothing",
    "a photo of formal clothing",
    "a photo of edgy alternative streetwear",
]


def run_full_check(folder="test_images"):
    # Ask Windows what files actually exist in this folder right now.
    # This is the key line — nothing is hardcoded, it discovers files itself.
    all_files = os.listdir(folder)

    # Keep only actual photo files, in case anything else ended up in there
    photo_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Put them in a predictable order (test_01, test_02, test_03...)
    photo_files.sort()

    # Now loop through every single photo we found, one at a time
    for filename in photo_files:
        # Build the full path, e.g. "test_images\test_03_jeans.jpg"
        path = os.path.join(folder, filename)

        # --- Check 1: what color is it? ---
        rgb = analyse_color(path)
        color_name = closest_color_name(rgb)

        # --- Check 2: what type of garment is it? ---
        type_results = classify_garment(path, TYPE_LABELS)
        top_type, type_confidence = type_results[0]  # best guess is first

        # --- Check 3: what style is it? ---
        style_results = classify_garment(path, STYLE_LABELS)
        top_style, style_confidence = style_results[0]

        # Print all three results together for this one photo
        print(f"{filename}")
        print(f"   Color : {color_name}  {rgb}")
        print(f"   Type  : {top_type}  ({type_confidence*100:.1f}%)")
        print(f"   Style : {top_style}  ({style_confidence*100:.1f}%)")
        print()  # blank line before the next photo, for readability


# This only runs if you execute this file directly (not if imported elsewhere)
if __name__ == "__main__":
    run_full_check()