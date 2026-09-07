"""Shared helpers for the Shop page and Try It On page's match results."""

import json
from urllib.parse import quote

from PIL import Image

CATALOG_JSON = "catalog.json"
CATALOG_IMAGES_DIR = "catalog_images"

# Placeholder - Monica's real UPI VPA (e.g. "monica@okhdfcbank") needs to
# replace this before the Buy button is used for a real transaction.
MERCHANT_UPI_ID = "yourupi@upi"


def load_catalog_items():
    """Plain catalog.json content - no embeddings, this is just for display."""
    with open(CATALOG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def catalog_image_path(item):
    return f"{CATALOG_IMAGES_DIR}/{item['filename']}"


# Catalog photos come in whatever shape they were shot in, so a grid of
# them has ragged card heights and looks unfinished. Retailers solve this
# by cropping every product image to one tile shape; 3:4 portrait suits
# clothing, which is taller than it is wide.
TILE_ASPECT = (3, 4)


def tile_image(path, aspect=TILE_ASPECT):
    """
    Centre-crops a photo to a consistent aspect ratio, taking the largest
    region that fits so nothing is stretched. Crops rather than pads
    because a letterboxed product photo looks like a mistake, while a
    slightly tighter crop just looks like a considered one.
    """
    image = Image.open(path).convert("RGB")
    width, height = image.size
    target_ratio = aspect[0] / aspect[1]

    if width / height > target_ratio:      # too wide - trim the sides
        new_width = round(height * target_ratio)
        left = (width - new_width) // 2
        box = (left, 0, left + new_width, height)
    else:                                   # too tall - trim top and bottom
        new_height = round(width / target_ratio)
        top = (height - new_height) // 2
        box = (0, top, width, top + new_height)

    return image.crop(box)


def build_upi_link(item):
    """A upi://pay deep link pre-filled with the item's name and price.
    Opens the customer's own UPI app to complete payment manually -
    there's no payment gateway or order tracking behind this."""
    params = (
        f"pa={MERCHANT_UPI_ID}"
        f"&pn={quote('Outfit Pairing AI')}"
        f"&am={item['price']}"
        f"&cu=INR"
        f"&tn={quote(item['name'])}"
    )
    return f"upi://pay?{params}"
