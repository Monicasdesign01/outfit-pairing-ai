"""Shared helpers for the Shop page and Try It On page's match results."""

import json
import os
from urllib.parse import quote

CATALOG_JSON = "catalog.json"
CATALOG_IMAGES_DIR = "catalog_images"
CATALOG_NOBG_DIR = os.path.join(CATALOG_IMAGES_DIR, "nobg")

# Placeholder - Monica's real UPI VPA (e.g. "monica@okhdfcbank") needs to
# replace this before the Buy button is used for a real transaction.
MERCHANT_UPI_ID = "yourupi@upi"


def load_catalog_items():
    """Plain catalog.json content - no embeddings, this is just for display."""
    with open(CATALOG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def catalog_image_path(item):
    return f"{CATALOG_IMAGES_DIR}/{item['filename']}"


def catalog_nobg_image_path(item):
    """
    Path to the background-removed cutout of a catalog item, built once by
    build_catalog_embeddings.py and cached on disk (catalog_images/nobg/).
    Used for the mannequin preview (Step 9) so it shows the actual garment,
    not the full studio photo (which usually has a model wearing it, not
    just the item on its own). Returns None if it isn't there yet - the
    mannequin falls back to a plain colour for that item rather than
    crashing.
    """
    nobg_filename = os.path.splitext(item["filename"])[0] + "_nobg.png"
    path = os.path.join(CATALOG_NOBG_DIR, nobg_filename)
    return path if os.path.exists(path) else None


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
