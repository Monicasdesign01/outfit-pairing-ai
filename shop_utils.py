"""Shared helpers for the Shop page and Try It On page's match results."""

import json
from urllib.parse import quote

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
