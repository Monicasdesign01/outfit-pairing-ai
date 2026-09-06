"""
Tests for the catalog helpers and the UPI payment link. The link is a
real payment deep link, so a malformed one is worse than a broken button:
it could open a payment app with the wrong amount.
"""

import json
from urllib.parse import parse_qs, urlparse

from shop_utils import MERCHANT_UPI_ID, build_upi_link, catalog_image_path, load_catalog_items

ITEM = {"filename": "test_12_shirt.jpg", "name": "Everyday Cotton Shirt", "price": 1499}


def parse_upi(link):
    parsed = urlparse(link)
    return parsed.scheme, parse_qs(parsed.query)


def test_upi_link_carries_the_right_payee_and_amount():
    scheme, params = parse_upi(build_upi_link(ITEM))
    assert scheme == "upi"
    assert params["pa"] == [MERCHANT_UPI_ID]
    assert params["am"] == ["1499"]
    assert params["cu"] == ["INR"]


def test_item_names_with_spaces_are_url_encoded():
    """An unencoded space would truncate the transaction note, or make
    some UPI apps reject the link outright."""
    link = build_upi_link(ITEM)
    assert " " not in link
    _, params = parse_upi(link)
    assert params["tn"] == ["Everyday Cotton Shirt"]


def test_amount_matches_the_catalog_price_exactly():
    """Rounding or reformatting the price here would mean charging a
    different amount than the one displayed next to the button."""
    for price in [999, 1099, 2499, 3199]:
        _, params = parse_upi(build_upi_link({**ITEM, "price": price}))
        assert params["am"] == [str(price)]


def test_catalog_image_path_points_into_the_catalog_folder():
    assert catalog_image_path(ITEM) == "catalog_images/test_12_shirt.jpg"


def test_every_catalog_item_has_the_fields_the_app_reads():
    """The Shop page and the matching engine both index into these keys
    directly, so a missing one is a crash rather than a bad result."""
    catalog = load_catalog_items()
    assert catalog, "catalog is empty"
    ids = set()
    for item in catalog:
        for field in ("id", "filename", "name", "category", "price", "color", "style"):
            assert field in item, f"{item.get('filename')} is missing {field}"
        assert isinstance(item["price"], int)
        assert item["id"] not in ids, f"duplicate id {item['id']}"
        ids.add(item["id"])


def test_every_catalog_image_file_actually_exists():
    """A missing image file breaks the Shop page at render time, which is
    the first thing anyone sees."""
    import os

    for item in load_catalog_items():
        assert os.path.exists(catalog_image_path(item)), f"missing image for {item['id']}"
