"""
Shop page - browsable catalog, cosmetic only. No cart, no accounts, no
checkout logic. Exists to give the AI feature a realistic storefront
around it.
"""

import streamlit as st

from shop_utils import load_catalog_items, catalog_image_path, build_upi_link

st.title("Shop")
st.caption("Vintage, celebrity, and new clothing.")

catalog = load_catalog_items()

# A storefront with no way to narrow 49 items down is a wall of photos.
# These two controls are pure display-layer filtering over catalog.json -
# no search index or backend involved, deliberately.
filter_col, sort_col = st.columns([3, 1])
with filter_col:
    categories = sorted({item["category"] for item in catalog})
    selected = st.multiselect("Filter by category", options=categories, placeholder="All categories")
with sort_col:
    sort_choice = st.selectbox("Sort by", ["Featured", "Price: low to high", "Price: high to low"])

visible = [item for item in catalog if not selected or item["category"] in selected]
if sort_choice == "Price: low to high":
    visible = sorted(visible, key=lambda i: i["price"])
elif sort_choice == "Price: high to low":
    visible = sorted(visible, key=lambda i: i["price"], reverse=True)

st.caption(f"{len(visible)} of {len(catalog)} items")
st.divider()

if not visible:
    st.info("No items in that category yet.")

COLUMNS = 3
rows = [visible[i:i + COLUMNS] for i in range(0, len(visible), COLUMNS)]

for row in rows:
    cols = st.columns(COLUMNS)
    for col, item in zip(cols, row):
        with col:
            with st.container(border=True):
                st.image(catalog_image_path(item), width="stretch")
                st.markdown(f"**{item['name']}**")
                # Grouped digits read as a price; a bare 1499 reads as an ID.
                st.markdown(f"### ₹{item['price']:,}")
                st.caption(f"{item['category'].title()} · {item['color'].title()}")
                st.link_button("Buy", build_upi_link(item), width="stretch")
