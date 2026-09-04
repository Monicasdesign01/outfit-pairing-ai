"""
Shop page - browsable catalog, cosmetic only. No cart, no accounts, no
checkout logic. Exists to give the AI feature a realistic storefront
around it (Section 4 of outfit-pairing-ai-MASTER.md).
"""

import streamlit as st

from shop_utils import load_catalog_items, catalog_image_path, build_upi_link

st.title("Shop")
st.caption("Vintage, celebrity, and new clothing. Browse the catalog below.")

catalog = load_catalog_items()

COLUMNS = 3
rows = [catalog[i:i + COLUMNS] for i in range(0, len(catalog), COLUMNS)]

for row in rows:
    cols = st.columns(COLUMNS)
    for col, item in zip(cols, row):
        with col:
            st.image(catalog_image_path(item), use_container_width=True)
            st.markdown(f"**{item['name']}**")
            st.write(f"₹{item['price']}")
            st.link_button("Buy", build_upi_link(item), use_container_width=True)
