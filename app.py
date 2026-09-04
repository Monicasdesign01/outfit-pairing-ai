"""
Step 8 - the Streamlit app entrypoint. Tabbed/multi-page navigation
(Streamlit's native st.Page/st.navigation mechanism, not hand-rolled
session-state page switching) between the Shop page and the Try It On
Your Clothes page - a "super-app" pattern, the way Swiggy's Instamart
sits inside the main Swiggy app. See Section 4 of the master file.
"""

import streamlit as st

st.set_page_config(page_title="Outfit Pairing AI", layout="wide")

shop_page = st.Page("app_pages/shop.py", title="Shop", default=True)
try_it_on_page = st.Page("app_pages/try_it_on.py", title="Try It On Your Clothes")

pg = st.navigation([shop_page, try_it_on_page])
pg.run()
