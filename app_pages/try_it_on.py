"""
Try It On Your Clothes - the AI feature. Upload a photo, run the full
pipeline (background removal -> color/type/style detection -> filter ->
retrieve -> re-rank -> explanation), show ranked matches with a category
filter (Section 4 of outfit-pairing-ai-MASTER.md).
"""

import os
import tempfile
import time

import streamlit as st

from matching_engine import find_matches
from explanation import get_explanation, build_template_explanation
from shop_utils import catalog_image_path, build_upi_link

# Real constraint, not a workaround being hidden: Gemini's free tier
# rate-limits at ~15 requests/minute, and firing off a live call per
# retrieved match (up to 9-14 of them for some uploads) was taking
# several minutes per upload once the SDK's automatic retry/backoff
# kicked in (see outfit-pairing-ai-MASTER.md Section 11, Step 8). Only
# the top-scoring matches - the ones actually most likely to matter to
# the customer - get a live explanation; the rest use the template
# fallback outright, without even attempting a call that would likely
# just be rate-limited anyway.
LIVE_EXPLANATION_LIMIT = 3

st.title("Try It On Your Clothes")
st.caption("Upload a photo of something you own, and get matched with items that pair well with it.")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_id = (uploaded_file.name, uploaded_file.size)

    # Only run the (expensive - CLIP, background removal, and a live
    # LLM call per match) pipeline when this is actually a new upload,
    # not on every rerun the category filter widget below triggers.
    if st.session_state.get("uploaded_file_id") != file_id:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        try:
            with st.spinner("Analyzing your photo and finding matches..."):
                result = find_matches(temp_path, top_k=3)
                uploaded_desc = {
                    "category": result["category"],
                    "color": result["color"],
                    "style": result["style"],
                }
                # result["matches"] is already sorted by final_score
                # (rerank() does this), so the first LIVE_EXPLANATION_LIMIT
                # entries are genuinely the top-scoring ones.
                t_explain_start = time.time()
                for i, match in enumerate(result["matches"]):
                    t0 = time.time()
                    if i < LIVE_EXPLANATION_LIMIT:
                        text, source = get_explanation(uploaded_desc, match)
                    else:
                        text = build_template_explanation(uploaded_desc, match)
                        source = "template"
                    print(f"[TIMING] explanation for '{match['name']}' ({source}): {time.time() - t0:.2f}s")
                    match["explanation"] = text
                    match["explanation_source"] = source
                print(
                    f"[TIMING] all {len(result['matches'])} explanations: "
                    f"{time.time() - t_explain_start:.2f}s"
                )
        finally:
            os.remove(temp_path)

        st.session_state["uploaded_file_id"] = file_id
        st.session_state["match_result"] = result
        st.session_state["uploaded_preview"] = uploaded_file.getvalue()

    result = st.session_state["match_result"]

    st.image(st.session_state["uploaded_preview"], caption="Your upload", width=250)
    st.write(f"Detected: **{result['category']}**, **{result['color']}**, **{result['style']}**")

    if not result["matches"]:
        st.info("No catalog items pair with this category yet.")
    else:
        categories_present = sorted({m["category"] for m in result["matches"]})
        selected_categories = st.multiselect(
            "Filter by category", options=categories_present, default=categories_present
        )

        filtered_matches = [m for m in result["matches"] if m["category"] in selected_categories]
        filtered_matches.sort(key=lambda m: m["final_score"], reverse=True)

        st.subheader(f"{len(filtered_matches)} matching item(s)")

        for match in filtered_matches:
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(catalog_image_path(match), use_container_width=True)
            with cols[1]:
                st.markdown(f"**{match['name']}**  ·  ₹{match['price']}")
                st.write(match["explanation"])
                st.link_button("Buy", build_upi_link(match))
            st.divider()
