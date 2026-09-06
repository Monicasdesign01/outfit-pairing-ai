"""
Try It On Your Clothes - the AI feature. Upload a photo, crop it down to
just the one garment to match if the photo shows more than one (e.g. a
full outfit), review the detected category/color/style (and correct
anything wrong - automatic detection is measurably imperfect, see
Section 11), then get ranked matches with a category filter (Section 4
of outfit-pairing-ai-MASTER.md).
"""

import os
import tempfile
import time

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from matching_engine import analyze_uploaded_photo, find_matches_from_details
from explanation import get_explanation, build_template_explanation
from shop_utils import catalog_image_path, build_upi_link
from pairing_rules import CATEGORY_LABELS, STYLE_LABELS, describe_category
from color_detector import ALL_COLOR_NAMES
from outfit_builder import build_outfit

CATEGORY_OPTIONS = sorted(CATEGORY_LABELS.keys())
# ALL_COLOR_NAMES is color_detector.py's own single source of truth for
# every value closest_color_name() can return - reconstructing this list
# by hand from its internal reference dicts (as an earlier version did)
# missed "beige" (a real return value that isn't a dict key) and crashed
# the app with a ValueError the first time a real upload detected it.
COLOR_OPTIONS = ALL_COLOR_NAMES
STYLE_OPTIONS = sorted(STYLE_LABELS.keys())

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

# Below this, CLIP's category guess is worth flagging to the customer.
# Chosen from measured data rather than picked by feel: across the
# catalog, CLIP averages ~79% confidence when it gets the category right
# and ~65% when it gets it wrong (run evaluate.py to reproduce), so the
# threshold sits between those two means.
LOW_CONFIDENCE_THRESHOLD = 0.70

st.title("Try It On Your Clothes")
st.caption("Upload a photo of something you own, and get matched with items that pair well with it.")

st.subheader("1. Upload a photo")
uploaded_file = st.file_uploader(
    "Upload a photo",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="A photo of one garment works best. If it shows a whole outfit, you can crop it in the next step.",
)

if uploaded_file is None:
    st.info("Upload a photo of a top or a bottom to see what it pairs with.")

if uploaded_file is not None:
    file_id = (uploaded_file.name, uploaded_file.size)

    # A genuinely new upload starts over from the crop step - old
    # detection/results no longer apply to a different photo.
    if st.session_state.get("uploaded_file_id") != file_id:
        st.session_state["uploaded_file_id"] = file_id
        st.session_state["crop_confirmed"] = False
        st.session_state.pop("detected", None)
        st.session_state.pop("match_result", None)

    if not st.session_state.get("crop_confirmed"):
        st.subheader("2. Crop to one garment")
        st.caption(
            "If your photo shows more than one garment (e.g. a full outfit), drag the box "
            "around just the one you want matched — a top or a bottom, not both."
        )
        original_image = Image.open(uploaded_file).convert("RGB")

        crop_col, preview_col = st.columns([2, 1])
        with crop_col:
            # aspect_ratio=None allows a free-form box, not a fixed shape -
            # a "top" crop and a "bottom" crop are very different shapes.
            cropped_image = st_cropper(
                original_image, realtime_update=True, box_color="#8A6552", aspect_ratio=None
            )
        with preview_col:
            with st.container(border=True):
                st.caption("This is what gets analysed")
                st.image(cropped_image, width="stretch")

        col_a, col_b = st.columns(2)
        with col_a:
            use_crop_clicked = st.button("Use this crop", type="primary", width="stretch")
        with col_b:
            use_full_clicked = st.button("Use the full photo instead", width="stretch")

        if use_crop_clicked or use_full_clicked:
            final_image = cropped_image if use_crop_clicked else original_image

            suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                final_image.save(tmp.name)
                temp_path = tmp.name

            try:
                with st.spinner("Analyzing your photo..."):
                    details = analyze_uploaded_photo(temp_path)
            finally:
                os.remove(temp_path)

            st.session_state["detected"] = details
            st.session_state["uploaded_preview"] = final_image
            st.session_state["crop_confirmed"] = True
            st.rerun()

    else:
        st.subheader("2. Check what we detected")

        upload_col, detail_col = st.columns([1, 2])
        with upload_col:
            with st.container(border=True):
                st.image(st.session_state["uploaded_preview"], width="stretch")
                st.caption("Your upload")
                if st.button("Crop a different area", width="stretch"):
                    st.session_state["crop_confirmed"] = False
                    st.session_state.pop("detected", None)
                    st.session_state.pop("match_result", None)
                    st.rerun()

        detected = st.session_state["detected"]

        # CLIP reports how confident it is, and that number is genuinely
        # informative: measured across the catalog, it averages ~79% when
        # the category is right and ~65% when it's wrong (see evaluate.py).
        # So a low score is a real signal that this is a photo worth
        # double-checking, not decoration - say so instead of presenting
        # every guess with the same authority.
        with detail_col:
            with st.container(border=True):
                confidence = detected.get("category_confidence", 1)
                described = describe_category(detected["category"])
                if confidence < LOW_CONFIDENCE_THRESHOLD:
                    st.warning(
                        f"Not very sure about this one — **{confidence*100:.0f}%** confident it's "
                        f"{described}. Worth checking before continuing."
                    )
                else:
                    st.success(f"Detected {described} — {confidence*100:.0f}% confident.")

                st.caption("Correct anything that looks wrong, then find matches.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    category = st.selectbox(
                        "Category", CATEGORY_OPTIONS, index=CATEGORY_OPTIONS.index(detected["category"])
                    )
                with col2:
                    color = st.selectbox("Colour", COLOR_OPTIONS, index=COLOR_OPTIONS.index(detected["color"]))
                with col3:
                    style = st.selectbox("Style", STYLE_OPTIONS, index=STYLE_OPTIONS.index(detected["style"]))

                find_clicked = st.button("Find my matches", type="primary", width="stretch")

        if find_clicked:
            with st.spinner("Finding matches..."):
                result = find_matches_from_details(detected["embedding"], category, color, style, top_k=3)
                uploaded_desc = {"category": category, "color": color, "style": style}

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
                print(f"[TIMING] all {len(result['matches'])} explanations: {time.time() - t_explain_start:.2f}s")

            st.session_state["match_result"] = result

        if "match_result" in st.session_state:
            result = st.session_state["match_result"]

            if not result["matches"]:
                st.info("No catalog items pair with this category yet.")
            else:
                st.divider()
                st.subheader("3. What it pairs with")

                categories_present = sorted({m["category"] for m in result["matches"]})
                selected_categories = st.multiselect(
                    "Filter by category", options=categories_present, default=categories_present
                )

                filtered_matches = [m for m in result["matches"] if m["category"] in selected_categories]
                filtered_matches.sort(key=lambda m: m["final_score"], reverse=True)

                if filtered_matches:
                    st.markdown("##### Outfit preview")
                    preview_options = {f"{m['name']} (₹{m['price']})": m for m in filtered_matches}
                    preview_label = st.selectbox("See it paired with:", list(preview_options.keys()))
                    preview_match = preview_options[preview_label]

                    with st.container(border=True):
                        img_col, plus_col, match_col = st.columns([1, 0.2, 1])
                        with img_col:
                            st.image(st.session_state["uploaded_preview"], width="stretch")
                            st.caption(f"Your {result['category']} — {result['color']}")
                        with plus_col:
                            st.markdown(
                                "<div style='text-align:center; font-size:2rem; padding-top:2.5rem;'>+</div>",
                                unsafe_allow_html=True,
                            )
                        with match_col:
                            st.image(catalog_image_path(preview_match), width="stretch")
                            st.caption(f"{preview_match['name']} — ₹{preview_match['price']}")

                # A single complementary item usually isn't an outfit -
                # jeans plus a shirt still needs a layer. This assembles a
                # whole look from the same ranked candidates.
                outfit = build_outfit(result["category"], filtered_matches)
                if outfit:
                    st.markdown("##### Complete the look")
                    st.caption(
                        "Each piece is scored against the ones already chosen, not just against "
                        "your upload, so the outfit holds together as a whole."
                    )
                    with st.container(border=True):
                        pieces = st.columns(len(outfit) + 1)
                        with pieces[0]:
                            st.image(st.session_state["uploaded_preview"], width="stretch")
                            st.caption(f"**Yours** — {describe_category(result['category'])}, {result['color']}")
                        for column, piece in zip(pieces[1:], outfit):
                            with column:
                                st.image(catalog_image_path(piece), width="stretch")
                                st.caption(
                                    f"**{piece['slot'].title()}** — {piece['name']}  \n"
                                    f"₹{piece['price']:,} · fits the look {piece['cohesion']:.2f}"
                                )
                        st.markdown(
                            f"**Outfit total: ₹{sum(p['price'] for p in outfit):,}** "
                            f"across {len(outfit)} item(s) plus your own piece."
                        )

                st.markdown(f"##### {len(filtered_matches)} matching item(s)")

                for rank, match in enumerate(filtered_matches, start=1):
                    with st.container(border=True):
                        image_col, detail_col = st.columns([1, 2])
                        with image_col:
                            st.image(catalog_image_path(match), width="stretch")
                        with detail_col:
                            st.markdown(f"**{rank}. {match['name']}**")
                            st.markdown(f"### ₹{match['price']:,}")
                            st.caption(
                                f"{match['category'].title()} · {match['color'].title()} · {match['style'].title()}"
                            )
                            st.write(match["explanation"])

                            # The ranking is a weighted sum, not a black box -
                            # showing the parts lets anyone check the order is
                            # actually justified by the scores behind it.
                            with st.expander(f"Why this ranked here — score {match['final_score']:.3f}"):
                                for label, value in match["score_breakdown"].items():
                                    share = value / match["final_score"] if match["final_score"] else 0
                                    st.write(f"**{label}** — {value:.3f}")
                                    st.progress(min(share, 1.0))
                                st.caption(
                                    f"Colour relationship: {match['color_relationship']} · "
                                    f"Silhouette: {match['style_relationship']} · "
                                    f"Weights: similarity 50%, colour 35%, silhouette 15%"
                                )

                            st.link_button("Buy", build_upi_link(match), width="stretch")
