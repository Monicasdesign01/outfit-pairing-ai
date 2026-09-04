"""
Step 5 - the matching engine: classify -> filter -> retrieve -> re-rank.
See outfit-pairing-ai-MASTER.md Section 11 for the full explanation.
"""

import os
import json
import time
import numpy as np
import faiss

from remove_background import remove_background
from classify_garment import classify_garment, get_image_embedding
from color_detector import get_dominant_color, closest_color_name
from pairing_rules import (
    CATEGORY_LABELS,
    STYLE_LABELS,
    get_paired_categories,
    color_score,
    silhouette_score,
)

CATALOG_JSON = "catalog.json"
CATALOG_EMBEDDINGS_FILE = "catalog_embeddings.npz"


def classify_uploaded_item(image_path):
    """
    Step 5.1 - Classify. Reuses Step 3's CLIP zero-shot classifier
    (classify_garment), scored against one prompt per catalog category
    (pairing_rules.CATEGORY_LABELS) so the result maps straight onto a
    real catalog category.
    """
    labels = list(CATEGORY_LABELS.values())
    results = classify_garment(image_path, labels)
    top_label, confidence = results[0]

    # results gives us the winning CLIP prompt text; map it back to the
    # short category name (e.g. "a photo of jeans" -> "jeans")
    label_to_category = {v: k for k, v in CATEGORY_LABELS.items()}
    category = label_to_category[top_label]

    return category, confidence


def load_catalog():
    """Loads catalog.json and lines its cached embeddings up by id."""
    with open(CATALOG_JSON, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    data = np.load(CATALOG_EMBEDDINGS_FILE)
    ids = list(data["ids"])
    embeddings = data["embeddings"]
    id_to_embedding = {ids[i]: embeddings[i] for i in range(len(ids))}

    for item in catalog:
        item["embedding"] = id_to_embedding[item["id"]]

    return catalog


def build_category_indices(catalog):
    """
    Step 5.3 - Retrieve, part 1: one FAISS index per category, not one
    shared index with results filtered afterwards. FAISS has no clean
    built-in way to say "search everything, but only actually consider
    these N items" - a shared index would need extra bookkeeping to
    throw away disallowed results after the search runs. Building a
    separate index per category means the filtering already happened
    before any search runs at all.

    At this catalog's current size (15 items across 8 categories), a
    plain brute-force distance comparison would be exactly as fast as
    FAISS - there's no real speed win yet. FAISS is used anyway because
    this is the code that needs to scale, not today's tiny catalog: the
    same per-category-index approach keeps working unchanged if the
    catalog grows to hundreds or thousands of real items later.
    """
    indices_by_category = {}

    for category in CATEGORY_LABELS:
        items_in_category = [item for item in catalog if item["category"] == category]
        if not items_in_category:
            continue

        vectors = np.array([item["embedding"] for item in items_in_category], dtype=np.float32)
        faiss.normalize_L2(vectors)  # so inner product = cosine similarity

        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

        indices_by_category[category] = (index, items_in_category)

    return indices_by_category


def retrieve_candidates(uploaded_embedding, allowed_categories, indices_by_category, top_k=3):
    """
    Step 5.3 - Retrieve, part 2: search only the FAISS indices for
    categories the filter step allowed, and merge their results into
    one candidate list.
    """
    query_vector = np.array([uploaded_embedding], dtype=np.float32)
    faiss.normalize_L2(query_vector)

    candidates = []

    for category in allowed_categories:
        if category not in indices_by_category:
            continue

        index, items_in_category = indices_by_category[category]
        k = min(top_k, len(items_in_category))
        similarities, positions = index.search(query_vector, k)

        for similarity, position in zip(similarities[0], positions[0]):
            item = items_in_category[position]
            candidates.append({**item, "similarity": float(similarity)})

    return candidates


def classify_uploaded_style(image_path):
    """Same idea as classify_uploaded_item, but for style rather than category."""
    labels = list(STYLE_LABELS.values())
    results = classify_garment(image_path, labels)
    top_label, confidence = results[0]
    label_to_style_name = {v: k for k, v in STYLE_LABELS.items()}
    return label_to_style_name[top_label]


# How much each signal counts toward the final ranking. Visual similarity
# carries the most weight since it's what FAISS already retrieved on.
# Colour outweighs style/silhouette deliberately: Step 3B's accuracy
# check found CLIP's style classification measurably weaker than type
# classification (~10/15 vs ~12/15) with a known bias toward over-
# predicting "formal" - so the less reliable signal is weighted lower
# rather than trusted equally.
SIMILARITY_WEIGHT = 0.5
COLOR_WEIGHT = 0.35
STYLE_WEIGHT = 0.15


def rerank(candidates, uploaded_color, uploaded_style):
    """
    Step 5.4 - Re-rank. Reorders the FAISS candidates using hand-written
    colour-theory and silhouette rules, combined with the similarity
    score FAISS already gave each one.
    """
    for candidate in candidates:
        c_score = color_score(uploaded_color, candidate["color"])
        s_score = silhouette_score(uploaded_style, candidate["style"])

        # color_score/silhouette_score max out at 2, so /2 puts every
        # signal on the same 0-1 scale before weighting.
        candidate["final_score"] = (
            SIMILARITY_WEIGHT * candidate["similarity"]
            + COLOR_WEIGHT * (c_score / 2)
            + STYLE_WEIGHT * (s_score / 2)
        )

    return sorted(candidates, key=lambda c: c["final_score"], reverse=True)


def find_matches(image_path, top_k=3):
    """The full Step 5 pipeline: classify -> filter -> retrieve -> re-rank."""
    # Timing logs added while diagnosing a slow first upload through the
    # Streamlit app - printed to stdout (visible in the streamlit run
    # console) so it's obvious which stage is actually slow, rather than
    # guessing. See outfit-pairing-ai-MASTER.md Section 11, Step 8.
    t_start = time.time()

    # Background-removed first, matching exactly how every catalog photo
    # was processed before its embedding/color/style were computed -
    # otherwise this would be comparing "garment plus background" against
    # clean catalog embeddings, an apples-to-oranges comparison.
    nobg_path = os.path.splitext(image_path)[0] + "_matchtmp_nobg.png"
    remove_background(image_path, nobg_path)
    t_bg = time.time()
    print(f"[TIMING] background removal: {t_bg - t_start:.2f}s")

    category, _ = classify_uploaded_item(nobg_path)
    allowed_categories = get_paired_categories(category)
    t_classify = time.time()
    print(f"[TIMING] classify category (CLIP): {t_classify - t_bg:.2f}s")

    catalog = load_catalog()
    indices_by_category = build_category_indices(catalog)
    t_index = time.time()
    print(f"[TIMING] load catalog + build FAISS indices: {t_index - t_classify:.2f}s")

    uploaded_embedding = get_image_embedding(nobg_path)
    t_embed = time.time()
    print(f"[TIMING] compute uploaded embedding (CLIP): {t_embed - t_index:.2f}s")

    candidates = retrieve_candidates(uploaded_embedding, allowed_categories, indices_by_category, top_k)
    t_retrieve = time.time()
    print(f"[TIMING] FAISS retrieval: {t_retrieve - t_embed:.2f}s")

    uploaded_rgb = get_dominant_color(nobg_path)
    uploaded_color = closest_color_name(uploaded_rgb)
    t_color = time.time()
    print(f"[TIMING] color detection: {t_color - t_retrieve:.2f}s")

    uploaded_style = classify_uploaded_style(nobg_path)
    t_style = time.time()
    print(f"[TIMING] classify style (CLIP): {t_style - t_color:.2f}s")

    os.remove(nobg_path)

    ranked = rerank(candidates, uploaded_color, uploaded_style)
    t_rerank = time.time()
    print(f"[TIMING] re-rank: {t_rerank - t_style:.2f}s")

    for item in ranked:
        del item["embedding"]  # not needed past this point, keeps output readable

    print(f"[TIMING] find_matches TOTAL (excludes explanation calls): {t_rerank - t_start:.2f}s")

    return {
        "category": category,
        "color": uploaded_color,
        "style": uploaded_style,
        "matches": ranked,
    }


if __name__ == "__main__":
    result = find_matches("test_images/test_03_jeans.jpg")

    print(f"Uploaded item: {result['category']}, {result['color']}, {result['style']}\n")
    print("Ranked matches:")
    for m in result["matches"]:
        print(
            f"  {m['name']} ({m['category']}, {m['color']}, {m['style']}) "
            f"- similarity {m['similarity']:.3f}, final score {m['final_score']:.3f}"
        )
