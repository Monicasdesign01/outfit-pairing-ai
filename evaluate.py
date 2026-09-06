"""
Measures how accurate the automatic detection actually is, against the
catalog's own human-assigned labels as ground truth.

Why only category and colour are scored here, and not style: category was
assigned by hand when each item was catalogued, and colour was set by
looking at every photo directly (that's what "color_verified" means in
catalog.json), so both are independent of what the code predicts. Style
was *computed* by the same CLIP classifier this script would be testing,
so scoring style against it would just measure the classifier against
itself and return a meaningless 100%. It is deliberately left unscored
rather than reported as a number that looks good but proves nothing.

Run:  python evaluate.py
"""

import json
import os
from collections import defaultdict

from classify_garment import classify_garment
from color_detector import get_dominant_color, closest_color_name
from pairing_rules import CATEGORY_LABELS

CATALOG_JSON = "catalog.json"
NOBG_DIR = os.path.join("catalog_images", "nobg")


def nobg_path_for(item):
    """The background-removed version, which is what the live pipeline
    actually classifies - evaluating on the raw photo instead would be
    measuring something the app never does."""
    return os.path.join(NOBG_DIR, os.path.splitext(item["filename"])[0] + "_nobg.png")


def predict_category(image_path):
    label_to_category = {v: k for k, v in CATEGORY_LABELS.items()}
    results = classify_garment(image_path, list(CATEGORY_LABELS.values()))
    top_label, confidence = results[0]
    return label_to_category[top_label], confidence


def evaluate():
    with open(CATALOG_JSON, encoding="utf-8") as f:
        catalog = json.load(f)

    scored = []
    skipped = []

    for item in catalog:
        path = nobg_path_for(item)
        if not os.path.exists(path):
            skipped.append(item["filename"])
            continue

        predicted_category, confidence = predict_category(path)
        predicted_color = closest_color_name(get_dominant_color(path))

        scored.append({
            "filename": item["filename"],
            "true_category": item["category"],
            "predicted_category": predicted_category,
            "confidence": confidence,
            "true_color": item["color"],
            "predicted_color": predicted_color,
        })
        print(".", end="", flush=True)

    print("\n")
    total = len(scored)
    if not total:
        print("Nothing to score - run build_catalog_embeddings.py first to create the nobg images.")
        return

    category_hits = [r for r in scored if r["predicted_category"] == r["true_category"]]
    color_hits = [r for r in scored if r["predicted_color"] == r["true_color"]]

    print("=" * 62)
    print(f"Evaluated {total} catalog items" + (f" ({len(skipped)} skipped, no nobg image)" if skipped else ""))
    print("=" * 62)
    print(f"Category accuracy : {len(category_hits)}/{total}  ({100*len(category_hits)/total:.1f}%)")
    print(f"Colour accuracy   : {len(color_hits)}/{total}  ({100*len(color_hits)/total:.1f}%)")
    print("Style             : not scored - no ground truth independent of the classifier (see module docstring)")

    print("\n--- Category accuracy per true category ---")
    per_category = defaultdict(lambda: [0, 0])
    for r in scored:
        per_category[r["true_category"]][1] += 1
        if r["predicted_category"] == r["true_category"]:
            per_category[r["true_category"]][0] += 1
    for category in sorted(per_category, key=lambda c: per_category[c][0] / per_category[c][1]):
        hits, count = per_category[category]
        print(f"  {category:<8} {hits}/{count}  ({100*hits/count:.0f}%)")

    print("\n--- Category misses ---")
    for r in scored:
        if r["predicted_category"] != r["true_category"]:
            print(f"  {r['filename']:<32} {r['true_category']:>8} -> {r['predicted_category']:<8} ({r['confidence']*100:.0f}% confident)")

    print("\n--- Colour misses ---")
    for r in scored:
        if r["predicted_color"] != r["true_color"]:
            print(f"  {r['filename']:<32} {r['true_color']:>8} -> {r['predicted_color']}")

    # Confidence is worth reporting because the app uses it: a low-confidence
    # prediction is exactly the case where the customer's correction matters.
    wrong = [r for r in scored if r["predicted_category"] != r["true_category"]]
    if wrong and category_hits:
        avg_right = sum(r["confidence"] for r in category_hits) / len(category_hits)
        avg_wrong = sum(r["confidence"] for r in wrong) / len(wrong)
        print(f"\nMean CLIP confidence when correct: {avg_right*100:.1f}%")
        print(f"Mean CLIP confidence when wrong  : {avg_wrong*100:.1f}%")


if __name__ == "__main__":
    evaluate()
