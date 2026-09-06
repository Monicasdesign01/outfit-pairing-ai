"""
Step 6 - confirm Steps 2-5 (background removal, color detection, CLIP
classification, the matching engine) actually work together as one flow,
in plain text, before any visual layer (Step 8) gets built on top.
"""

import os
import sys

# Run from the project root (`python scripts/smoke_test_pipeline.py`):
# Python puts this script's own folder on the path, not the project root,
# so the project's modules and its relative data paths need pointing at.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from matching_engine import find_matches  # noqa: E402

TEST_IMAGES_DIR = "test_images"


def gather_upload_paths():
    """Every photo in test_images/. Two extra ad-hoc photos (test.jpg,
    test2.jpg) used to be appended here, but they are gitignored scratch
    files that don't exist in a fresh clone - so anyone else running this
    got two guaranteed failures that looked like real pipeline bugs."""
    return [
        os.path.join(TEST_IMAGES_DIR, filename)
        for filename in sorted(os.listdir(TEST_IMAGES_DIR))
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def run():
    upload_paths = gather_upload_paths()
    succeeded = 0
    failed = []

    for path in upload_paths:
        print(f"\n--- {path} ---")
        try:
            result = find_matches(path, top_k=3)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append((path, e))
            continue

        print(f"  Classified: {result['category']}, {result['color']}, {result['style']}")
        if not result["matches"]:
            print("  No matches retrieved (allowed category has no catalog items).")
        for m in result["matches"]:
            print(
                f"    -> {m['name']} ({m['category']}, {m['color']}, {m['style']}) "
                f"final_score={m['final_score']:.3f}"
            )
        succeeded += 1

    print(f"\n=== Summary: {succeeded}/{len(upload_paths)} uploads ran through the full pipeline without error ===")
    if failed:
        print("Failed uploads:")
        for path, e in failed:
            print(f"  {path}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    run()
