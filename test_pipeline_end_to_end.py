"""
Step 6 - confirm Steps 2-5 (background removal, color detection, CLIP
classification, the matching engine) actually work together as one flow,
in plain text, before any visual layer (Step 8) gets built on top.
"""

import os
from matching_engine import find_matches

TEST_IMAGES_DIR = "test_images"


def gather_upload_paths():
    """Every numbered test photo, plus the two leftover generic photos
    that were never copied into the catalog - genuinely 'new' uploads
    the pipeline hasn't seen as a catalog item."""
    paths = []
    for filename in sorted(os.listdir(TEST_IMAGES_DIR)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(TEST_IMAGES_DIR, filename))
    paths.append("test.jpg")
    paths.append("test2.jpg")
    return paths


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
