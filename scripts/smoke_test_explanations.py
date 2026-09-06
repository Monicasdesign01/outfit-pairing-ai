"""
Step 7 - confirm the explanation layer works together with Step 5's real
matching output, not just on hand-built sample data.
"""

import os
import sys

# Run from the project root (`python scripts/smoke_test_explanations.py`).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from matching_engine import find_matches  # noqa: E402
from explanation import get_explanation  # noqa: E402

UPLOAD_PATHS = ["test_images/test_03_jeans.jpg", "test_images/test_8_kurta.jpg"]


def run():
    for path in UPLOAD_PATHS:
        result = find_matches(path, top_k=2)
        uploaded = {"category": result["category"], "color": result["color"], "style": result["style"]}

        print(f"\n--- {path} ---")
        print(f"Uploaded: {uploaded['category']}, {uploaded['color']}, {uploaded['style']}")

        for match in result["matches"]:
            text, source = get_explanation(uploaded, match)
            print(f"  {match['name']} ({match['category']}, {match['color']}, {match['style']})")
            print(f"    [{source}] {text}")


if __name__ == "__main__":
    run()
