"""
Step 7 - confirm the explanation layer works together with Step 5's real
matching output, not just on hand-built sample data.
"""

from matching_engine import find_matches
from explanation import get_explanation

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
