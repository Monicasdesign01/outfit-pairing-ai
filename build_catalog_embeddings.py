import os
import json
import numpy as np

from remove_background import remove_background
from classify_garment import get_image_embedding

CATALOG_JSON = "catalog.json"
CATALOG_IMAGES_DIR = "catalog_images"
NOBG_DIR = os.path.join(CATALOG_IMAGES_DIR, "nobg")
OUTPUT_FILE = "catalog_embeddings.npz"


def build_embeddings():
    with open(CATALOG_JSON, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    os.makedirs(NOBG_DIR, exist_ok=True)

    ids = []
    embeddings = []

    for item in catalog:
        image_path = os.path.join(CATALOG_IMAGES_DIR, item["filename"])
        nobg_filename = os.path.splitext(item["filename"])[0] + "_nobg.png"
        nobg_path = os.path.join(NOBG_DIR, nobg_filename)

        # Only remove the background once per catalog photo — this is
        # exactly the kind of thing worth caching, since it never
        # changes for a given catalog item.
        if not os.path.exists(nobg_path):
            print(f"Removing background: {item['filename']}")
            remove_background(image_path, nobg_path)

        print(f"Embedding: {item['filename']}")
        embedding = get_image_embedding(nobg_path)

        ids.append(item["id"])
        embeddings.append(embedding)

    ids_array = np.array(ids)
    embeddings_array = np.array(embeddings, dtype=np.float32)

    np.savez(OUTPUT_FILE, ids=ids_array, embeddings=embeddings_array)
    print(f"\nSaved {len(ids)} catalog embeddings to {OUTPUT_FILE}")
    print(f"Embedding vector size: {embeddings_array.shape[1]}")


if __name__ == "__main__":
    build_embeddings()
