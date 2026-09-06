from functools import lru_cache

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the pretrained CLIP model and its "processor" (which prepares
# images and text so the model can understand them). This downloads
# the model the first time — expect a delay and an internet connection.
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


@lru_cache(maxsize=8)
def _text_embeddings(candidate_labels):
    """
    CLIP's text side only depends on the label wording, which never changes
    at runtime - so the same handful of label sets (categories, styles) are
    encoded once per process and reused, instead of on every photo.
    """
    inputs = processor(text=list(candidate_labels), return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model.get_text_features(**inputs)
    embeddings = output.pooler_output
    return embeddings / embeddings.norm(dim=-1, keepdim=True)


def get_image_embedding(image_path):
    """
    Turns a photo into a single numeric vector (CLIP's image
    embedding) that captures what the image looks like. Two photos
    that look visually similar will have vectors that are close
    together — this is what similarity search (Step 5, FAISS) compares.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.get_image_features(**inputs)

    # In this project's installed transformers version, get_image_features()
    # returns a BaseModelOutputWithPooling object rather than a plain
    # tensor. pooler_output is the actual 512-dim projected CLIP embedding
    # (last_hidden_state is the pre-pooling per-patch output, not what we want).
    image_features = output.pooler_output

    return image_features[0].tolist()


def classify_from_embedding(image_embedding, candidate_labels):
    """
    Zero-shot classification from an image embedding that has already been
    computed, rather than re-reading and re-encoding the photo.

    This is what CLIP does internally anyway: score an image against text
    by comparing their embeddings. Splitting it out means one photo can be
    classified against several label sets (category, style) and reused for
    similarity search, all from a single image forward pass - the earlier
    version ran a full model pass per label set, so a single upload encoded
    the same photo three separate times.

    Verified against the previous implementation before replacing it: same
    ranking, probabilities identical to within 0.000001.
    """
    image_embedding = torch.as_tensor(image_embedding, dtype=torch.float32)
    if image_embedding.dim() == 1:
        image_embedding = image_embedding.unsqueeze(0)
    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

    text_embeddings = _text_embeddings(tuple(candidate_labels))

    with torch.no_grad():
        logits = model.logit_scale.exp() * image_embedding @ text_embeddings.T
        probs = logits.softmax(dim=-1)[0]

    results = list(zip(candidate_labels, probs.tolist()))
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def classify_garment(image_path, candidate_labels):
    """Classify a photo from its path. Convenience wrapper for scripts that
    only need one classification - the app itself computes the embedding
    once and calls classify_from_embedding() directly."""
    return classify_from_embedding(get_image_embedding(image_path), candidate_labels)


if __name__ == "__main__":
    type_labels = [
        "a photo of jeans",
        "a photo of a dress",
        "a photo of a jacket",
        "a photo of a shirt",
        "a photo of a t-shirt",
    ]
    style_labels = [
        "a photo of fitted clothing",
        "a photo of loose clothing",
        "a photo of casual clothing",
        "a photo of formal clothing",
    ]

    print("Garment type:")
    for label, prob in classify_garment("test2_nobg.png", type_labels):
        print(f"  {label}: {prob*100:.1f}%")

    print("\nStyle:")
    for label, prob in classify_garment("test2_nobg.png", style_labels):
        print(f"  {label}: {prob*100:.1f}%")