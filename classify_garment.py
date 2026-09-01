import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the pretrained CLIP model and its "processor" (which prepares
# images and text so the model can understand them). This downloads
# the model the first time — expect a delay and an internet connection.
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


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


def classify_garment(image_path, candidate_labels):
    image = Image.open(image_path).convert("RGB")

    # The processor turns both the image and our text labels into
    # numbers the model can actually work with
    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)

    # Run the image and labels through CLIP
    outputs = model(**inputs)

    # This score tells us how well each label matches the image
    logits_per_image = outputs.logits_per_image

    # Convert raw scores into probabilities that add up to 100%
    probs = logits_per_image.softmax(dim=1)[0]

    # Pair each label with its probability, sort best-match first
    results = list(zip(candidate_labels, probs.tolist()))
    results.sort(key=lambda x: x[1], reverse=True)

    return results


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