"""
Step 7 - a plain-language explanation for each match, plus a fallback.

The LLM's job here is phrasing only - every fact it's given (category,
color, style, and how they relate) was already decided by Step 5's rules.
It is never asked to invent a reason or a product detail, only to put
already-decided facts into a natural sentence. If the LLM call fails for
any reason (no API key configured, network issue, rate limit, anything),
get_explanation() falls back to a template built from the same facts, so
a live demo never breaks because of a third-party API.
"""

import os

from pairing_rules import explain_color_relationship, explain_silhouette_relationship

GEMINI_MODEL = "gemini-2.5-flash-lite"

COLOR_PHRASES = {
    "complementary": "{upload_color} and {match_color} are complementary colors, giving the outfit a bold contrast.",
    "analogous": "{upload_color} and {match_color} are analogous tones that blend smoothly together.",
    "same": "Matching {upload_color} tones keep the look cohesive.",
    "neutral": "{match_color} is a neutral shade that pairs easily with almost anything, including {upload_color}.",
    "none": "The colors don't clash, though there's no strong color-theory relationship between {upload_color} and {match_color} here.",
}

SILHOUETTE_PHRASES = {
    "balanced": "The {match_style} fit also balances nicely against your {upload_style} piece.",
    "matching": "Both pieces share a {upload_style} feel, which reads as intentional.",
    "none": "",
}


def _subject_phrase(category):
    """'jeans' is the only plural-only noun in CATEGORY_LABELS, so it
    needs 'These jeans pair' rather than 'This jeans pairs'."""
    if category == "jeans":
        return "These jeans", "pair"
    return f"This {category}", "pairs"


def build_template_explanation(uploaded, match):
    """The deterministic fallback - always works, no API involved."""
    color_relationship = explain_color_relationship(uploaded["color"], match["color"])
    silhouette_relationship = explain_silhouette_relationship(uploaded["style"], match["style"])

    subject, verb = _subject_phrase(match["category"])
    sentences = [f"{subject} {verb} well with your {uploaded['category']}."]

    sentences.append(COLOR_PHRASES[color_relationship].format(
        upload_color=uploaded["color"], match_color=match["color"]
    ))

    silhouette_sentence = SILHOUETTE_PHRASES[silhouette_relationship].format(
        upload_style=uploaded["style"], match_style=match["style"]
    )
    if silhouette_sentence:
        sentences.append(silhouette_sentence)

    return " ".join(sentences)


def build_llm_prompt(uploaded, match):
    color_relationship = explain_color_relationship(uploaded["color"], match["color"])
    silhouette_relationship = explain_silhouette_relationship(uploaded["style"], match["style"])
    return (
        "Write one short, friendly sentence (max 30 words) telling a customer why this "
        "catalog item is a good pairing for their uploaded item. Only use the facts given "
        "below - do not invent any other product detail, brand, or reason.\n\n"
        f"Uploaded item: a {uploaded['color']} {uploaded['style']} {uploaded['category']}.\n"
        f"Suggested item: \"{match['name']}\", a {match['color']} {match['style']} {match['category']}.\n"
        f"Color relationship: {color_relationship}.\n"
        f"Silhouette/style relationship: {silhouette_relationship}."
    )


def get_llm_explanation(uploaded, match):
    """
    Returns the LLM's explanation text, or None if it couldn't get one
    for any reason (no API key configured, network issue, rate limit,
    unexpected response). Never raises - the caller always has a safe
    fallback available.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = build_llm_prompt(uploaded, match)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (response.text or "").strip()
        return text if text else None
    except Exception:
        return None


def get_explanation(uploaded, match):
    """
    Step 7's main entry point. Returns (text, source) where source is
    "llm" or "template", so callers/tests can tell which path was used.
    """
    llm_text = get_llm_explanation(uploaded, match)
    if llm_text:
        return llm_text, "llm"
    return build_template_explanation(uploaded, match), "template"


if __name__ == "__main__":
    sample_uploaded = {"category": "jeans", "color": "navy", "style": "casual"}
    sample_match = {"name": "Everyday Cotton Shirt", "category": "shirt", "color": "gray", "style": "casual"}
    text, source = get_explanation(sample_uploaded, sample_match)
    print(f"[{source}] {text}")
