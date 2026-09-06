"""
Tests for the explanation layer. The important property here is that the
LLM is never load-bearing: if the API key is missing or the call fails,
the app must still produce a sensible sentence rather than break a live
demo.
"""

import pytest

from explanation import build_llm_prompt, build_template_explanation, get_explanation

UPLOADED = {"category": "jeans", "color": "navy", "style": "casual"}
MATCH = {"name": "Everyday Cotton Shirt", "category": "shirt", "color": "white", "style": "casual"}


def test_template_explanation_mentions_both_items():
    text = build_template_explanation(UPLOADED, MATCH)
    assert "shirt" in text
    assert "jeans" in text
    assert text.endswith(".")


@pytest.mark.parametrize("category", ["jeans", "pants", "shorts"])
def test_plural_only_garments_read_correctly(category):
    """'This jeans pairs' reads as broken English. An earlier version
    special-cased only jeans, so pants and shorts still read wrong."""
    text = build_template_explanation(UPLOADED, {**MATCH, "category": category})
    assert f"These {category} pair" in text
    assert f"This {category}" not in text


@pytest.mark.parametrize("category", ["skirt", "dress", "blazer", "top"])
def test_singular_garments_read_correctly(category):
    text = build_template_explanation(UPLOADED, {**MATCH, "category": category})
    assert f"This {category} pairs" in text


def test_template_explanation_is_deterministic():
    assert build_template_explanation(UPLOADED, MATCH) == build_template_explanation(UPLOADED, MATCH)


def test_falls_back_to_template_when_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    text, source = get_explanation(UPLOADED, MATCH)
    assert source == "template"
    assert text == build_template_explanation(UPLOADED, MATCH)


def test_falls_back_to_template_when_the_api_call_fails(monkeypatch):
    """A rate limit or network error must degrade to the template rather
    than propagate - this path is what keeps a live demo working."""
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")

    import explanation

    monkeypatch.setattr(
        explanation, "get_llm_explanation", lambda *args, **kwargs: None
    )
    text, source = explanation.get_explanation(UPLOADED, MATCH)
    assert source == "template"
    assert text


def test_prompt_gives_the_model_facts_and_forbids_inventing_more():
    """The LLM only phrases a decision the rules already made - if the
    prompt stopped constraining it, it could invent products."""
    prompt = build_llm_prompt(UPLOADED, MATCH)
    assert "do not invent" in prompt.lower()
    assert "navy" in prompt and "white" in prompt
    assert MATCH["name"] in prompt
