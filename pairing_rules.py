"""
Hand-written rules for Step 5 (the matching engine). No machine learning
here on purpose — there is no free dataset of "outfits that go together",
so these tables encode ordinary styling/colour-theory conventions by hand.
They are a reasonable starting point, not a validated ground truth.
"""

# One CLIP text prompt per catalog category. Whatever CLIP scores highest
# for an uploaded photo maps directly onto one of catalog.json's own
# category values - no separate translation step needed.
#
# "top" and "shirt" wording tuned 2026-09-05 against all 34 catalog items
# as ground truth: the generic "a photo of a top"/"a photo of a shirt"
# was the single biggest source of misclassification (top constantly
# read as shirt), measured at 67.6% overall accuracy. Describing the
# actual visual distinction (collar/buttons vs none) raised accuracy to
# 79.4% - a real, measured change, not a guess. Rewording skirt, dress,
# or shorts too (tried separately) made things *worse* every time -
# same "more emphatic wording backfires with CLIP" lesson already found
# for style labels in Step 3B - so those were deliberately left alone.
CATEGORY_LABELS = {
    "dress": "a photo of a dress",
    "blazer": "a photo of a blazer",
    "jeans": "a photo of jeans",
    "skirt": "a photo of a skirt",
    "pants": "a photo of pants",
    "shorts": "a photo of shorts",
    "top": "a photo of a casual t-shirt or tank top with no collar or buttons",
    "shirt": "a photo of a collared button-up shirt",
    "kurta": "a photo of a kurta",
    "hoodie": "a photo of a hooded zip-up sweatshirt",
}

# Which categories are allowed to be suggested alongside which.
# A bottom pairs with tops/outerwear, never another bottom. A dress is
# already a complete outfit on its own, so it only pairs with outerwear
# layered over it. "pants" and "shorts" are bottoms too, added alongside
# jeans/skirt with identical pairing behavior - there's no meaningful
# styling difference in this simple rule set between "jeans" and "pants"
# as a category of bottom, so they get the same treatment rather than
# inventing a distinction the rules don't actually need.
BOTTOMS = {"jeans", "skirt", "pants", "shorts"}
TOPS = {"top", "shirt", "kurta", "hoodie"}

# Grammatically plural-only garments: "these jeans pair", never "this
# jeans pairs". Kept here with the rest of the category vocabulary so the
# explanation text and the app's own copy can't disagree about it.
PLURAL_CATEGORIES = {"jeans", "pants", "shorts"}


def describe_category(category, article=True):
    """Category name as it should read in a sentence - 'jeans' or 'a
    skirt'. Without an article: 'jeans' / 'skirt'."""
    if not article or category in PLURAL_CATEGORIES:
        return category
    return f"an {category}" if category[0] in "aeiou" else f"a {category}"

PAIRING_RULES = {
    **{bottom: TOPS | {"blazer"} for bottom in BOTTOMS},
    **{top: BOTTOMS | {"blazer"} for top in TOPS},
    "blazer": BOTTOMS | TOPS | {"dress"},
    "dress": {"blazer"},
}


def get_paired_categories(category):
    """Which catalog categories are allowed to pair with the given one."""
    return PAIRING_RULES.get(category, set())


# Same CLIP style prompts used in Step 3B's accuracy check (the milder
# wording that tested better than the more emphatic version), mapped to
# short names the same way CATEGORY_LABELS maps category prompts.
STYLE_LABELS = {
    "fitted": "a photo of fitted clothing",
    "loose": "a photo of loose clothing",
    "casual": "a photo of casual clothing",
    "formal": "a photo of formal clothing",
    "edgy": "a photo of edgy alternative streetwear",
}


# Colour relationships, kept deliberately simple: two colours score high
# if they are classic complementary or analogous pairs, and neutrals
# (black/white/gray/cream/navy) are treated as safely pairing with almost
# everything, matching ordinary styling advice.
NEUTRAL_COLORS = {"black", "white", "gray", "cream", "navy"}

COMPLEMENTARY_PAIRS = {
    frozenset({"blue", "yellow"}),
    frozenset({"red", "green"}),
    frozenset({"pink", "olive"}),
    frozenset({"navy", "yellow"}),
    frozenset({"brown", "blue"}),
}

ANALOGOUS_PAIRS = {
    frozenset({"blue", "navy"}),
    frozenset({"red", "pink"}),
    frozenset({"olive", "green"}),
    frozenset({"olive", "brown"}),
    frozenset({"brown", "cream"}),
}


# How much each signal counts toward the final ranking. Visual similarity
# carries the most weight since it's what FAISS already retrieved on.
# Colour outweighs style/silhouette deliberately: Step 3B's accuracy
# check found CLIP's style classification measurably weaker than type
# classification (~10/15 vs ~12/15) with a known bias toward over-
# predicting "formal" - so the less reliable signal is weighted lower
# rather than trusted equally.
#
# These live here, next to the scoring functions they weight, so anything
# that needs them (the matching engine, the outfit builder) can use the
# same numbers without importing the CLIP-loading modules.
SIMILARITY_WEIGHT = 0.5
COLOR_WEIGHT = 0.35
STYLE_WEIGHT = 0.15


def color_score(color_a, color_b):
    """
    Higher is better. 2 = complementary, 1 = analogous or a neutral
    involved, 0 = no particular relationship either way (not a clash,
    just not a scored one in this simple table).
    """
    if color_a == color_b:
        return 1  # same colour is a safe, tonal pairing, not a clash
    if color_a in NEUTRAL_COLORS or color_b in NEUTRAL_COLORS:
        return 1
    pair = frozenset({color_a, color_b})
    if pair in COMPLEMENTARY_PAIRS:
        return 2
    if pair in ANALOGOUS_PAIRS:
        return 1
    return 0


# Silhouette/style balance: a fitted piece is often paired with something
# looser to avoid the outfit reading as too tight overall, and matching
# formality levels tend to look more intentional than mismatched ones.
def silhouette_score(style_a, style_b):
    """
    Higher is better. Deliberately simple, hand-written rule table.
    Takes the short style names from STYLE_LABELS (e.g. "fitted"),
    not the full CLIP prompt text.
    """
    if style_a == style_b:
        return 1  # matching formality/casualness reads as intentional
    balanced_pairs = {frozenset({"fitted", "loose"})}
    if frozenset({style_a, style_b}) in balanced_pairs:
        return 2  # fitted + loose balances the overall silhouette
    return 0


# Step 7 needs to say *why* two colours or styles pair well, not just
# score them - so these mirror color_score()/silhouette_score()'s exact
# same logic but return a descriptive label instead of a number.
def explain_color_relationship(color_a, color_b):
    if color_a == color_b:
        return "same"
    if color_a in NEUTRAL_COLORS or color_b in NEUTRAL_COLORS:
        return "neutral"
    pair = frozenset({color_a, color_b})
    if pair in COMPLEMENTARY_PAIRS:
        return "complementary"
    if pair in ANALOGOUS_PAIRS:
        return "analogous"
    return "none"


def explain_silhouette_relationship(style_a, style_b):
    if style_a == style_b:
        return "matching"
    if frozenset({style_a, style_b}) == frozenset({"fitted", "loose"}):
        return "balanced"
    return "none"
