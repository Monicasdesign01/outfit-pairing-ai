# Understanding (and explaining) this project

Written for Monica, to be read without any AI assistant open. Everything
here is grounded in what the code actually does and what was actually
measured — no claims you can't back up if someone digs.

The master file (`outfit-pairing-ai-MASTER.md`) is the full history. This
is the version you'd actually reread.

---

## 1. The 60-second version

> "It's a clothing store where you upload a photo of something you already
> own, and it finds items from the catalog that go *with* it — then
> explains why.
>
> The interesting part is that 'looks similar' and 'goes together' are
> opposite goals. If you upload jeans, a normal image-similarity search
> returns more jeans. That's useless. So the pipeline classifies the item
> first, filters the catalog down to categories that actually pair with it
> — tops, not more bottoms — and only *then* searches for visual
> similarity inside that filtered set. Finally it re-ranks using
> colour-theory and silhouette rules.
>
> I measured it rather than guessing: category detection is 67% across the
> catalog, and I know exactly which categories fail and why."

If you only remember one sentence, remember: **items that look similar are
not the same as items that go together.** Everything else follows from it.

---

## 2. The pipeline, in the order it runs

Upload a photo → **classify → filter → retrieve → re-rank** → explain.

1. **Classify** — What is this? CLIP decides the category (jeans, shirt…),
   OpenCV k-means finds the dominant colour, CLIP also guesses style.
2. **Filter** — Which categories pair with it? Hand-written rules say a
   bottom pairs with tops and a blazer, never another bottom. This is the
   step that makes the whole thing work.
3. **Retrieve** — Of the allowed items, which look closest? FAISS searches
   embeddings, one index per category.
4. **Re-rank** — Reorder using colour theory and silhouette balance:
   `similarity×0.5 + colour×0.35 + silhouette×0.15`.
5. **Explain** — Gemini writes one sentence about a match that has
   *already been decided*. It never picks anything, so it can't invent a
   product that isn't in the catalog. If the API fails, a template
   sentence is used instead, so a live demo never breaks.

**Why the weights are uneven:** style detection measured weaker than
category detection, so the least reliable signal is trusted least. That's
a decision you can defend, not a guess.

---

## 3. What each file does

| File | What it does |
|---|---|
| `app.py` | Entry point; sets up the two pages |
| `app_pages/shop.py` | The storefront — grid, filter, sort |
| `app_pages/try_it_on.py` | The AI feature end to end (upload → crop → confirm → match) |
| `matching_engine.py` | The pipeline: classify → filter → retrieve → re-rank |
| `pairing_rules.py` | What pairs with what, colour/silhouette scoring, the weights |
| `outfit_builder.py` | Builds a whole outfit, with the cohesion idea |
| `classify_garment.py` | CLIP: image embeddings + zero-shot classification |
| `color_detector.py` | k-means dominant colour + naming it |
| `remove_background.py` | rembg, so photos match catalog conditions |
| `explanation.py` | Gemini call + the template fallback |
| `evaluate.py` | Measures accuracy against human-assigned labels |
| `tests/` | 72 tests over the logic that matters |

**Read them in this order** if you want to understand the codebase:
`pairing_rules.py` (small, pure, it's the idea) → `color_detector.py` →
`matching_engine.py` → `try_it_on.py`.

---

## 4. The numbers, and what they mean

Run `python evaluate.py` any time to regenerate these.

- **Category: 67.3%** (33/49). But six categories are at **100%** —
  dress, blazer, jeans, shirt, hoodie, pants. The failures concentrate in
  `top` (41%) and `shorts` (33%).
- **Why they fail:** corsets and camisoles catalogued as `top` get read as
  `dress`; flowy shorts get read as `skirt`. That's real visual ambiguity,
  not randomness.
- **Colour: 44.9%** (22/49). Naming a colour from one dominant RGB value
  against a small palette has a genuine ceiling.
- **Confidence is informative:** CLIP averages **78.6%** confidence when
  it's right and **64.7%** when it's wrong. That gap is why the app warns
  you below 70%.
- **Speed: 3.2s** per upload, down from 14.8s.

**Do not quote 79.4% or "~65%".** Those were older numbers measured on
smaller, easier subsets. Quoting a number you can't reproduce is the one
thing that will actually hurt you.

---

## 5. Six stories that show judgment

Interviewers care more about how you think than what you built. Each of
these is true and has evidence behind it.

**1. Hitting a ceiling and changing strategy.** Colour naming wouldn't get
better. Four techniques were tried and measured — hand-tuned swatches, two
uses of the xkcd colour dataset, CIE LAB perceptual distance — and none
beat the others meaningfully. So instead of optimising a component that
isn't the core of the project, the catalog colours were verified by hand
once, and the app lets the customer correct a wrong guess in one click.
*Knowing when to stop optimising is the point of this story.*

**2. Profiling instead of guessing.** Uploads took ~15 seconds. The timing
logs showed the same photo was being encoded by CLIP three separate times
— once for category, once for style, once for the search vector. Scoring
one cached embedding against cached text embeddings cut it to 3.2s. The
old and new versions were compared before deleting the old one:
identical rankings, probabilities matching to 0.000001.

**3. A test caught a real bug.** A test asserting "every reference colour
should name itself" failed: pure grey was being named *beige*, because at
zero saturation the hue reading is meaninglessly 0, which the "is it warm?"
check accepted.

**4. …and the first fix made it worse.** Measurement caught that too:
accuracy dropped 42.9% → 40.8%. Diffing the two runs item by item showed
it had fixed one grey top but turned two white shirts grey. Measuring
those photos gave the actual boundary, and the final version scores
44.9%. *The lesson: "I fixed it" isn't true until you measure it.*

**5. Building something, then deleting it.** A 3D mannequin preview was
built to show outfits worn. Getting it to a quality worth showing needed
cloth-simulation software and 3D garment models — flat photos can't
provide those. So it was deleted rather than kept as something
half-working, and replaced with a 2D pairing card that uses real photos.

**6. CI caught what local testing missed.** Tests passed locally, then
failed on the first CI run. Cause: `python -m pytest` puts the project on
the import path, a bare `pytest` doesn't — and CI used the bare form.
*Local verification was hiding the problem, not proving its absence.*

---

## 6. Questions you'll get, and honest answers

**"Why CLIP instead of training your own classifier?"**
No training data, and no need — CLIP already learned image/word
relationships from the internet, so it classifies from text prompts with
zero examples. Training a classifier for ten categories would have needed
thousands of labelled photos to do worse.

**"Why FAISS for 49 items? Brute force would be identical."**
Correct, and at this size it is. It's there because the *approach* has to
scale, not today's catalog — one index per category keeps working
unchanged at thousands of items. Also, one index per category means the
filtering already happened before any search runs.

**"Why rules for pairing instead of ML?"**
There's no free dataset of "outfits that go together" to learn from. Rules
also stay explainable — I can tell a customer *why* something was
suggested, which the score breakdown in the app actually shows.

**"Isn't the LLM doing the real work?"**
No — it only phrases a decision the rules already made. It's given the
facts and told not to invent any others. If it fails, a template sentence
covers it, so the demo never breaks on a third-party API.

**"67% accuracy isn't very good."**
It's honest, and it's uneven in a useful way: six of ten categories are at
100%, and I know the failing cases are corsets read as dresses. I also
built the app around imperfect detection rather than pretending — it shows
its confidence and lets you correct it in one click.

**"What would you do next?"**
Prompt wording for `top` and `shorts` specifically, since evaluation
pinpointed them. Longer term, a learned colour classifier instead of
nearest-neighbour matching. And reducing memory use, which is the real
constraint on the free hosting tier.

---

## 7. Test yourself (the part that actually works)

Close everything and try to answer out loud. Anywhere you hesitate, open
that file, read it, and try again tomorrow.

1. Why doesn't this just use image similarity? What breaks if it did?
2. What happens, step by step, between clicking upload and seeing matches?
3. Where do the pairing rules live, and what would you change to make
   blazers pair with dresses?
4. Why is style weighted lowest in the ranking?
5. What does `evaluate.py` deliberately *not* measure, and why?
6. Why does the app show a confidence warning below 70% specifically?
7. What's in `cohesion_score()` and what would the outfit look like
   without it?

A stronger version: **change something small and see if the tests catch
you.** Edit a pairing rule so bottoms pair with bottoms, run `pytest`, and
watch it fail. That's the fastest way to learn what the tests protect.

---

## 8. Working on this without an assistant

**Your three commands:**

```bash
pytest                 # did I break anything?
python evaluate.py     # how well does it work right now?
streamlit run app.py   # does it actually work in the browser?
```

**When something breaks:** read the last line of the error first — it
names the file and line. Most errors are a typo, a missing import, or a
wrong filename. Compare against a working file nearby.

**When you change catalog.json:** re-run `python build_catalog_embeddings.py`,
or the app will look for embeddings that don't exist.

**Before pushing:** run `pytest`. If it's green, push — CI will double
check on Linux, which is where the deployed app actually runs.

**Don't be afraid of breaking it.** Everything is in git. `git status`
shows what you changed; `git diff` shows exactly what's different; and if
you need to abandon a change, `git checkout <file>` restores it.

---

## 9. Two things still open

- **`catalog_images/beige_blazer.jpg` has a visible iStock watermark.**
  Replace it with a freely-licensed photo (Unsplash/Pexels) and re-run
  `build_catalog_embeddings.py`, or remove that item. Worth checking the
  other photos while you're there.
- **Memory on the free hosting tier** is the real limit (~725MB measured
  against 1GB). If the app ever gets killed for resources, the fix is a
  smaller CLIP variant or making background removal optional — not more
  debugging.
