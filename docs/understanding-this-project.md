# Understanding and explaining this project

Everything you need to talk about this project confidently, in plain
language. Every number here can be reproduced by running
`python evaluate.py` — never quote a number you can't reproduce.

---

## 1. The 30-second answer to "tell me about your project"

Interviewers want this shape: **problem → approach → result → hardest part.**

> "It's a clothing store where you upload a photo of something you already
> own, and it recommends catalog items that go *with* it, and explains why.
>
> The key idea is that 'looks similar' and 'goes together' are opposite
> goals. If you upload jeans, a normal image-similarity search gives you
> more jeans — useless. So I classify the item first, filter the catalog
> to only categories that pair with it, search for visual similarity
> *inside* that filtered set, then re-rank using colour and silhouette
> rules.
>
> I measured it instead of guessing — category detection is 67% across 49
> items, and I know exactly which categories fail and why. The hardest
> part was colour detection, where I hit a real ceiling and solved it by
> changing the design instead of the algorithm."

That last sentence usually gets the follow-up question — which is good,
because it's your strongest story.

---

## 2. What each step does

| Step | What happens | Why it exists |
|---|---|---|
| **Upload + crop** | You pick a photo, crop to one garment | Photos often show a whole outfit; the model can only classify one thing |
| **Background removal** | `rembg` strips the background | Catalog photos were processed the same way — comparing a garment-with-background against clean images would be unfair |
| **Classify** | CLIP names the category and style; k-means finds the dominant colour | You can't filter until you know what the item is |
| **Confirm** | You can correct category/colour/style | Detection is imperfect, so the design accounts for it instead of hiding it |
| **Filter** | Rules pick which categories can pair with it | **This is the step that makes the project work** — a bottom pairs with tops, never another bottom |
| **Retrieve** | FAISS finds the visually closest items *within* allowed categories | Narrowing first means the search can't return nonsense |
| **Re-rank** | Score = similarity×0.5 + colour×0.35 + silhouette×0.15 | "Looks closest" isn't "pairs best" |
| **Explain** | Gemini writes one sentence about a match already chosen | Phrasing is what language models are good at; deciding is not |
| **Complete the look** | Builds a full outfit, scoring each new piece against the ones already picked | One item isn't an outfit — jeans + shirt still needs a layer |

**Why the weights are uneven:** style detection measured weaker than
category detection, so the least reliable signal gets the least influence.
That's a decision you can defend.

---

## 3. The challenges I faced (and what I did)

These are the most valuable part of your interview. Each is true.

**1. Colour detection hit a real ceiling.**
Naming a colour from one dominant RGB value against a small palette kept
getting ~45%. I tried four different techniques — hand-tuned reference
colours, two versions using a public colour-survey dataset, and a
perceptual colour space (CIE LAB). None was meaningfully better.
*What I did:* stopped optimising a component that wasn't the core of the
project. I verified all 49 catalog colours by eye once, and made the app
show its guess in an editable dropdown so a wrong detection is a one-click
fix. **Lesson: knowing when to stop tuning is an engineering skill.**

**2. Uploads took 15 seconds.**
I had timing logs in the pipeline. They showed the same photo was being
encoded by CLIP **three separate times** — once for category, once for
style, once for the search vector.
*What I did:* encoded the image once and compared that one embedding
against cached text embeddings. **15s → 3.2s.** I checked old vs new gave
identical results before deleting the old code.

**3. The app worked locally and crashed in production.**
Twice. First, `opencv-python` needs system graphics libraries that servers
don't have — the fix is `opencv-python-headless`. Second, PyTorch's
CPU-only build isn't on normal PyPI, so it needed an extra package index.
*What I did:* fixed both, then set up CI that installs the real
requirements on Linux, so this class of bug gets caught before deploying.

**4. A crash from a value that wasn't in a list.**
The colour detector could return `"beige"`, but the dropdown built its
options from two dictionaries — and `"beige"` wasn't a key in either. The
first time a photo was detected as beige, the page crashed.
*What I did:* didn't just add the missing string. I made one list in the
detector itself that defines every value it can return, so the two can't
drift apart again. Then tested 20,000 random colours to confirm nothing
else could escape.

**5. A test found a bug I didn't know about.**
I wrote a test saying "every reference colour should name itself." It
failed — pure grey was being named *beige*, because when a colour has no
saturation the hue reading is meaninglessly 0, which my "is it warm?"
check accepted.

**6. …and my first fix made things worse.**
Accuracy dropped from 42.9% to 40.8%. I compared the two runs item by
item: it fixed one grey top but turned two white shirts grey. I measured
the actual brightness of those three photos, found the boundary had to sit
between them, and set it from data. Final: **44.9%**.
**Lesson: "I fixed it" isn't true until you measure it.**

**7. Tests passed locally and failed in CI.**
`python -m pytest` puts your project on the import path; a bare `pytest`
doesn't — and CI used the bare form. My local testing had been *hiding*
the problem, not proving it worked. Fixed with a `pytest.ini`.

**8. The catalog photos aren't clean product shots.**
They're real lifestyle photos — a model wearing the item, often with a bag
or other clothing in frame. That limits what image processing can do,
because background removal strips the background, not the extra objects.
*What I did:* designed around it rather than pretending, and let the
customer crop their own photo to just the garment.

**9. Free hosting has a 1GB memory limit.**
I measured 725MB. It fits, but there's not much headroom. I know the fix
if it ever fails (a smaller CLIP model, or making background removal
optional) — that's better than being surprised.

---

## 4. The tools, and why each one

| Tool | What it is | Why I chose it |
|---|---|---|
| **CLIP** | A model trained on image–text pairs, so images and words live in the same "space" | Lets me classify from text prompts with **zero training examples** — I had no labelled dataset |
| **FAISS** | A library for fast similarity search over vectors | Free, runs locally, and one index per category means filtering happens *before* searching |
| **OpenCV (k-means)** | Classic image processing; k-means groups pixels | Finding the most common colour is counting and grouping — a solved maths problem, not a job for AI |
| **rembg** | Background removal | Makes uploads match how catalog photos were processed |
| **Gemini** | A large language model | Good at phrasing; it never chooses matches, so it can't invent products |
| **Streamlit** | Builds web UIs in pure Python | Kept the whole project in one language so effort went into the pipeline, not frontend |
| **pytest** | Testing framework | 72 tests protect the rules that encode the project's actual idea |
| **GitHub Actions** | Runs tests automatically on every push | Catches Linux-only failures before the deployed app does |
| **Git/GitHub** | Version control | Every change is recoverable; the history shows how the project evolved |

---

## 5. Skills I gained (say these plainly)

- **Using pretrained models instead of training my own** — knowing when
  zero-shot is enough is a real judgment call
- **Vector embeddings and similarity search** — how images become numbers
  and how "closeness" is measured (cosine similarity)
- **Measuring instead of assuming** — I built an evaluation script and it
  corrected two of my own earlier claims
- **Testing and CI** — including the experience of tests catching a bug I
  wrote myself
- **Debugging production vs local differences** — the two deployment
  crashes taught me more than the features did
- **Making trade-offs and defending them** — rules vs ML, FAISS vs brute
  force, stopping work on colour detection
- **Python** (coming from Java), **Git**, **API integration with a
  fallback path** so a demo never breaks

---

## 6. How to show the project is good

Point at evidence, not adjectives.

| Say this | Show this |
|---|---|
| "It's measured, not guessed" | `python evaluate.py` prints accuracy and every individual miss |
| "It's tested" | `pytest` — 72 tests, green CI badge on the README |
| "It's live" | The Streamlit URL, working in a browser |
| "It's honest" | The README lists real numbers, including the weak ones |
| "It's explainable" | The "why this ranked here" panel shows the score breakdown |
| "I profiled it" | 15s → 3.2s, with the reason (three CLIP passes → one) |

**The strongest single move in an interview:** say a number, then offer to
show how you'd reproduce it. Most candidates can't.

---

## 7. Interview questions and answers

### About your project

**"Walk me through your project."**
Use the 30-second answer in section 1, then stop and let them ask.

**"Why not just use image similarity?"**
Because similarity finds *more of the same thing*. Upload jeans and you
get more jeans. The recommendation I want is the opposite — something
different that goes with it. That's why category filtering comes before
the similarity search, not after.

**"How did you evaluate it?"**
I wrote `evaluate.py`, which scores detection against the catalog's
human-assigned labels for all 49 items. Category is 67.3%, colour 44.9%.
More useful than the headline: six of ten categories are at 100%, and the
failures concentrate in `top` (41%) and `shorts` (33%).

**"Why are those two so bad?"**
Real visual ambiguity, not randomness. Corsets and camisoles catalogued as
`top` get read as `dress`; flowy shorts get read as `skirt`. Knowing the
*cause* is what makes it fixable — the next step is prompt wording that
describes the visual difference.

**"67% doesn't sound very good."**
It's honest, and the app is designed around imperfect detection rather
than pretending otherwise: it shows its confidence and lets the user
correct it in one click. I'd rather have a measured 67% I can improve than
an unmeasured claim.

**"How do you know the recommendations are actually good?"**
I don't, and I'm careful not to claim it. I can measure the *components* —
does it correctly identify the garment and its colour — because I have
ground truth for those. Measuring "is this a good outfit" would need real
users, click data, or human raters. That would be A/B testing or
precision@k with labelled pairs, and I didn't have that data.
*(This answer is strong precisely because it's honest about the limit.)*

### About CLIP and embeddings

**"How does CLIP work?"**
It's trained on huge numbers of image–caption pairs. It learns to put an
image and its matching text close together in the same mathematical space,
and non-matching pairs far apart (contrastive learning). Because they
share a space, you can compare a picture to a sentence directly using
cosine similarity.

**"So how do you classify with no training data?"**
Zero-shot. I write one sentence per category — "a photo of jeans", "a
photo of a collared button-up shirt" — encode those sentences, encode the
image, and pick the sentence closest to the image. No labelled examples
needed.

**"What is an embedding?"**
A list of numbers representing something's meaning — 512 numbers per image
here. Similar things get similar numbers, so "how alike are these?"
becomes simple maths instead of a judgment call.

**"Did changing the prompt wording matter?"**
A lot. Rewriting just the `top` and `shirt` prompts to describe the actual
visual difference — collar and buttons versus none — raised accuracy from
67.6% to 79.4% on the catalog at the time. Being more emphatic in wording
made it *worse*, which I only knew because I measured.

### About recommendation systems

**"Is this collaborative filtering or content-based?"**
Content-based. It works from the item's own visual features, not from what
other users liked.

**"What about the cold-start problem?"** *(very common question)*
Content-based systems mostly avoid it. There are no users and no history
at all — a brand new catalog item can be recommended immediately, because
all I need is its photo. The trade-off is the opposite one: I can't
personalise to a specific person's taste, because I don't collect any.

**"How would you add personalisation?"**
Start by logging which suggestions people click or buy. Once there's
enough of that, a hybrid: keep the content-based pipeline for cold items,
and blend in collaborative signals as a re-ranking term — the re-ranker is
already a weighted sum, so it's the natural place to add one.

**"Why rules for pairing instead of machine learning?"**
There's no free dataset of "outfits that go together" to learn from.
Rules also stay explainable — the app literally shows the user why a match
ranked where it did, which a learned model wouldn't.

**"Why FAISS for only 49 items? Brute force is the same speed."**
Correct, at this size it is. I chose it because the *approach* has to scale
— one index per category keeps working unchanged at thousands of items.
It also means the filtering already happened before any search runs.

**"How would you scale this to a million items?"**
Swap the exact index for an approximate one (FAISS supports that directly),
precompute embeddings in a batch job rather than at request time, and
cache. The pipeline shape wouldn't change.

**"Isn't the LLM doing the real work?"**
No. It only phrases a decision the rules already made. It's given the
facts and told not to invent anything else, and if the API fails a
template sentence takes over — so the demo never breaks on a third party.

### About engineering practice

**"What was the hardest bug?"**
The one my own tests caught: pure grey was being named beige, because when
saturation is zero the hue value is meaningless. What makes it a good
story is what happened next — my first fix made measured accuracy *worse*,
so I compared the runs item by item and set the threshold from measured
values instead of intuition.

**"How do you test something with a model in it?"**
I test the parts that are deterministic — the pairing rules, colour
naming, the payment link, the fallback logic. Model inference isn't unit
tested; it's *evaluated* separately with `evaluate.py`. Mixing those two
would give me slow tests that prove nothing.

**"What does your CI do?"**
Installs the real requirements on Linux with Python 3.12, checks the
modules import, and runs the tests. It's set up that way deliberately —
both of my real production outages were Linux-only dependency problems, so
CI would have caught them.

**"Tell me about a time you were wrong."**
Two good ones: the colour fix that made accuracy worse, and my accuracy
numbers themselves — when I built proper evaluation, it showed my earlier
figures had been measured on smaller, easier subsets. I corrected them
publicly in the README rather than keeping the better-looking number.

**"What would you do differently?"**
Build the evaluation script first, not late. I tuned things for a while
based on impressions before I had a way to measure, and some of that time
was wasted.

**"What's next?"**
Targeted prompt work for `top` and `shorts`, since evaluation pinpointed
them. Then a learned colour classifier instead of nearest-neighbour
matching. And reducing memory use, which is the real constraint on free
hosting.

### Questions to ask them

Asking good questions is part of being assessed:
- "How do you measure whether a model change actually improved things?"
- "How much of the work is building new models versus keeping existing
  ones working?"
- "What does code review look like here?"

---

## 8. Things that impress — and things to avoid

**Impresses:**
- Numbers you can reproduce on the spot
- Knowing exactly *where* your system fails and why
- Admitting a limit before they find it ("I can't claim the outfits are
  good, only that the components are measured")
- A story where you were wrong and measurement corrected you
- Deleting or stopping work on something that wasn't worth it

**Avoid:**
- Quoting **79.4%** or **"~65%"** — old numbers from easier subsets
- Saying "AI does it" — be specific about which part does what
- Claiming the LLM chooses matches (it doesn't)
- Overstating scale — 49 items is a demo catalog, say so plainly
- Saying "it works perfectly"

---

## 9. Test yourself

Close everything. Answer out loud. Where you hesitate, open that file and
read it, then try again tomorrow.

1. Why doesn't this just use image similarity?
2. What happens step by step between upload and seeing matches?
3. Where do the pairing rules live, and how would you make blazers pair
   with dresses?
4. Why is style weighted lowest?
5. What does `evaluate.py` deliberately not measure, and why?
6. Why does the confidence warning trigger below 70% specifically?
7. What is cohesion in `outfit_builder.py`, and what would the outfit look
   like without it?

**Best exercise:** break something on purpose. Edit `pairing_rules.py` so
bottoms pair with bottoms, run `pytest`, watch it fail, then undo with
`git checkout pairing_rules.py`. Ten minutes of that teaches more than an
hour of reading.

---

## 10. Working on it without help

```bash
pytest                 # did I break anything?
python evaluate.py     # how well does it work right now?
streamlit run app.py   # does it work in the browser?
```

- **Errors:** read the *last* line first — it names the file and line.
- **Changed `catalog.json`?** Re-run `python build_catalog_embeddings.py`.
- **Before pushing:** run `pytest`. Green means push; CI double-checks on
  Linux.
- **Nothing is unrecoverable.** `git status` shows what changed,
  `git diff` shows exactly what, `git checkout <file>` undoes it.

**Read the code in this order:** `pairing_rules.py` (small, pure Python,
and it *is* the idea) → `color_detector.py` → `matching_engine.py` →
`app_pages/try_it_on.py`.

---

## 11. Still open

- `catalog_images/beige_blazer.jpg` has a visible **iStock watermark**.
  Replace it with a freely-licensed photo and re-run
  `build_catalog_embeddings.py`, or remove that item. Check the others too.
- Memory on free hosting: 725MB measured against a 1GB limit.

---

*Question patterns above were checked against published interview guides
for recommender systems, computer vision and CLIP, and general ML project
interviews — the "explain the problem, data, approach, metrics, result and
main challenge" framework, cold-start, and evaluation-metric questions are
the ones that come up most consistently.*
