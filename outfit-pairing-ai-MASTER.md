# Outfit-Pairing AI — Master Project File (v6)

> **If you are Claude reading this in a new chat or in Claude Code: start here.**
>
> This single file replaces all earlier versions. It is the only source of truth. Read Section 2 carefully — the code currently sitting in the project folder is in a messier state than GitHub shows, because a lot of today's work was never committed. Do not assume GitHub reflects the local files.

---

## 1. Briefing for Claude — read before responding

**Who you are working with:** Monica. Complete beginner in Python, prior Java experience. Learning Python alongside building this project.

**⚠️ Mode change as of v6 — read carefully, this overrides earlier instructions:**

Monica is now under real time pressure and needs Steps 4–11 **built quickly**, most likely using Claude Code rather than back-and-forth chat. The priority has shifted from "teach me as we go" to "get it done, let me understand it after." This is a deliberate, informed trade-off she made — not a preference to second-guess.

**What this means in practice:**

- **Build efficiently.** Don't pause for Socratic guessing on every design decision the way earlier steps did. Make the call, implement it, keep moving. Time is the constraint now.
- **But do not skip documentation of understanding entirely.** After completing each major step (4, 5, 6, 7, 8, 8B, and so on), write a short plain-language summary of **what was built and why**, in the same tone as Section 8 of this file ("Why this stack"). Append these summaries to a new section at the bottom of this file, under "Step-by-step explanations for later review." This is non-negotiable even while moving fast — it's the only thing standing between Monica and having built a project she can't explain in an interview.
- **Step 5 (the matching engine) is the one place worth slowing down for**, even under time pressure. It's the intellectual core of the project and the most likely thing an interviewer probes. Write its explanation section with extra care — the classify → filter → retrieve → re-rank logic, and why FAISS-per-category was chosen.
- **Keep this master file updated as you go**, the same discipline used for Steps 1–3B. Update Section 6's status table and Section 2's "current position" after each step. If this file goes stale, Monica loses the ability to resume, explain to her mentor, or pick this up in a fresh chat.
- **Still verify, don't assume**, when running commands — silent unsaved-file bugs happened multiple times in earlier steps and cost real time.
- **Be honest about uncertainty** in the explanation summaries too — if something is a known limitation or a judgment call, say so plainly rather than presenting it as more solid than it is. This is what makes the explanations actually useful for an interview, not just filler text.

**What this project is for:** a resume/portfolio project demonstrating practical AI engineering. Does not need to be production-perfect. Needs to be finished, free, and explainable in an interview — the explanation summaries below exist specifically to make the "explainable" part still true even though the build itself moved fast.

---

## 2. Current position — read this whole section, it is not just a status line

### What's cleanly done and committed to GitHub

- **Step 1 — Workspace.** Done.
- **Step 1B — Environment + GitHub.** Done. Python 3.12.7 via `py` launcher, venv, Git configured, repo at `github.com/Monicasdesign01/outfit-pairing-ai`, `.gitattributes` fixing binary-file misdetection.
- **Step 2 — Color detector (original version).** Committed.
- **Step 2B — Background removal (original version).** Committed.
- **Step 3 — CLIP classifier, type + style.** Committed.
- **Last confirmed commit:** `6039302` — "Split background removal into two variants: cloth segmentation for on-body photos, plain removal for flat-lay product shots"

### ⚠️ What happened after that commit — NOT yet pushed to GitHub

A long experimentation session followed the last commit. Nothing from it has been committed. The local project folder is currently **ahead of and messier than GitHub**. Here is exactly what exists locally right now and what state it's in:

**Files that contain good, working improvements — worth keeping and committing:**
- `color_detector.py` — significantly improved from the GitHub version. Now has:
  - `k=5` instead of `k=3` in the k-means call
  - An 80% "solid vs multi-color" threshold (`analyze_color()` function, replacing the old `get_dominant_color()`)
  - Groups with the same `closest_color_name()` result get merged together before the threshold check, so e.g. two different shades of red correctly combine into one "red" entry instead of being treated as separate colors
  - Alpha filtering was tightened from `alpha_flat > 0` to `alpha_flat > 128` to reduce edge-noise from soft segmentation masks (this alone did not fully fix the black-pixel bug described below — see known issues)
  - **This file's improvements are real and should be kept and committed.**

**Files created during a segmentation experiment that was ultimately abandoned:**
- `Cloth_only.py` — uses `rembg`'s `u2net_cloth_seg` model to segment upper/lower/full-body clothing from a person, instead of plain background removal
- `crop_upper_body.py` — crops a specific third of the stacked segmentation output
- `analyse_garment.py` (note: British spelling, not `analyze_garment.py`) — combines segmentation + cropping + color analysis into one pipeline, includes `decide_section()` which uses CLIP to guess upper vs lower
- `check_blank_crops.py` — a diagnostic script that measures what percentage of a cropped image is actually visible (non-transparent)

**Decision made at the end of this session: abandon the cloth-segmentation approach.** Reasons, confirmed together: too much engineering time spent on a component that isn't the intellectual core of the project (that's Step 5), for a model with real, unresolved reliability problems. Specific bugs found and NOT fixed, because the whole approach was dropped instead:

1. **The "full body" category bug.** `decide_section()` only ever returns `"upper"` or `"lower"`, never `"full"`. `u2net_cloth_seg` stacks its output as upper/lower/full, in that order. Any single-piece garment spanning the torso-to-hip area — dresses, kurtas, skirts, jumpsuits — actually gets segmented into the "full" section, which the code never looks at. Result: cropping into the wrong (empty) section for roughly 40% of test photos (6 of 15), all silently returning near-blank images that then produced meaningless color results.
2. **Flat-lay failure.** `u2net_cloth_seg` requires a human body to anchor to. Product photos with no person in frame (like a garment laid flat on a table) produce patchy, unreliable segmentation — confirmed directly on `test_06_top1.jpg`, a flat-lay vest.
3. **Unusual pose/silhouette failure.** A photo with crossed legs produced a cut-off result partway down. A sculptural, non-standard-silhouette dress (`test_14_differentdress.jpg`) produced an almost entirely blank segmentation — the model appears to rely on typical/expected clothing shapes.
4. **A genuine black-pixel bug that was never fully root-caused.** Cropped segmentation output was returning heavily black/gray/brown-dominated color results even on garments that are clearly bright colors (a yellow kurta, an olive dress, a pink skirt). Partially attributed to soft alpha edges at transparency boundaries carrying hidden `(0,0,0)` color data through the `> 0` alpha filter; raising the threshold to `> 128` did not fully resolve it, and the true root cause was not isolated before the approach was abandoned.

**Decision (revised, time-pressure call): keep it simple for now, refine later.** `color_detector.py` should always return **one** primary color — the single biggest k-means group, no 80%/threshold gating blocking the answer. This is deliberately simpler than the k=5/merge/threshold version explored earlier today: under deadline pressure, a color detector that always gives a usable answer is more valuable than one that sometimes correctly says "multi-color" but gives nothing actionable. **This is a documented, deliberate trade-off, not an oversight** — flag it as a known limitation in the README later ("color detection reports the single dominant color; does not yet distinguish a genuinely multi-color garment from a solid garment photographed with visible skin/hair/background").

Concretely: use `k=3`, take the single largest group, run it through `closest_color_name()`, return that. Drop the 80% threshold and the multi-color list output for now.

### A separate, real bug that was found and fixed this session: file encoding

`remove_background_solid.py` (an earlier name for the recovered plain-background file) was created using `git show <commit>:file.py > newfile.py` in PowerShell. This silently wrote the file in UTF-16 with a BOM, which Python's import system cannot parse (`SyntaxError: source code string cannot contain null bytes`). **Lesson for future file recovery: never use PowerShell `>` redirection to create a `.py` file. Recreate it by hand in VS Code and save normally (UTF-8), or use `git show <commit>:file.py | Out-File -Encoding utf8 newfile.py` if redirection is truly needed.** The file was ultimately deleted and recreated by hand in VS Code as `remove_background.py`, which fixed it — confirmed via `Format-Hex` showing no `FF FE` BOM and no null bytes between characters.

### ⚠️ Required cleanup before continuing to Step 4

This has **not been done yet**. Whoever picks this up next (a new chat, or Claude Code) should do this first:

1. Confirm `remove_background.py` in the project root contains the `u2netp` version (plain background removal) — verify with `type remove_background.py`, should show `new_session("u2netp")`.
2. Rewrite `color_detector.py` to the simplified version: `k=3`, return the single largest group's color name, no threshold, no multi-color list. See the revised decision above.
3. Decide whether to delete `Cloth_only.py`, `crop_upper_body.py`, `analyse_garment.py`, `check_blank_crops.py`, or simply leave them uncommitted/unused in the folder. Recommend deleting them, or committing them separately with a commit message explicitly noting they were an abandoned experiment, so the GitHub history stays honest rather than silently vanishing work that was actually done.
4. Run `git status` to see the full real diff before committing anything.
5. Commit the working `color_detector.py` and `remove_background.py` improvements with an honest message, e.g.: *"Improve color detection (k=5, merge same-named clusters, 80% solid threshold); attempted cloth segmentation for on-body photos, found it unreliable across flat-lays/unusual poses/full-body garments, reverted to plain background removal"*
6. Push.

### Step 3B — accuracy check, real numbers and findings (usable for the README later)

15 real test photos were gathered (`test_images/` folder), including deliberately ambiguous combo outfits.

**Type classification:** roughly 12 of 15 correct or reasonably defensible on manual review. One label-wording fix was tested and confirmed to work: renaming the "hoodie" label to "a photo of a hooded zip-up sweatshirt" fixed a real misclassification (a hoodie had been called "jacket").

**Style classification:** weaker than type, as expected going in. Roughly 10 of 15 correct on the original label wording. A label-wording "improvement" (making fitted/loose more emphatic — "tight fitted clothing" / "loose baggy clothing") was tested and made results *worse*, not better — nearly everything started reading as "fitted" regardless of actual fit. This is a real, useful finding: more emphatic wording made one label dominate rather than improving accuracy. **Recommendation: revert style labels to the original, milder wording** (`"a photo of fitted clothing"`, `"a photo of loose clothing"`, `"a photo of casual clothing"`, `"a photo of formal clothing"`), keep `"a photo of edgy alternative streetwear"` as a genuinely useful fifth addition since it correctly captured at least one garment (a plaid vest) that didn't fit the original four categories.

**Real, specific pattern identified in style misclassifications (confirmed through careful manual re-review, not just accepted from the first pass):** CLIP tends to over-predict "formal" for any single-piece garment (dress, kurta) based on overall silhouette, even when a specific styling detail — fringe trim, an unconventional print — signals otherwise to a human eye. This is a strong, specific, honest sentence for the README's known-limitations section.

**Color detection multi-color handling was informally validated:** confirmed working correctly on a striped shirt (correctly reported as multi-color, dominated by white/cream/gray as expected for thin pinstripes) and correctly merging near-identical shades on a solid cream shirt.

### ✅ Cleanup checklist — completed 2026-09-02

All six items above are done: `remove_background.py` confirmed on `u2netp`; `color_detector.py` reverted to the simplified `k=3`, single-largest-group, no-threshold version; the abandoned segmentation files (`Cloth_only.py`, `crop_upper_body.py` were already committed, `analyse_garment.py` and `check_blank_crops.py` committed separately with an explicit "abandoned experiment" note rather than deleted, so the debugging work stays visible in history); working improvements committed with an honest message; pushed to GitHub.

**Update, 2026-09-02, later the same day:** all four abandoned segmentation files (`Cloth_only.py`, `crop_upper_body.py`, `analyse_garment.py`, `check_blank_crops.py`) have now been **deleted from the project entirely** via `git rm` (commit `c8e1c1f`), at Monica's request, to keep the working tree clean going forward. They no longer exist locally or on GitHub's current `main` — the file lists and bug descriptions below are kept as historical record of what was tried and why it was abandoned; they are not a description of what's currently in the project folder. The reasoning is preserved in git history across three commits: `6039302` (original addition of `Cloth_only.py`/`crop_upper_body.py`), `3b40b5a` (addition of `analyse_garment.py`/`check_blank_crops.py` with the abandoned-experiment explanation), and `c8e1c1f` (final removal).

### Step 4 — done, 2026-09-02

`catalog.json` and CLIP embeddings built. **Placeholder catalog, by deliberate choice**: no real catalog photos existed yet, so the 15 `test_images/` photos were reused as a stand-in catalog (copied into `catalog_images/`, `test_images/` left untouched) so the full pipeline could be built and actually run today rather than blocked on sourcing real product photos. Swap in real catalog photos later without changing any code — just replace `catalog_images/` and regenerate `catalog.json` + embeddings.

Full write-up in Section 11 below. Short version: `catalog_images/` was explicitly un-ignored in `.gitignore` (the exact risk flagged in Section 7); `build_catalog_embeddings.py` background-removes each catalog photo once (cached to `catalog_images/nobg/`) and computes a CLIP image embedding for it, all cached to `catalog_embeddings.npz`. Found and fixed a real bug: the installed `transformers` version (5.15.1) changed `CLIPModel.get_image_features()` to return a `BaseModelOutputWithPooling` object instead of a plain tensor — the actual 512-dim embedding is `.pooler_output`. Verified by inspecting the object directly rather than assuming the remembered API still held.

### Step 5 — done, 2026-09-02

Full write-up and confirmed test results in Section 11 below (`matching_engine.py` + `pairing_rules.py`). Also installed `faiss-cpu==1.15.0` (added to `requirements.txt`, regenerated cleanly in UTF-8 while at it — the existing `requirements.txt` had the same UTF-16/BOM corruption already documented for `.py` files in this section, likely from an earlier `pip freeze > requirements.txt` in PowerShell).

### Step 6 — done, 2026-09-02

`test_pipeline_end_to_end.py` runs `find_matches()` across all 18 available photos (16 in `test_images/` + `test.jpg` + `test2.jpg`); 18/18 completed without error. Full write-up in Section 11 below.

### Step 7 — done, 2026-09-02, LLM path confirmed live 2026-09-04

`explanation.py` (Gemini `gemini-3.5-flash-lite` via `google-genai`, falling back to a hand-written template). A real `GEMINI_API_KEY` is now configured as a persistent Windows user environment variable and confirmed working end-to-end - see the 2026-09-04 update in Section 11 below, including a real model-name fix (`2.5-flash-lite` → `3.5-flash-lite`, the old one was retired) and a VS-Code-restart gotcha worth knowing about if another key ever needs adding (see Section 3).

### Next action

Begin **Step 8: the Streamlit app** — Shop page + Try It On page, native multi-page mechanism, category filter. Before writing UI code, confirm which Streamlit navigation API currently applies (`pages/` folder vs. `st.navigation`/`st.Page`) against current docs — this file already flags that the interface has changed across versions and warns not to build against a remembered API.

---

## 3. Environment — confirmed working, do not re-derive

| Detail | Value |
|---|---|
| Project path | `D:\outfit-pairing-ai` |
| Machine | Windows Dell laptop, Intel UHD Graphics (no NVIDIA GPU — confirmed via `Get-CimInstance Win32_VideoController`) |
| Python for this project | **3.12.7**, installed alongside 3.14.7, via `py -3.12` |
| Virtual env | `py -3.12 -m venv venv` |
| Terminal | Mix of Command Prompt and PowerShell in use; both work. Activation: `venv\Scripts\activate.bat` (cmd) or `venv\Scripts\Activate.ps1` (PowerShell, may need `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first, already resolved once and works now) |
| pip | bare `pip` not on PATH — always `python -m pip install ...` |
| PyTorch | **CPU-only build**, installed via `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` — confirmed correct choice, no NVIDIA GPU present |
| Git | 2.55.0, configured as user "Monica" / `monicaraoa219@gmail.com` |
| `.gitattributes` | Added to fix binary-file misdetection for `.txt/.py/.md/.json` on Windows |

**Every new terminal session starts with:**
```
D:
cd outfit-pairing-ai
venv\Scripts\activate.bat
```

**File-recovery warning (see Section 2):** never use PowerShell `>` redirection to write a `.py` file — it produces UTF-16 with a BOM that Python cannot import. Use VS Code directly, or `Out-File -Encoding utf8`.

**API keys (e.g. `GEMINI_API_KEY`), confirmed working 2026-09-04:** set as a persistent Windows user environment variable, run once in your own terminal, never through Claude/chat, never committed: `[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-actual-key", "User")`. `.gitignore` already excludes `.env` as a backup if a `.env` file is ever used instead. **Gotcha:** this does not become visible to an already-open VS Code / Claude Code session, even in a brand-new terminal panel inside it — the process tree hosting the tool calls only re-reads the environment at its own startup, not on a Windows registry change. **Fully quit and reopen VS Code** (not just open a new terminal) for a newly-set key to actually be visible.

---

## 4. What the project does

Concept e-commerce site, vintage/celebrity/new clothing. Customer uploads a photo of something they own, gets matched with complementary catalog items, plus a plain-language explanation.

**Two sections in one Streamlit app** — the accurate terminology for this, confirmed via research: **tabbed navigation** / **multi-page app** (the literal Streamlit mechanism), following a **super-app pattern** with the AI feature as one **vertical**, the way Swiggy's Instamart sits inside the main Swiggy app. When explaining out loud: *"It uses tabbed navigation — like Swiggy, where the main ordering screen and Instamart sit as separate sections in one app."* Not "split screen" — that means simultaneous side-by-side display, which this isn't.

- **Shop page** — browsable catalog, cosmetic only, no cart/checkout
- **Try It On Your Clothes** — the AI feature: upload, get ranked matches, explanations, category filter

Both paths end at a UPI payment link, handled manually.

### Step 8 build detail — how the multi-page app should actually be built

This didn't survive the file's condensing earlier and needs to be explicit for whoever builds Step 8:

- **Use Streamlit's built-in multi-page mechanism** — do not hand-roll page-switching with session state or if/else blocks. Streamlit has a real, native way to do this (classic: a `pages/` folder where each file becomes a page; newer Streamlit versions also support `st.navigation` / `st.Page`). **Confirm which API applies against current Streamlit documentation when this step is reached** — the interface has changed across versions, don't build against a remembered API without checking.
- **Shop page contents:** a grid or list of catalog items pulled from `catalog.json` (built in Step 4) — image, name, price, a Buy button that opens the UPI link (`upi://pay?pa=...`). No cart, no accounts, no checkout logic. It exists purely to give the AI feature a realistic storefront around it.
- **Try It On Your Clothes page contents:** file upload widget, run the full pipeline (background removal → color/type/style detection → filter → retrieve → re-rank → explanation), display the uploaded item alongside ranked matches with their explanations.
- **Category filter:** after matches are returned, show a multi-select widget listing the categories present in the results (e.g. shirts, crop tops, kurtis) so the user can untick ones they don't want, and the displayed list re-filters instantly. This is a real, deliberate feature — not optional polish — because pure ranking assumes the system knows the user's taste, and it doesn't.
- **Spin wheel: deliberately excluded.** This was considered and dropped earlier in the project. A random pick isn't AI and would undercut the project's actual argument that every recommendation is deliberately ranked. Do not add it back in without discussing it first.

---

## 5. The core idea — the single most important thing to understand

**Items that look similar are not the same as items that go together.**

**classify → filter → retrieve → re-rank:**
1. Classify the uploaded item
2. Filter the catalog to only categories that pair with it
3. Retrieve visually similar items *within that filtered set*
4. Re-rank using colour-theory and silhouette rules

This is the intellectual core of the project and the single best thing to explain in an interview.

---

## 6. Full step list

| Step | What | Status |
|---|---|---|
| 1 | Set up workspace | done |
| 1B | Virtual environment + Git/GitHub | done |
| 2 | Colour detector + colour naming | done, simplified version committed (k=3, single largest group, no threshold) |
| 2B | Background removal | done (plain u2netp version — segmentation abandoned, see Section 2) |
| 3 | Garment classifier (CLIP) | done |
| 3B | Accuracy check | done, real numbers in Section 2, needs README write-up eventually |
| 4 | Build catalog.json + embeddings | done (placeholder catalog — reused test_images/ photos, see Section 2) |
| 5 | Matching engine (classify to filter to retrieve to re-rank) | done — the intellectual core, tested end-to-end on two real cases |
| 6 | Test pipeline end-to-end, text only | done — 18/18 uploads ran without error |
| 7 | RAG explanation layer + fallback | done — Gemini (`gemini-3.5-flash-lite`) + template fallback; real key confirmed working 2026-09-04, all 11 test matches returned `source == "llm"` |
| 8 | Streamlit app — Shop, Try It On, category filter | not started |
| 8B | Deploy free on Streamlit Community Cloud | not started |
| 9 | Stretch — 3D mannequin | not started |
| 10 | Optional — UPI buy link | not started |
| 11 | README + architecture diagram | not started |

---

## 7. Known risks — updated

1. **`.gitignore` will break deployment** unless catalog images are explicitly un-ignored at Step 4 (`!catalog_images/`).
2. **FAISS needs one index per category**, not a single filtered index — plan this at Step 5.
3. **Deployment memory (Streamlit Cloud, ~1GB)** — CPU-only PyTorch already in use (good), plan for smallest CLIP variant, background removal as an optional toggle if needed.
4. **Catalog image copyright** — use Unsplash/Pexels or Monica's own photos, not scraped brand photos.
5. **Style detection is measurably weaker than type detection** — confirmed with real numbers in Section 2, not just a prediction anymore. Weight style lower in Step 5 re-ranking; document honestly.
6. **Colour-theory rules will need hand-tuning** — expected, not yet encountered since Step 5 hasn't started.
7. **NEW — garment segmentation for on-body photos was attempted and abandoned.** See Section 2 for full detail. Plain background removal is used for all photos going forward; some color-detection imprecision on garments where skin/hair are visible in the photo is accepted as a documented limitation rather than solved.
8. **NEW — PowerShell `>` file redirection corrupts `.py` files** (UTF-16 BOM). Always create/recreate Python files through VS Code directly.

---

## 8. Why this stack — plain language, for interview answers

*(Unchanged from previous version — still accurate.)*

**OpenCV k-means for colour, not AI:** finding the most common colour is counting and grouping, a solved maths problem.

**CLIP, not a self-trained classifier:** already learned image-word relationships from the internet; zero-shot, no training data needed.

**FAISS, one index per category:** free, local, fast; filtering handled by which index you query.

**Rules for pairing, not ML:** no free dataset for "good outfit" exists; rules stay explainable.

**LLM for explanation text only:** phrasing is what language models do well; deciding the match is not, since it would invent products.

**Streamlit over a real website:** whole interface stays in Python; deliberate scope choice.

**Plain background removal over cloth segmentation (NEW):** segmentation promised cleaner color detection by isolating garments from skin/hair, but proved unreliable across flat-lays, unusual poses, and full-body garments — a real engineering trade-off, decided against after hands-on testing rather than assumed upfront. Good interview material: shows judgment about when to stop optimizing a non-core component.

---

## 9. Still genuinely open

- Which free LLM for Step 7 — check what's current when reached
- Whether CLIP fits Streamlit Cloud's memory limit in Step 8B — find out by trying
- Exact PyTorch/FAISS install commands — confirm against current docs, not memory
- ~~Whether to delete or archive the abandoned segmentation files~~ — resolved 2026-09-02: deleted via `git rm` (commit `c8e1c1f`), see Section 2

---

## 10. Housekeeping rule

**This file is the only source of truth.** Update it whenever the plan changes — do not leave changes living only in a chat conversation.

**When resuming in a new chat or in Claude Code: share THIS file, and nothing else.** If using Claude Code, explicitly say "read outfit-pairing-ai-MASTER.md in this project and continue from where it says we are" — Claude Code will not automatically know about this file or this conversation otherwise.

There is also a separate Word document (`Outfit-Pairing-AI-Overview-v2.docx`) written for Monica's academic guide/mentor — for humans reviewing the project, not for briefing Claude. If the two ever disagree, this file wins.

---

## 11. Step-by-step explanations for later review

> **For Claude/Claude Code: append one dated entry here after finishing each step, before moving to the next.** Plain language, aimed at Monica reading this later to understand what was built and why — not code comments, not a changelog. Every entry must explicitly cover:
> - **What was built** in this step, in plain terms
> - **Which model, library, or tool was used** (name it specifically — e.g. "CLIP via HuggingFace transformers," "FAISS," "a specific rembg model")
> - **Why that one was chosen over the realistic alternatives** — what else could have been used, and the actual reason it wasn't
> - **Any real limitation or judgment call** made along the way, stated honestly
>
> Keep each entry to a few short paragraphs — thorough enough to actually explain the step, not so long Monica won't read it.

*(Entries begin below as each step is completed, one dated entry per step, each covering what was built, which model/tool was used, and why. If this section is still empty, no steps have been built yet under the v6 fast-build mode.)*

### 2026-09-02 — Step 4: catalog.json + embeddings

**What was built:** A `catalog.json` file listing 15 catalog items (id, filename, name, category, price), a `catalog_images/` folder holding the actual photos, and a script (`build_catalog_embeddings.py`) that turns every catalog photo into a numeric "fingerprint" (its CLIP embedding) and saves all 15 of them to one cache file, `catalog_embeddings.npz`, so they never need recomputing on a customer visit.

**Which tools were used:** CLIP (`openai/clip-vit-base-patch32`) via HuggingFace `transformers`, the same model already used for Step 3's type/style classification — but a different function on it. Step 3 asks CLIP "which of these text labels fits this photo best?" Step 4 asks it "turn this photo into a vector of numbers" (`get_image_features`), with no text involved at all. That vector is what similarity search compares in Step 5. Background removal (`rembg`, `u2netp`) is run once per catalog photo first, and cached to `catalog_images/nobg/`, so catalog embeddings are computed on the same kind of clean, background-free image a customer's uploaded photo will be — comparing garment-to-garment, not garment-plus-background-to-garment.

**Why this, not an alternative:** embeddings could be recomputed live every time someone visits the site, but that's real, unnecessary CPU work on a free-tier deployment for something that never changes per catalog photo — computing once and caching to a file is the obvious efficient choice. A `.npz` file (NumPy's compressed array format) was used over, say, storing embeddings inside `catalog.json` itself, because catalog.json stays small and human-readable as hand-built metadata, while the embeddings are a separate, larger, purely numeric cache that Step 5 loads separately.

**Real judgment call — placeholder catalog:** there were no actual catalog product photos in the project yet (no photos from Unsplash/Pexels, none of Monica's own). Rather than block Step 4 entirely on sourcing real photos, the existing 15 `test_images/` photos (already used for the Step 3B accuracy check) were reused as a stand-in catalog, copied into `catalog_images/` so `test_images/` itself stays untouched. This means the current catalog.json entries (names, categories, prices) are placeholder data invented to make the pipeline runnable end-to-end today, not real inventory — swap in real photos and metadata later without touching any code, then just rerun `build_catalog_embeddings.py`.

**Real bug found and fixed:** the installed `transformers` version (5.15.1) changed what `CLIPModel.get_image_features()` returns — it now comes back as a `BaseModelOutputWithPooling` object instead of a plain tensor, with `last_hidden_state` (per-patch features, shape `[1, 50, 768]`) alongside the actual `pooler_output` (the real 512-dim CLIP embedding, shape `[1, 512]`). The first version of the code silently used the wrong field and produced a broken embedding shape. Caught by actually inspecting the object's contents and checking the installed library version, not by assuming a remembered API still applied — exactly the "verify, don't assume" discipline this file asks for.

**Known limitation, stated honestly:** the catalog is currently 15 placeholder items with made-up names/prices, reused test photos rather than real product photography. This is fine for building and demoing the pipeline, but the README (Step 11) should say plainly that the deployed catalog needs real, appropriately-licensed photos before this is a genuine e-commerce demo rather than a technical proof of concept.

### 2026-09-02 — Step 5: the matching engine (classify → filter → retrieve → re-rank)

**What this step is:** the intellectual core of the whole project — Monica's own upload gets turned into ranked catalog matches, not by "find the most visually similar item overall" (which would just return more of the same thing), but by first working out what pairs with the uploaded item at all, and only then searching for good-looking options within that narrowed set. The four stages, in the order they actually run:

1. **Classify** — CLIP (same model and same zero-shot approach as Step 3) looks at the uploaded photo and picks its category from a fixed list matching the catalog's own categories (dress, blazer, jeans, skirt, top, shirt, kurta, hoodie).
2. **Filter** — a small hand-written table says which categories are allowed to pair with which. Bottoms (jeans, skirts) pair with tops, shirts, kurtas, hoodies, and blazers, but never with each other or with dresses. Dresses, being a complete outfit on their own, only pair with a blazer layered over them. This is plain rule logic, not a model — there's no free "what pairs with what" dataset, and rules stay explainable in an interview.
3. **Retrieve** — FAISS searches for the most visually similar catalog items, but only within the categories the filter step allowed, using **one FAISS index per category** rather than one big index with results filtered afterward. The reason: FAISS has no clean built-in way to say "search everything, but only actually consider these N items" — a single shared index would need extra bookkeeping to discard disallowed results after the fact, whereas a separate index per category means the filtering already happened before any search runs, so the query only ever touches items it's allowed to return.
4. **Re-rank** — the shortlist FAISS returns gets reordered using two more hand-written rule tables: a colour-relationship table (which colours are considered complementary, analogous, or safely neutral against each other) and a silhouette table (e.g. a fitted top balances a loose bottom, matching formality levels pair better than mismatched ones). Both scores get combined with the FAISS similarity score into one final ranking.

**Honest note on FAISS at this scale:** the catalog is 15 placeholder items split across 8 categories — some categories have one or two items in them. At this size, a plain brute-force comparison (just computing distance to every vector directly, no index at all) would be exactly as fast as FAISS, probably faster once you count index-building overhead. FAISS is used anyway because the *code* — one index per category, built once, queried per upload — is what needs to scale, not today's tiny catalog. If the catalog grows to hundreds or thousands of real items later, this same code keeps working without a rewrite; a brute-force version would eventually need replacing. This is a deliberate engineering choice made for where the project is headed, not because it's faster today — worth saying exactly this way if asked about it in an interview, rather than overstating a performance win that doesn't exist yet at this size.

**Why the weighting favors colour over style/silhouette:** Section 3B's accuracy check already found CLIP's style classification measurably weaker than its type classification (roughly 10/15 vs 12/15 correct, with a specific known failure pattern of over-predicting "formal"). Building the re-rank step to weight colour more heavily than style/silhouette is a direct, deliberate response to that earlier, real finding — not a new assumption.

**Known limitations / judgment calls, stated plainly:**
- The colour and silhouette rule tables are hand-tuned by judgment, not derived from any dataset — they encode reasonably standard colour-theory/styling conventions, but they are a starting point to refine later, not a validated ground truth.
- Catalog items now carry a `color` and `style` field computed by running the same colour detector and CLIP style classifier used elsewhere in the project once per catalog photo (cached into `catalog.json`, the same "compute once" principle as the embeddings) — so their accuracy inherits the exact same limitations already documented for Steps 2 and 3.
- The catalog is still the Step 4 placeholder (reused test photos), so re-ranked results are a demonstration of the ranking logic working correctly, not a claim that these specific pairings are genuinely good outfits from a real inventory.

**Confirmed working, 2026-09-02:** ran the full pipeline end-to-end on two real cases rather than assuming the code was correct once it ran without errors. Uploading `test_03_jeans.jpg` (bottom) correctly retrieved only tops/shirts/kurtas/hoodies/blazers, and re-ranking visibly reordered the FAISS results — "Everyday Cotton Shirt" (0.746 similarity) moved ahead of "Relaxed Fit Top" (0.848 similarity, the highest raw similarity) once colour and style scoring were added, confirming the re-rank step does real work rather than just echoing FAISS's order. Uploading `test_9_dress.jpg` correctly returned only the one blazer in the catalog, matching the "dress only pairs with a blazer" rule. `classify_uploaded_item`, `get_paired_categories`, `build_category_indices`, `retrieve_candidates`, and `rerank` all live in `matching_engine.py`; the rule tables live in `pairing_rules.py`.

**Known limitation found while demonstrating the colour rule, 2026-09-02:** asked to show the colour-scoring rule visibly differentiating results (not just running without error), tracing through the catalog's actual colours found a real gap: `color_score()` treats `black/white/gray/cream/navy` as neutral and auto-scores 1 whenever either side is neutral, and in the current 15-photo placeholder catalog *every* bottom (both jeans, the skirt) and most tops/the blazer came back neutral — the only non-neutral colours (brown, yellow, olive) all happen to sit on garments in categories a bottom-family upload reaches. Tested two real uploads (navy jeans, red shirt) and both landed in all-neutral candidate pools, so `color_score` stayed constant at 1 in both live demos — not a bug, just an artefact of this placeholder catalog's colour distribution. Confirmed the rule itself works correctly by calling `color_score()` directly against the catalog's real non-neutral colours: a hypothetical blue upload scores 2 (complementary) against both the brown top and yellow kurta, red scores 0 against all three, green scores 1 (analogous) against the olive dress only. **This is exactly the kind of thing the Step 4 catalog rebuild (swapping in real, colour-diverse product photos) should fix** — a real catalog with non-neutral bottoms/outerwear would let this rule visibly do its job in an actual live demo, not just a direct function call.

### 2026-09-02 — Step 7: the RAG explanation layer + fallback

**What was built:** `explanation.py`, with one job — turn Step 5's already-decided match (category, colour, style, and *why* they were scored the way they were) into one short, friendly sentence for the customer. `get_explanation(uploaded, match)` tries an LLM first and falls back to a hand-written template if the LLM call fails for any reason at all (no API key configured, network issue, rate limit, unexpected response) — never raises, always returns something usable. The LLM is only ever given facts Step 5 already decided (e.g. "colour relationship: complementary") and is explicitly instructed not to invent any other product detail — it phrases, it doesn't decide, exactly matching this project's own stated principle in Section 8 ("LLM for explanation text only... deciding the match is not, since it would invent products").

**Which tool was used, and why:** Google's Gemini API, via the `google-genai` Python package, model `gemini-2.5-flash-lite`. This wasn't assumed from memory - Section 9 explicitly flagged "check what's current when reached," so a live web search was run before writing any code. As of the search (2026-09-02): Gemini's free tier requires no credit card and includes several current models (2.5 Flash-Lite among them, chosen here for its higher free-tier rate limit over plain Flash); the older `google-generativeai` package is deprecated in favour of the unified `google-genai` SDK used here. The `Client(api_key=...)` and `client.models.generate_content(model=..., contents=...)` calls were confirmed to actually exist on the installed package version via direct inspection before being used in code - the same "verify, don't assume" discipline that caught the `transformers` API change in Step 4.

**Why a fallback at all:** free LLM tiers can rate-limit, go down, or simply not have a key configured (which is the actual state of this project right now - no `GEMINI_API_KEY` exists yet). A live demo that hard-fails because a third-party API hiccuped would be a bad look in an interview setting. The template fallback reuses two small new rule-explanation helpers in `pairing_rules.py` (`explain_color_relationship`, `explain_silhouette_relationship`) that mirror `color_score()`/`silhouette_score()`'s exact same logic but return a descriptive label ("complementary", "neutral", "balanced", etc.) instead of a number, since a sentence needs to say *why*, not just carry a score.

**Verified, not just assumed working:**
- Confirmed the template path alone across complementary, analogous, same-colour, neutral, and no-relationship cases - all five produce sensible, readable sentences.
- Deliberately set an invalid `GEMINI_API_KEY` and confirmed the code catches the failure and falls back to the template rather than crashing - the exact scenario the fallback exists for.
- Ran the whole thing against real Step 5 output (`test_explanations_end_to_end.py`, two real uploads, several real matches each) rather than only hand-built sample data.
- Found and fixed one real, if minor, bug during this: "This jeans pairs well..." reads wrong because "jeans" is plural-only - added a small exception (`_subject_phrase()`) so it correctly reads "These jeans pair well...".

**Known limitations, stated plainly:**
- No `GEMINI_API_KEY` is configured for this project yet, so every explanation seen so far - in this write-up and in testing - has come from the template fallback, not the LLM. The LLM code path is written and its method calls verified to exist, but has not actually been exercised against a real, working key. Whoever adds a key later should do one real end-to-end check that `get_explanation()` actually returns `source == "llm"` before assuming that path works in practice, not just in theory.
- The template's phrasing is serviceable but repetitive by hand-written-template nature (especially the neutral-colour case, which is by far the most common outcome given Section 2's colour-demo-limitation finding above) - the LLM path exists specifically to make this read less templated once a key is added.
- Gemini's exact free-tier model lineup and rate limits are known to shift over time (the same search that found `gemini-2.5-flash-lite` also surfaced a December-2025 rate-limit reduction on plain Flash) - if this model name stops working later, that's an external API change, not a bug in this code, and the fix is a one-line constant update (`GEMINI_MODEL` in `explanation.py`).

**Update, 2026-09-04 — LLM path confirmed working with a real key.** Monica added a real `GEMINI_API_KEY`, set as a persistent Windows user environment variable (`[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "...", "User")`, never typed into chat or committed - `.env` was already in `.gitignore` as a backup, though an env var was used instead). One real setup snag, worth recording: the key wasn't visible right away even in a freshly-opened terminal, because the actual parent process hosting these tool calls (VS Code / the Claude Code session) had started *before* the registry change, and Windows only pushes persistent env-var changes to processes that re-read the environment at their own startup - not to an already-running process tree, no matter how new a terminal window opened inside it looks. Fully restarting VS Code fixed it.

Once the key was live, the very first real call failed with a genuine external API-drift bug - exactly the kind anticipated in the limitation above, and resolved exactly the way predicted: `google.genai.errors.ClientError: 404 NOT_FOUND ... This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemini-3.5-flash-lite`. Confirmed the replacement model actually works with a direct call first, then updated the one `GEMINI_MODEL` constant in `explanation.py` to `"gemini-3.5-flash-lite"`. Reran `test_explanations_end_to_end.py` against real Step 5 output (two uploads, 11 total matches) and every single one now returns `source == "llm"`, not the template - e.g. *"This gray casual Everyday Cotton Shirt is a great neutral match that completes your navy casual jeans!"* The phrasing is noticeably less repetitive than the template version, as expected. The fallback path was not lost in the process - it's still there for whenever the API is unavailable, just no longer the *only* path actually exercised.

**What was built:** `test_pipeline_end_to_end.py`, a plain-text integration test that runs the full Step 2-5 pipeline (`find_matches()`, via `matching_engine.py`) on every real photo available — all 16 files in `test_images/` plus the two leftover generic photos (`test.jpg`, `test2.jpg`) that were never copied into the catalog — and prints each upload's classified category/colour/style plus its ranked matches. Each upload is wrapped in its own try/except so one failure wouldn't stop the run, and a final summary line counts how many completed without error.

**Which tools were used:** no new model or library — this step is deliberately just plain Python calling code already built in Steps 2-5 (`rembg`, CLIP via `transformers`, FAISS, the hand-written rule tables), run in a loop with text output. That's the whole point of doing this step before Step 8: confirm the pipeline holds together as one flow while debugging is still just reading printed text, not clicking through a Streamlit app.

**Why this, not an alternative:** the alternative would be skipping straight to building the Streamlit UI and discovering integration problems there, where every fix means re-running the app in a browser instead of just re-reading a terminal. Doing a plain-text pass first is standard practice and explicitly what the master plan called for.

**Result: 18/18 uploads ran through the full pipeline without error.** No crashes, no leftover temp files (every background-removed temp image was created and cleaned up correctly across all 18 runs, confirmed by checking the filesystem afterward rather than assuming). A few classifications show the already-documented Step 3B accuracy limits playing out on new photos - e.g. `test_12_shirt.jpg` was misclassified as "kurta", and combo photos like `test_10_topAndJeans.jpg` and `test_02_BlazerAndSkirt.jpg` only ever get one single-label category, since the classifier has no way to say "this photo shows two garments." Both are expected, already-known limitations resurfacing, not new bugs introduced by this step - Step 6's job was confirming the pipeline *runs* end-to-end, not re-measuring accuracy (Step 3B already did that separately).

**Known limitation, stated plainly:** this test reuses the same photos already sitting in the catalog for 16 of its 18 cases, so it mostly proves the code path doesn't crash rather than simulating genuinely novel customer photos - only `test.jpg` and `test2.jpg` are real "the pipeline has never seen this exact photo as a catalog item" cases. That's an acceptable scope for this step (confirming the flow works, text only), not a claim that this constitutes real-world validation.

---

## 12. Technology used per step — quick reference

| Step | What it does | Technology / model used | Why this, not an alternative |
|---|---|---|---|
| 1 / 1B | Workspace, environment, version control | Python 3.12.7, `venv`, Git, GitHub | Standard, free, reproducible setup — nothing project-specific here |
| 2 | Find a garment's dominant colour | OpenCV, k-means clustering (k=3, single largest group, no multi-color detection) | Simplicity chosen deliberately under time pressure — always returns a usable answer rather than sometimes correctly flagging multi-color but giving nothing actionable; documented as a known limitation to revisit later |
| 2B | Remove photo background before analysis | `rembg`, `u2netp` model | Lightweight (~5MB) pretrained segmentation model; the default heavier rembg model caused a real out-of-memory error on this machine |
| 3 | Identify garment type and style from a photo | CLIP (`openai/clip-vit-base-patch32`) via HuggingFace `transformers`, PyTorch (CPU build) | Zero-shot — no training data needed; CPU build used since the machine has no NVIDIA GPU |
| 3B | Measure how accurate Step 3 actually is | Manual scoring against 15 real test photos | No shortcut for this — accuracy has to be checked against real, human-verified answers |
| 4 | Build the searchable catalog | `catalog.json` (hand-built data) + CLIP embeddings, cached to a file | Embeddings only need computing once per catalog photo, not on every customer visit |
| 5 | Find and rank complementary items | FAISS (`faiss-cpu`, one `IndexFlatIP` per category) + hand-written pairing/colour/silhouette rule tables (`pairing_rules.py`) | FAISS is free/local; one index per category avoids needing to filter a shared index after the fact, and scales unchanged if the catalog grows later (a brute-force comparison would be equally fast at today's 15-item size — FAISS is chosen for where this is headed, not a speed win yet). Rules are used for pairing/colour/silhouette because no free dataset of "good outfits" exists, and rules stay explainable; colour is weighted above style in re-ranking since Step 3B found style classification measurably weaker |
| 6 | Confirm Steps 2–5 work together | Plain Python, text output only | Debugging is easier before any visual layer is added |
| 7 | Generate a plain-language reason for each match | Google Gemini (`gemini-3.5-flash-lite`) via `google-genai` + a hand-written template fallback | Chosen after checking current free-tier options rather than from memory; the fallback avoids a live demo failing if the API is down or rate-limited. Confirmed working with a real key 2026-09-04 (after fixing one real model-name drift: `2.5-flash-lite` was retired mid-project in favor of `3.5-flash-lite`) |
| 8 | User-facing app: Shop page + Try It On page | Streamlit, native multi-page mechanism | Keeps the whole interface in Python; deliberately simpler than a hand-built React/Flask site so effort stays on the AI pipeline |
| 8B | Put the app online with a public link | Streamlit Community Cloud (free tier) | Free hosting; smallest CLIP variant and CPU-only PyTorch used to fit memory limits |
| 9 (stretch) | Show the outfit on a 3D figure | Three.js + a free `.glTF` mannequin, flat texture overlay | Real cloth simulation is specialist paid software; this gets visual impact without that cost |
| 10 (optional) | Let a few real people buy an item | A UPI payment link, manual order tracking | A full payment gateway is unnecessary overhead for 2–10 manual orders |
| 11 | Document the project | Markdown README, architecture diagram | Most recruiters spend under a minute on a repo — this is the highest-value hour in the project |



