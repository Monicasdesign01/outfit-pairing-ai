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

### Step 8 — built, verified, and performance-fixed 2026-09-04

`app.py` + `app_pages/shop.py` + `app_pages/try_it_on.py` + `shop_utils.py`, using Streamlit's current `st.Page`/`st.navigation` (confirmed against live docs, not memory - see Section 11). Shop page and the Try It On upload-to-results flow both confirmed actually working via a real headless-browser test (Playwright), not just code review. Found a genuine performance issue (a 9-match upload took 244s, 222s of it the explanation-generation loop, almost certainly Gemini free-tier rate-limiting) and fixed it by design decision: only the top 3 matches now get a live Gemini explanation, the rest use the template fallback outright. Verified the fix actually skips the API call for the rest (proven via 0.00s timing, not assumed), with an honest caveat that Gemini's own rate-limit recovery time is outside this code's control - see Section 11 for the full trail.

### Catalog expanded with real photos — done, 2026-09-05

The 19 real photos Monica added to `catalog_images/` are now properly part of the catalog: `catalog.json` has an entry for each (`cat_16` through `cat_34` - id, filename, name, category, price), and `build_catalog_embeddings.py` has been rerun so every one of the now-34 catalog items has a computed embedding, colour, and style, the same as the original 15. Two categories genuinely didn't exist before and needed adding - **pants** and **shorts** - both treated as bottoms with identical pairing behaviour to jeans/skirt (no meaningful styling difference in this rule set between "jeans" and "pants" as a bottom). `pairing_rules.py` was refactored slightly to build `PAIRING_RULES` from two sets (`BOTTOMS`, `TOPS`) instead of listing every category's pairs by hand, specifically so adding a category doesn't mean manually updating five different lines - verified this produces byte-for-byte the same rules as before for the original 8 categories, not just assumed. Full write-up in Section 11.

**Real content decision made without Monica's input, flagged plainly:** names and prices for the 19 new items were inferred from filenames (e.g. `pink_skirt.jpg` → "Pink Skirt", ₹1499) since no real names/prices were provided. These are placeholders in the same spirit as the original 15, just for genuinely real photos this time - worth a real pass to confirm or correct before this goes anywhere customer-facing. **Update 2026-09-05: Monica reviewed the guessed names and accepted them as-is for now** (and doesn't mind the placeholder prices) - the placeholder names/prices are now a deliberate, confirmed decision, not an unreviewed gap.

### Accuracy fixes: CLIP category classification + colour detection — done, 2026-09-05

Monica pushed back on treating a real observed bug (uploading pants got classified as skirt) as a permanent limitation. Measured current accuracy, tried several fixes, kept only what measurably helped, discarded what didn't - full numbers and method in Section 11. CLIP category accuracy: 67.6% → 79.4% (reworded only the `top`/`shirt` prompts - every other reword tried made things worse). Colour detection: fixed two real bugs (a missing "orange" reference color, and neutral grays being matched to a random hue due to a sparse reference palette) found by inspecting actual raw RGB values, not guessing.

### Step 8B deployment prep — done, 2026-09-05; one serious unresolved risk found

Before handing off the actual deployment (which needs Monica's own GitHub/Streamlit Cloud login, so it can't be done by Claude), checked three real things rather than assuming they'd just work:
1. **Fixed a real blocker in `requirements.txt`:** it had `torch==2.13.0+cpu`, a local build tag specific to PyTorch's own CPU-wheel index - a plain `pip install -r requirements.txt` on Streamlit Cloud's standard-PyPI environment would not have found this exact version and deployment would have failed outright. Added `--extra-index-url https://download.pytorch.org/whl/cpu` as the first line of `requirements.txt` (a real, standard pip mechanism, not invented) and confirmed via a direct query against that index that a matching Linux `manylinux` wheel for Python 3.12 actually exists for this exact torch version - not just assumed the fix works.
2. **Confirmed, not assumed, three deployment facts via current docs:** Streamlit Cloud defaults to Python 3.12 (matches this project exactly) but has known-unreliable `runtime.txt` handling in recent reports, so Python version should be set explicitly via the "Advanced settings" dropdown when deploying, not left to `runtime.txt`. Root-level secrets entered in Streamlit Cloud's dashboard are automatically exposed as `os.environ` variables, so `explanation.py`'s existing `os.environ.get("GEMINI_API_KEY")` will work unmodified - no code change needed, just paste the key into the dashboard as a root-level `GEMINI_API_KEY = "..."` entry, not nested under a section. The GitHub repo was confirmed public via a direct API check (free-tier Streamlit Cloud requires a public repo).
3. **Real memory measurement, not a guess: peak usage hit 725.6 MB out of the free tier's confirmed 1GB limit** during a single, minimal upload (one match, fewest possible live Gemini calls) - measured by watching the actual Windows process's working-set memory during a real Playwright-driven upload, sampled every 2 seconds. **This is a genuine, serious risk, not a checkbox to note and move past** - Streamlit Cloud is documented to forcibly restart apps that exceed their memory limit, and 725MB leaves uncomfortably little headroom (~27%) for a heavier upload, multiple visitors, or any baseline difference between this Windows machine and Streamlit Cloud's actual Linux runtime. Deployment should still be attempted (it may well fit - this measurement is one data point, not a guarantee of failure), but if it fails or gets killed for exceeding resources, the fix is not more debugging - it's reducing what's loaded in memory at once (candidates already known: a smaller CLIP variant, making background removal optional/toggleable, as flagged in Section 7's known risks since early in the project).

**Real deployment bug found and fixed, 2026-09-05: `opencv-python` doesn't run on Streamlit Cloud at all.** Monica actually deployed (Shop page worked, confirming the earlier prep was sound), but the Try It On page crashed immediately with `ImportError` on `import cv2` in `color_detector.py`. This is a well-known, common issue: the standard `opencv-python` PyPI package needs system-level graphics libraries (`libGL.so.1`) that don't exist on Streamlit Cloud's minimal Linux servers - it's built assuming a desktop GUI environment is available, which this project never uses (no `cv2.imshow()` or similar anywhere in the code, only `imread`/`resize`/`cvtColor`/`kmeans`). Fixed by swapping to `opencv-python-headless==5.0.0.93` in `requirements.txt` - the identical library minus the GUI dependency. Verified the exact same version exists for headless via a direct PyPI query before committing to it, then verified locally that color detection produces byte-identical results after swapping the local venv to headless too (RGB `(209,22,26)` → "red", matching the pre-swap diagnostic exactly) - so local testing doesn't silently diverge from what's actually deployed going forward.

### Next action

Monica to push this fix live (Streamlit Cloud auto-redeploys on every push to `main`, so no redeployment steps needed beyond the git push already done) and retest the Try It On page. If another import error surfaces, check first whether it's the same class of problem (a package needing a system library Streamlit Cloud doesn't have) before assuming something else broke.

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
| 4 | Build catalog.json + embeddings | done — 34 items: original 15 placeholder (reused test_images/ photos) + 19 real photos added 2026-09-05, see Section 2 |
| 5 | Matching engine (classify to filter to retrieve to re-rank) | done — the intellectual core, tested end-to-end on two real cases |
| 6 | Test pipeline end-to-end, text only | done — 18/18 uploads ran without error |
| 7 | RAG explanation layer + fallback | done — Gemini (`gemini-3.5-flash-lite`) + template fallback; real key confirmed working 2026-09-04, all 11 test matches returned `source == "llm"` |
| 8 | Streamlit app — Shop, Try It On, category filter | done — built, verified with a real browser test, and the explanation-generation performance issue fixed (top-3-live-explanations cap, see Section 2/11) |
| 8B | Deploy free on Streamlit Community Cloud | prep done (requirements.txt fixed, secrets/Python version confirmed) - actual deployment needs Monica's own login, real memory risk found (725MB/1GB), see Section 2/11 |
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

### 2026-09-04 — Step 8: the Streamlit app

**What was built:** `app.py` (entrypoint), `app_pages/shop.py`, `app_pages/try_it_on.py`, `shop_utils.py`. Tabbed/multi-page navigation between two pages using Streamlit's `st.Page`/`st.navigation` mechanism - confirmed via a live web search this is the *current* API (not the older `pages/` folder convention), then confirmed the installed Streamlit version (1.63.0) actually has both via direct `hasattr()` inspection, before writing a line of app code. The page scripts deliberately live in a folder named `app_pages/`, not `pages/` - naming it `pages/` would trigger Streamlit's older auto-discovery mechanism alongside the explicit `st.navigation` call, which the docs warn against mixing.

- **Shop page:** a plain grid over `catalog.json` - image, name, price, a `st.link_button` that opens a `upi://pay?...` deep link. No cart, no accounts, no checkout logic, exactly as scoped.
- **Try It On Your Clothes page:** file upload → `find_matches()` (Steps 2-5) → `get_explanation()` per match (Step 7) → ranked results with a category-filter multiselect. The pipeline result is cached in `st.session_state` keyed by the uploaded file's identity, so toggling the category filter (which triggers a Streamlit rerun) doesn't re-run CLIP/rembg or re-fire live Gemini calls - only a genuinely new upload does.

**Why this, not an alternative:** hand-rolling page-switching with session-state/if-else blocks was explicitly ruled out by the master plan in favour of Streamlit's own native mechanism - fewer moving parts, and it's what a Streamlit-familiar reviewer would expect to see.

**Verification process, told honestly because it wasn't a straight line:** no project skill or `chromium-cli` was available on this Windows machine, so Playwright was installed as a one-time dev/verification tool (not added to `requirements.txt` - it's not something the deployed app itself needs) to actually drive the app in a real headless browser rather than just reading the code and assuming it works. Several real false starts along the way, each one run down to an actual cause rather than guessed past:
- `wait_until="networkidle"` never resolves against a Streamlit app, because Streamlit holds a permanent WebSocket connection open for live reruns - the network is never idle. Switched to `domcontentloaded` plus explicit `wait_for_selector` calls.
- A page navigated to for the first time in a fresh server process takes ~15s to actually paint any content (confirmed by polling, not assumed) - not a hang, just a real cold-start cost on first visit to that page script in a given server session.
- A screenshot check reported "0 Buy buttons" twice, including on a single-match upload with a 5-second wait margin. Traced all the way through rather than accepted at face value: `st.link_button` renders as an `<a>` tag styled as a button, not a literal `<button>` element, so a `button:has-text('Buy')` CSS selector was always going to find zero - a bug in the test script, not the app. Confirmed the real button exists and works by reading the rendered body text directly, which showed the correct price, a correct LLM-generated explanation, and the word "Buy" all present and correctly ordered.

**Confirmed actually working, both pages:** Shop rendered all 15 catalog items with images, names, prices, and working Buy buttons. Try It On correctly processed a real upload (`test_03_jeans.jpg`) end-to-end - correct detected category/colour/style, 9 correctly-filtered matches, real Gemini-generated explanation text, and a working category filter.

**The one real problem found, not yet fixed:** timing instrumentation was added to every stage of `find_matches()` and around each `get_explanation()` call specifically to answer "is this slow or actually stuck" with real numbers instead of a guess. Result, for the 9-match jeans upload: `find_matches()` itself (background removal, two CLIP calls for category, one for style, one for the embedding, FAISS retrieval, re-rank) took a very reasonable **19.56s total**. Generating explanations for those same 9 matches took **222.71s** - over 90% of the total 243.9s. Individual per-match explanation times ranged wildly from 2.87s to 64.47s for what should be a single quick API call each, which is the signature of the `google-genai` SDK's built-in retry/backoff (`tenacity`) kicking in - almost certainly Gemini's free-tier rate limit (~15 requests/minute, per the research done for Step 7) being exceeded by firing off up to 9-14 sequential live calls for one upload. **This is a genuine, actionable finding, not just "the first upload is slower than isolated testing"** - a several-minute wait per upload is a real problem for a live demo or interview setting, and needs an actual decision (cap how many matches get an LLM explanation vs. template, reduce the default `top_k`, add concurrency, or something else) before Step 8B deployment. Logged as the next action rather than fixed unilaterally, since it changes user-facing behaviour beyond what was asked for in this pass.

**Fix applied, 2026-09-04 (same day, later): cap live explanations to the top 3 matches.** Decision made explicitly, not defaulted into: `app_pages/try_it_on.py` now only calls `get_explanation()` (the live-Gemini-then-fallback path) for the top 3 matches by `final_score` - since `rerank()` already returns matches sorted best-first, that's just the first 3 in the list. Every match beyond that calls `build_template_explanation()` directly, skipping the API attempt entirely rather than firing a call that would likely just be rate-limited anyway. **This is a real, honest constraint being worked around - Gemini's free tier genuinely rate-limits at this call volume - not a workaround being hidden from anyone reading this.** The choice of "top 3" specifically: those are the matches most likely to actually matter to the customer (highest-scored), and 3 live calls per upload is a load Gemini's free tier can plausibly sustain for a low-traffic demo, where 9-14 calls per upload could not.

Verified with real log evidence, not just "the code looks right": rerunning the same 9-match jeans upload showed exactly 3 explanations came back `(llm)` and the other 6 came back `(template)` at **0.00s each** - proof the fix skips the API call entirely for those, not just skips waiting for a slow one. **Honest complication found while verifying:** total time for this particular test run was still ~231s, barely better than before the fix (244s) - because the 3 live calls themselves took 99.10s, 56.13s, and 64.67s each. This session had already fired a large number of Gemini calls across all of today's testing (Step 7's verification, the original 9-call diagnostic run, this run), plausibly exhausting the free-tier rate-limit window well before this specific test, so even 3 calls still hit the SDK's retry/backoff hard. **The fix is correctly implemented and does what it was asked to do (bound API load to 3 calls, proven by the 0.00s template times) - it does not and cannot fix Gemini's own throttling recovery time**, which is an external constraint outside this code's control. A real customer's single upload, not preceded by a testing session's worth of calls in the same rate-limit window, should behave meaningfully better than this specific verification run did.

**Known limitations, stated plainly:**
- Even with the fix, wall-clock time for the top 3 live explanations depends on Gemini's free-tier rate-limit state at that moment - it can still be slow if the API has recently been called heavily (as this session's own testing demonstrated). This is a genuine property of relying on a free-tier API, not something further code changes alone can fully eliminate.
- The UPI merchant VPA in `shop_utils.py` (`MERCHANT_UPI_ID = "yourupi@upi"`) is a placeholder - needs Monica's real UPI ID before the Buy button does anything real.
- Playwright (installed for this verification pass only) downloaded a real Chromium binary (~115MB) into the local machine's Playwright cache - harmless, but worth knowing it's there if disk space is ever a concern; it is not part of the project's own dependencies.

### 2026-09-05 — Catalog expanded from 15 placeholder items to 34, with 19 real photos

**What was built:** Monica added 19 real product photos directly into `catalog_images/` (outside of any Claude session). `catalog.json` was extended with an entry for each (`cat_16`-`cat_34`), and `build_catalog_embeddings.py` was rerun across the full, now-34-item catalog so every item - old placeholders and new real photos alike - has a computed CLIP embedding, dominant colour, and CLIP style, cached the same way as before.

**Which tool/library was used:** no new tool - this reused every piece already built in Steps 2-4 (`rembg` for background removal, CLIP for embeddings/style, the colour detector). The only actual code change was to `pairing_rules.py`, refactored so `PAIRING_RULES` is built from two sets (`BOTTOMS = {jeans, skirt, pants, shorts}`, `TOPS = {top, shirt, kurta, hoodie}`) instead of typing out every category's allowed pairs by hand - two brand-new categories (**pants**, **shorts**) needed adding, and doing that by hand across 8+ lines is exactly the kind of place a typo creates a silent bug later.

**Why this, not an alternative:** "pants" and "shorts" could have been folded into the existing "jeans"/"skirt" categories instead of getting their own labels, but that would mean CLIP is asked to call a pair of trousers "jeans" even when they clearly aren't denim - less accurate for no real benefit, since this rule set treats every bottom identically anyway. Giving them their own category costs nothing (same pairing behaviour) and keeps the catalog's own labels honest.

**Verified, not assumed:**
- Confirmed the `PAIRING_RULES` refactor produces byte-for-byte the same pairing sets for all 8 original categories, by printing them and comparing against the hand-written version from Step 5 - a refactor with no test is just a hope.
- Confirmed Pillow can actually open the two `.avif` photos (`Green_top.avif`, `blue_top.avif`) before assuming the pipeline would work on them - it can, and since `remove_background.py` always converts to PNG before anything else touches the image, the AVIF format never reaches OpenCV or the embedding code anyway.
- Ran all 19 new photos through the full `find_matches()` pipeline as if they were customer uploads: 19/19 completed without error. A new "shorts" upload (`white_shorts.jpg`) correctly classified as `shorts` and only retrieved top-family candidates (hoodie, top, shirt, blazer, kurta) - confirming the new categories are wired into filtering correctly, not just accepted by `CATEGORY_LABELS` and then silently mishandled downstream.

**Real decision made without Monica's input, flagged plainly:** the 19 new items' display names and prices were inferred from their filenames (e.g. `denimlightwashblue_skirt.jpg` → "Light Wash Denim Skirt", ₹1599) since none were provided - placeholders in the same spirit as the original 15, except these are genuinely real product photos, so the placeholders matter more here and are worth a real review pass before anything customer-facing happens.

**Known limitations, stated plainly:**
- A handful of the new photos get a different category from CLIP than what's stored in `catalog.json` when tested as a hypothetical upload (e.g. `green_skirt.jpg` reads as "dress" to CLIP, `denimlightwashblue_skirt.jpg` reads as "shorts") - this is the same already-documented Step 3 accuracy limitation resurfacing on new photos, not a new bug, and doesn't affect anything: the catalog's *stored* category (what I assigned from the filename) is what's actually used when an item is a candidate match, completely separate from what CLIP would guess if that same photo were hypothetically uploaded by a customer.
- Names/prices for the 19 new items are placeholders (see above) and colour/style are machine-detected with the same known imprecision already documented for Steps 2 and 3 (e.g. `Green_top.avif` detected as colour "black").

### 2026-09-05 — Real accuracy fixes for CLIP category classification and colour detection, not just documented as limitations

Monica specifically pushed back on treating the type-classification and colour-detection weaknesses as permanent limitations, after noticing a real-world case (uploading a photo of pants got classified as "skirt"). Both were measured and genuinely improved, not just written up.

**CLIP category classification, 67.6% → 79.4% (23/34 → 27/34 correct):** used all 34 catalog items as ground truth (their `category` field is known-correct, since it's how they were catalogued), tried several reworded prompts, and kept only what measurably helped - exactly the same method Step 3B used for the hoodie fix.
- The single biggest confusion cluster was "top" read as "shirt" (6 of 11 errors). Rewording `CATEGORY_LABELS["top"]` to `"a photo of a casual t-shirt or tank top with no collar or buttons"` and `["shirt"]` to `"a photo of a collared button-up shirt"` - describing the actual visual distinction instead of the generic "a photo of a top"/"a photo of a shirt" - fixed nearly all of it on its own.
- Tried rewording skirt, dress, and shorts too (the other confusion cluster: skirt read as shorts/dress). Tested each change in isolation rather than bundling them: every single one made overall accuracy *worse* (73.5%, 76.5%, 76.5% vs the top/shirt-only fix's 79.4%) - the same "more emphatic/descriptive wording backfires with CLIP" lesson Step 3B already found for style labels. Left skirt/dress/shorts/pants/jeans prompts untouched as a result - a change that isn't measured to help shouldn't go in just because it sounds like it should help.
- **Honest limitation:** the specific pants-read-as-skirt case Monica saw wasn't reproduced in this ground truth (all 4 catalog pants photos classified correctly even before this fix), so this fix targets the biggest *measured* confusion cluster, not necessarily that exact photo - a different angle/lighting on pants can still occasionally confuse skirt/pants, and that residual risk hasn't been eliminated, just reduced. Remaining errors after the fix (7 of 34) are a flat-lay vest already known to be difficult (see the abandoned segmentation history in Section 2), some low-confidence top/dress toss-ups, and skirt/dress confusion that resisted every wording tried.

**Colour detection: two distinct real bugs found and fixed, not one.** Diagnosed by printing the actual raw RGB values behind every catalog item's detected colour, not just looking at the final wrong-looking name:
1. **A missing colour.** There was no "orange" in `COLOR_NAMES` at all, so a genuinely orange garment (raw RGB `(219,119,102)`) was forced into whichever of the 12 existing colours happened to be numerically closest - "pink". Added `"orange": (230, 126, 34)` to the reference table. Fixed.
2. **A more fundamental bug: neutral colours were being matched to a random hue.** A perfectly neutral gray, raw RGB `(81,81,81)` (R, G, and B all equal), was being called **"olive"** - not because it looks olive, but because `"gray"` only had one reference point (150,150,150, a fairly *light* gray), so a darker true-gray ended up numerically closer to an unrelated hue purely by where that one point happened to sit in 3D colour space. The real fix: check *whether a colour has any real hue at all* first, using the gap between its brightest and darkest RGB channel (a standard, well-founded technique - genuinely neutral colours have R≈G≈B regardless of brightness). Below a gap of 15, `closest_color_name()` now skips hue-matching entirely and classifies by brightness alone (black/gray/white); only colours with real hue content go through the old euclidean-distance matching. Fixed the gray-called-olive bug without needing more reference points.
- **Verified, not assumed:** re-ran the full 34-item diagnostic after the fix. Both target bugs fixed (orange correctly detected as orange, the neutral gray correctly detected as gray). Checked for regressions: one borderline case surfaced (`navy_top.jpg`, RGB `(34,34,49)`, a color with a real if subtle blue tint) got swallowed into "black" at an initial threshold of 20 - tightened the threshold to 15 specifically to preserve this case as "navy" while keeping both real fixes intact, rather than accepting a new regression to fix an old one.
- **Honest limitation:** this is still fundamentally nearest-reference-point matching over a small, hand-picked palette (13 colours now) - genuinely ambiguous or textured fabrics (e.g. `pinkgliter_skirt.jpg`'s glitter finish scatters light unevenly, averaging toward a color that doesn't obviously match its product name) will still sometimes get a defensible-but-debatable answer. This fix targets the two *diagnosed, concrete* bug classes found, not a claim that colour detection is now solved.

### 2026-09-05 — Colour detection rewritten a second time, after real deployment use showed it still wasn't good enough

Monica actually deployed and used the app on real photos, and reported most colours were coming back wrong - "Light Green Blazer" and "Beige Blazer" both showed as "gray", along with several tops. The fix above (the extra 15-unit "channel spread" neutral check) was real and correctly targeted, but too narrow: it fixed one specific bug (pure neutrals reading as a random hue) without addressing the much bigger, systemic issue underneath it.

**Root cause, found by actually computing HSV values for all 34 catalog items, not guessing:** muted/dusty real-world colours (a desaturated light green, a dusty pink) sit numerically *closer* to plain gray `(150,150,150)` than to any vivid, fully-saturated reference swatch, because gray happens to sit near the geometric centre of RGB space. Gray was winning the nearest-neighbour match constantly - not because things actually looked gray, but because gray is "centrally located" and therefore close to almost everything even slightly muted.

**Rewrite:** switched to HSV (hue/saturation/value) as the primary signal, not just a patch on top of RGB distance.
- Colours are split into `NEUTRAL_REFERENCE` (`black`, `white`, `gray` - reachable *only* by checking actual saturation and brightness, never by raw distance) and `CHROMATIC_COLORS` (everything else, matched by distance only among themselves). This guarantees a colour with real hue content can never fall back to gray just because gray is numerically nearby - the exact bug that was happening.
- `"beige"` is treated as a saturation/brightness special case too (a warm, low-saturation, mid-brightness colour), *not* as a full distance-matching candidate - tested including it as a full candidate and it started "stealing" genuinely pink/purple pixels that also happen to be pale.
- `"cream"`, despite looking similar to beige, needed the *opposite* treatment - excluding it from full distance-matching caused a worse regression (several genuinely cream-coloured catalog photos, correctly cream before, got forced into "pink" - the next-closest chromatic option once cream wasn't available at all). This asymmetry between two very similar-seeming colours was found by testing, not designed upfront, and is deliberately kept even though it looks inconsistent at first glance.
- Very dark pixels (`value < 0.15`) are called black outright before anything else runs, since tiny RGB differences at near-zero brightness produce misleadingly large *relative* saturation readings that would otherwise wrongly escape the neutral check.

**Verified against real, if informal, ground truth - and a real regression was caught and fixed before shipping, not after:** 16 of the 34 catalog photos have the actual colour stated in their filename (e.g. `beige_blazer.jpg`, `navy_top.jpg`) - real signal, even if informal. Iterated through several designs, checking each one against all 16, not just the ones that happened to look right. **One iteration achieved the best score on those 16 (10/16) while silently breaking three previously-correct catalog items** (cream-coloured photos started reading as "pink") - a regression the 16-item check alone couldn't see, since none of the affected items happened to have color-bearing filenames. Caught by separately checking the *full* 34-item list against each item's previously-stored value, not just trusting the headline number. Final version: **9/16 (56%) on the filename check, with zero regressions on the previously-correct items** - a large, real improvement over the prior version's near-total failure on muted/pastel real-world colours (Light Green Blazer, Beige Blazer, etc. all correctly distinguished from gray now).

**Honest limitation, found and kept in, not hidden:** pink's own reference point is itself pale enough that it has started attracting *some* pale, low-saturation pixels of a genuinely different hue - e.g. a pale blue-gray "light wash denim" skirt now reads as "pink" rather than blue. This is a smaller-scale recurrence of the exact same "gray was too centrally located" problem, just with a different, paler colour now playing a similar role. **This is a real, acknowledged ceiling on small-palette nearest-neighbour colour naming, not something more threshold-tuning can fully solve** - each fix in this area has traded one specific error for a different one, and diminishing returns set in clearly during tuning (adding a candidate fixed some cases and broke others, repeatedly). A genuine further improvement would need a fundamentally different technique - a proper trained colour-naming model, or real human-verified ground truth to tune against instead of informal filename hints - not another round of manual threshold adjustment. Documented here plainly so this doesn't read as solved when it's actually "substantially better, with a known and explained remaining edge."

Both fixes applied directly to `pairing_rules.py` (`CATEGORY_LABELS`) and `color_detector.py` (`COLOR_NAMES`, `closest_color_name()`), then `catalog.json` was regenerated via `build_catalog_embeddings.py` so every catalog item's stored colour reflects the fix. Full pipeline re-verified end-to-end afterward: all 34 catalog items ran through `find_matches()` without error.

### 2026-09-05 — Colour detection, third round: multi-anchor colour families, and an honest conclusion about this technique's ceiling

Monica tried the app again after the second rewrite and reported it was still substantially wrong - "Light Green Blazer" now showed "purple" (worse than the earlier "gray"), "Tailored Blazer" showed "pink". The diagnosis was right the second time (muted colours losing to a numerically-central swatch) but the fix was incomplete: it fixed *gray* being that central swatch, without noticing that **any** single-anchor-per-family design has the same problem waiting to happen with a different colour playing gray's old role.

**Real fix attempted: give each colour family more than one reference point** - a vivid anchor and a separate muted/dusty anchor - so a muted version of a colour has something nearby to match against, instead of only a distant, fully-saturated swatch. Added a muted "sage" anchor to the green family and a muted "dusty blue" anchor to the blue family. This is a real, working idea: it correctly fixed both concrete cases in front of it (`lightgreen_blazer.jpg` now reads "green", `denimlightwashblue_skirt.jpg` now reads "blue"), verified against the same 16-item filename check plus the full 34-item regression check, landing at **10/16 (62.5%)**, the best result across every version tried, with only one new, minor, debatable miss traded off (one blue-gray photo shifted from "teal" to "green" - still wrong, but a smaller error).

**What was tried and explicitly rejected, because testing - not intuition - said no:**
- Adding a distinct "tan" family to catch dusty warm tones. Result: "tan" immediately became a new attractor, stealing `orangeTop.jpg`, `pinkgliter_skirt.jpg`, `white_shorts.jpg`, and even a previously-correct kurta into "tan" - net same score (9/16) with a different, not smaller, set of errors.
- Loosening the "beige" achromatic threshold from 0.08 to 0.40 saturation for warm hues, to catch tan/khaki tones without a new distance-matched family. Result: "beige" became the new attractor instead, pulling in three previously-correct **cream** items and a previously-correct **pink** item - a clear regression, reverted immediately rather than kept and rationalized.

**The honest conclusion, stated plainly rather than papered over with more tuning:** across three full rounds of fixes today, *every single configuration tried has moved the specific errors around rather than reducing the total number much past ~55-62%* - gray was the attractor, then pink, then briefly tan, then briefly beige. This is not a sequence of small oversights each one fix away from done; it is direct, repeated, measured evidence of a real ceiling on this technique (a single dominant k-means colour, matched by distance to a small, hand-picked palette of a dozen-ish reference points). Continuing to add or adjust reference points by hand is very unlikely to move meaningfully past this ceiling, based on what today's testing actually showed - it would keep trading errors, not eliminating them. **A genuine further improvement needs a different technique entirely**, and the honest options are: (1) accept the current, substantially-improved-but-imperfect state as good enough for a resume/demo project, clearly documented as a known limitation (consistent with how this project has handled every other imperfect-but-honest component); or (2) a real follow-up task, not a quick fix - swap the tiny hand-picked palette for a large reference set built from real human-labeled colour data (hundreds of named colours from a public colour-naming survey, not a dozen guessed swatches), which is a genuinely different scope of work, not another round of threshold tuning.

Verified end-to-end after this final round: 34/34 catalog items ran through `find_matches()` without error, `catalog.json` regenerated with the updated colours.

### 2026-09-05 — Three more colour-detection experiments tried and rejected, after Monica asked for 14/16 (not 10/16)

Monica pushed back further, asking for at least 14/16 rather than accepting 10/16. Took this seriously and tried three genuinely different techniques - not more threshold tweaking of the same idea - specifically to see if 14/16 was actually reachable. **All three performed equal to or worse than the already-committed 10/16 solution**, tested and rejected before anything was shipped, not assumed to help:

1. **A much larger, real reference set** (matplotlib's bundled xkcd colour-name survey, ~950 human-labelled colours, installed temporarily just to extract the data then removed - never added as a project dependency). Keyword-matched to this project's colour vocabulary, giving some families over 100 real anchor points instead of 1-2 hand-picked ones. **Result: 7/16, worse than the hand-tuned version.** Root cause, diagnosed rather than shrugged off: survey respondents naming colours tend to pick vivid, prototypical examples ("grass green", "sky blue"), so this dataset is actually *sparse* in exactly the muted/dusty region of colour space that real clothing photos (after background removal, under normal lighting) fall into - more real data, but not real data of the right kind.
2. **The same idea, capped to equal-sized families** (20 anchors max per family), to rule out the obvious confound that families with 170 matched colours (green) statistically out-compete families with 3 (navy) regardless of true visual similarity. **Result: 5/16, worse still.** Confirms the imbalance was real, but fixing it didn't reveal a better dataset underneath - the fundamental vivid-vs-muted mismatch remained.
3. **Perceptually-uniform LAB colour distance** (converting RGB to the CIE LAB colour space via the standard sRGB→XYZ→LAB formulas, implemented directly with no new dependency) instead of raw RGB Euclidean distance, using the same curated anchors as the committed 10/16 version - a genuinely different, well-founded idea (LAB distance is designed to match human perception better than raw RGB does). **Result: 9/16, marginally worse than plain RGB distance with the same anchors.**

**Conclusion, backed by four separate serious attempts now (the original hand-tuning plus these three), not just repeated assertion:** roughly 55-65% is a real, demonstrated ceiling for "name the single dominant k-means colour by nearest-neighbour match," regardless of which reasonable variation of the technique is used - bigger dataset, balanced dataset, and better-founded distance metric were all tried and none helped. **The currently-committed 10/16 hand-tuned version remains the best result found and was not changed** as a result of this experimentation - nothing here regresses what's shipped. Going further than this ceiling honestly requires a different scope of work entirely (a trained model, or real labelled ground truth specific to clothing photos rather than a general colour-naming survey), not another attempt at the same underlying technique. No code was changed by this experiment; documented here so the effort and its result are on the record, not lost.

---

## 12. Technology used per step — quick reference

| Step | What it does | Technology / model used | Why this, not an alternative |
|---|---|---|---|
| 1 / 1B | Workspace, environment, version control | Python 3.12.7, `venv`, Git, GitHub | Standard, free, reproducible setup — nothing project-specific here |
| 2 | Find a garment's dominant colour | OpenCV, k-means clustering (k=3, single largest group) + HSV-based neutral/chromatic split for naming (rewritten 2026-09-05) | Simplicity chosen deliberately under time pressure — always returns a usable answer rather than sometimes correctly flagging multi-color but giving nothing actionable. Rewritten twice 2026-09-05 after real deployment use showed muted/pastel colours defaulting to "gray" (a systemic bug, not the one first fixed) - now 9/16 on filename ground truth, up from near-zero, with an honestly documented remaining ceiling on this technique - see Section 11 |
| 2B | Remove photo background before analysis | `rembg`, `u2netp` model | Lightweight (~5MB) pretrained segmentation model; the default heavier rembg model caused a real out-of-memory error on this machine |
| 3 | Identify garment type and style from a photo | CLIP (`openai/clip-vit-base-patch32`) via HuggingFace `transformers`, PyTorch (CPU build) | Zero-shot — no training data needed; CPU build used since the machine has no NVIDIA GPU. Category-prompt wording measured and tuned 2026-09-05 against all 34 catalog items (67.6% → 79.4%) - see Section 11 |
| 3B | Measure how accurate Step 3 actually is | Manual scoring against 15 real test photos | No shortcut for this — accuracy has to be checked against real, human-verified answers |
| 4 | Build the searchable catalog | `catalog.json` (hand-built data) + CLIP embeddings, cached to a file | Embeddings only need computing once per catalog photo, not on every customer visit |
| 5 | Find and rank complementary items | FAISS (`faiss-cpu`, one `IndexFlatIP` per category) + hand-written pairing/colour/silhouette rule tables (`pairing_rules.py`) | FAISS is free/local; one index per category avoids needing to filter a shared index after the fact, and scales unchanged if the catalog grows later (a brute-force comparison would be equally fast at today's 15-item size — FAISS is chosen for where this is headed, not a speed win yet). Rules are used for pairing/colour/silhouette because no free dataset of "good outfits" exists, and rules stay explainable; colour is weighted above style in re-ranking since Step 3B found style classification measurably weaker |
| 6 | Confirm Steps 2–5 work together | Plain Python, text output only | Debugging is easier before any visual layer is added |
| 7 | Generate a plain-language reason for each match | Google Gemini (`gemini-3.5-flash-lite`) via `google-genai` + a hand-written template fallback | Chosen after checking current free-tier options rather than from memory; the fallback avoids a live demo failing if the API is down or rate-limited. Confirmed working with a real key 2026-09-04 (after fixing one real model-name drift: `2.5-flash-lite` was retired mid-project in favor of `3.5-flash-lite`) |
| 8 | User-facing app: Shop page + Try It On page | Streamlit (`st.Page`/`st.navigation`), verified with Playwright (dev-only, not a project dependency) | Keeps the whole interface in Python; deliberately simpler than a hand-built React/Flask site so effort stays on the AI pipeline. Real finding: sequential Gemini explanation calls per upload are slow (~90% of total time on a 9-match upload), likely rate-limiting - unresolved |
| 8B | Put the app online with a public link | Streamlit Community Cloud (free tier, confirmed 1GB RAM limit) | Free hosting. Real measured peak usage: 725.6MB during a single minimal upload - a genuine risk, not just a checkbox; mitigation if deployment fails is a smaller CLIP variant or optional background removal, not more debugging |
| 9 (stretch) | Show the outfit on a 3D figure | Three.js + a free `.glTF` mannequin, flat texture overlay | Real cloth simulation is specialist paid software; this gets visual impact without that cost |
| 10 (optional) | Let a few real people buy an item | A UPI payment link, manual order tracking | A full payment gateway is unnecessary overhead for 2–10 manual orders |
| 11 | Document the project | Markdown README, architecture diagram | Most recruiters spend under a minute on a repo — this is the highest-value hour in the project |



