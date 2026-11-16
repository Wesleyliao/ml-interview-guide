# AI / ML Interview Guide

A static study site with ML Design, Coding, ML Theory and ML Coding sections.

Open `docs/index.html` locally (or use the `START_HERE.html` shortcut). No server, npm installation or API keys are needed. Checklist progress stays in the current browser; it does not sync across devices or domains.

## Publish on GitHub Pages

1. Create a new public GitHub repository. Public repositories work with GitHub Free.
2. Upload this clean source tree, including `docs/`. Do not upload a parent ZIP directory, virtual environments or old Git history.
3. Open **Settings → Pages**.
4. Under **Build and deployment**, select **Deploy from a branch**, branch **main**, folder **/docs**, and Save.
5. Wait for the Pages deployment to complete. GitHub displays the URL, usually `https://USERNAME.github.io/REPOSITORY/`.

The published folder contains `index.html`, `.nojekyll`, and the Python files linked by the page. GitHub serves these files; it does not execute Python. No custom workflow is required.

## Edit and rebuild

- `study-hub/`: HTML templates, styling, JavaScript, reference designs and SVG diagrams.
- `curriculum/`: 106 ML theory topics with concise summaries, derivations, examples, tradeoff tables, model deep dives and primary sources.
- `ml_coding/`: nine framework-independent core exercises (NumPy/plain-Python implementations + an end-to-end LightGBM classification baseline), five plain-Python Applied AI exercises, and optional PyTorch labs, with tests.
- `scripts/build_study_hub.py`: shared outline, LeetCode list and site builder.
- `docs/`: generated publishing output; commit this after rebuilding.

```bash
python3 scripts/build_study_hub.py
# Optional frontend validation (requires Deno):
deno run --allow-read scripts/check_study_hub.mjs
# Core NumPy/plain-Python exercises (requires numpy and pytest):
python3 -m pytest ml_coding/classics/ -q
# All exercises, including optional PyTorch labs (requires requirements.txt):
python3 -m pytest ml_coding/ -q
```

Equations use LaTeX rendered by the bundled KaTeX library, with local fonts for offline use. Edit `formula` fields in `curriculum/ml_theory.json` using LaTeX (one equation per line); use `\(...\)` for inline math in prose.

The build itself needs only standard-library Python. Update source files, rebuild, and push the source and `docs/` changes. Pages redeploys automatically on pushes to its configured branch.

The ML Design section includes recommender/ads-marketplace, enterprise RAG, knowledge-work agent, coding-agent and scarce-label bootstrapping walkthroughs with annotated engineering-blog links across industry leaders. The Coding tab retains its 75-problem LeetCode checklist.

Reference implementations are educational. No company affiliation or official question attribution is implied.
