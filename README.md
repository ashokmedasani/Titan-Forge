# TitanForge — Aerospace Component Failure Prediction

**Live repo:** https://github.com/ashokmedasani/Titan-Forge

A story-driven Streamlit walkthrough of a real correction to a predictive-maintenance
model: a class-imbalance mistake I made a year ago, caught and fixed two different
ways, with both fixes verified against the original.

## The story, in short

TitanForge Industries forges aerospace components. 10,000 production runs were logged
with five sensor readings (air temperature, process temperature, rotational speed,
torque, tool wear) and a failure outcome. Only 3.48% of runs actually failed.

**A year ago (March 2025 exam):** trained a Random Forest and a Gradient Boosting
model, reported weighted-average accuracy/precision/recall/F1 — both looked strong
(97–98% accuracy) — and recommended Gradient Boosting for deployment. Never checked
how the target variable itself was distributed.

**Revisited now:** that omission was the real story. Weighted metrics were dominated
by the 96.5% majority class and hid that Random Forest was only catching ~20% of real
failures. Two fixes were tested and verified against each other: class-weighted
training (`class_weight='balanced'`, reweights the loss function, data stays
imbalanced) and **SMOTE** (actual data-level oversampling to a 50/50 training set).
Both lift recall substantially; neither beats the tuned Gradient Boosting model on F1,
which stays the overall recommendation.

## Files in this repo

| File | What it is |
|---|---|
| `app.py` | The Streamlit app — this is what deploys |
| `Titan_Forge.ipynb` | The original March 2025 analysis, unedited |
| `Titan_Forge_Corrected.ipynb` | The corrected analysis — relabeling, stratified split, class weighting, SMOTE, full metric comparison |
| `TitanForge_dataset.xlsx` | The dataset (10,000 rows, 3.48% failure rate) |
| `requirements.txt` | Python dependencies for the app |

## Push this to GitHub

From this folder:
```bash
git init
git add .
git commit -m "TitanForge: story walkthrough + SMOTE balancing comparison"
git branch -M main
git remote add origin https://github.com/ashokmedasani/Titan-Forge.git
git push -u origin main
```
If the repo already has commits, pull first with `git pull origin main --allow-unrelated-histories`,
resolve any conflicts, then push. The web UI's "uploading an existing file" also works —
drag in all five files listed above.

## Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app** → pick the `Titan-Forge` repo → branch `main` → main file `app.py`.
3. Click **Deploy**. First build takes 2–3 minutes.
4. You'll get a public URL like `https://titan-forge.streamlit.app`.

## Running it locally first

```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`. Check every stage before pushing/deploying.

## For the LinkedIn post

Once live, you'll have two links to share:
- **Live walkthrough:** your `*.streamlit.app` URL — the one people click first.
- **Full notebooks / code:** your GitHub repo URL — for anyone who wants every cell.

See `LinkedIn_Post.md` in this same folder for a ready draft and posting notes.
