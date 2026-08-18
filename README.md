# TitanForge — Aerospace Component Failure Prediction (Corrected)

A guided, step-by-step Streamlit walkthrough of a real correction: catching and
fixing a class-imbalance accuracy trap on a predictive-maintenance dataset.

- **Original analysis:** Advanced Analytics final exam, March 2025 — reported 97.2% accuracy, missed that failures are only 3.48% of the data, and never checked recall.
- **Corrected analysis:** August 2026 — stratified split, class-weighted training, evaluation on precision/recall/F1/PR-AUC instead of accuracy, and a cost-justified decision threshold.

Live app walks a viewer through: the research question → the one basic thing
skipped the first time (checking the target variable's balance) → what changed
→ results, before vs. after → the confusion matrix translated into plain
language → feature importance → the takeaway. The full notebook with all 8
corrections is linked from inside the app.

---

## Files in this repo

| File | What it is |
|---|---|
| `app.py` | The Streamlit app — this is what deploys |
| `Titan_Forge_Corrected.ipynb` | The full analysis notebook, all 8 corrections, cell by cell |
| `TitanForge_dataset.xlsx` | The dataset (10,000 rows, 3.48% failure rate) |
| `requirements.txt` | Python dependencies for the app |

---

## Step 1 — Push this to GitHub

1. Go to [github.com/new](https://github.com/new) and create a **public** repo — suggested name: `titanforge-failure-prediction`.
2. Do **not** initialize it with a README (you already have one here).
3. From this folder, run:
   ```bash
   git init
   git add .
   git commit -m "TitanForge: corrected class-imbalance analysis + Streamlit walkthrough"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/titanforge-failure-prediction.git
   git push -u origin main
   ```
   (Replace `YOUR-USERNAME` with your actual GitHub username. If you don't have
   Git installed or set up, GitHub's "uploading an existing file" web UI works
   too — drag all four files into the new repo.)

## Step 2 — Update the link inside `app.py`

Open `app.py` and change this line near the top to your real repo URL:
```python
GITHUB_URL = "https://github.com/YOUR-USERNAME/titanforge-failure-prediction"
```
Commit and push that one-line change before deploying.

## Step 3 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**, pick the `titanforge-failure-prediction` repo, branch `main`, main file `app.py`.
3. Click **Deploy**. First build takes 2-3 minutes (installing scikit-learn etc.).
4. You'll get a public URL like `https://titanforge-failure-prediction.streamlit.app`.

## Step 4 — What to paste into the LinkedIn post

Once both are live, you'll have two links to include:
- **Live walkthrough:** your `*.streamlit.app` URL — this is the one people click first.
- **Full notebook / code:** your GitHub repo URL — for anyone who wants to see every cell.

---

## Running it locally first (recommended before deploying)

```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`. Confirm the charts and numbers look right
before pushing to GitHub.
