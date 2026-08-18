# TitanForge — Aerospace Component Failure Prediction

**Live App:** https://titan-forge.streamlit.app/

A story-driven Streamlit walkthrough of a real correction to a predictive-maintenance
model: a class-imbalance mistake I made a year ago, caught and fixed two different
ways, with both fixes verified against the original.

## The story, in short

TitanForge Industries forges aerospace components. 10,000 production runs were logged
with five sensor readings (air temperature, process temperature, rotational speed,
torque, tool wear) and a failure outcome. Only 3.48% of runs actually failed.

**A year ago (March 2025):** trained a Random Forest and a Gradient Boosting
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

