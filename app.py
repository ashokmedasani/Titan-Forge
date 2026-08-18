"""
TitanForge — Aerospace Component Failure Prediction
A guided, step-by-step walkthrough of a real correction: catching and fixing
a class-imbalance accuracy trap. Built with Streamlit.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud, pointed at this repo's app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
)

st.set_page_config(
    page_title="TitanForge — Failure Prediction, Corrected",
    page_icon="✈️",
    layout="wide",
)

GITHUB_URL = "https://github.com/YOUR-USERNAME/titanforge-failure-prediction"  # <-- update after upload
NOTEBOOK_URL = f"{GITHUB_URL}/blob/main/Titan_Forge_Corrected.ipynb"

# ---------------------------------------------------------------------------
# Data + model pipeline (mirrors the notebook exactly)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_excel("TitanForge_dataset.xlsx")
    return df


@st.cache_data
def prep(df):
    df = df.copy()
    df["FAILURE"] = 1 - df["Failure_Status"]  # 1 = failure, 0 = OK
    work = df.drop(["UDI", "Product ID"], axis=1)
    work = pd.get_dummies(work, columns=["Type"], prefix="Type")
    X = work.drop(["Failure_Status", "FAILURE"], axis=1)
    y = work["FAILURE"]
    return X, y


@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )
    models = {
        "RF (original)": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1),
        "RF balanced": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=1, class_weight="balanced"
        ),
        "GB (original)": GradientBoostingClassifier(random_state=1),
        "GB tuned": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=1
        ),
    }
    fitted = {}
    rows = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        fitted[name] = m
        pred = m.predict(X_test)
        proba = m.predict_proba(X_test)[:, 1]
        rows.append(
            dict(
                Model=name,
                Accuracy=accuracy_score(y_test, pred),
                Precision=precision_score(y_test, pred, zero_division=0),
                Recall=recall_score(y_test, pred),
                F1=f1_score(y_test, pred),
                PR_AUC=average_precision_score(y_test, proba),
                Train_acc=accuracy_score(y_train, m.predict(X_train)),
            )
        )
    results = pd.DataFrame(rows).set_index("Model")
    baseline_recall = 0.0
    baseline_acc = 1 - y_test.mean()
    return fitted, results, X_test, y_test, baseline_acc, baseline_recall


df_raw = load_data()
X, y = prep(df_raw)
fitted, results, X_test, y_test, baseline_acc, baseline_recall = train_models(X, y)

n_total = len(df_raw)
n_fail = int(y.sum())
fail_rate = n_fail / n_total

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("✈️ TitanForge — Aerospace Component Failure Prediction")
st.caption(
    "A real correction, walked through step by step: original analysis (March 2025) "
    "vs. corrected analysis (August 2026)."
)

st.markdown(
    f"**Full notebook on GitHub:** [{NOTEBOOK_URL}]({NOTEBOOK_URL})  ·  "
    "everything below is the same analysis, running live."
)

st.divider()

# ---------------------------------------------------------------------------
# Step 1 — Research question
# ---------------------------------------------------------------------------

st.header("1 · The research question")
st.markdown(
    "Can we predict which machine components will fail — from sensor readings alone "
    "(temperature, rotational speed, torque, tool wear) — **before** they fail? "
    "And just as important: once we build a model, can we actually trust the number "
    "it reports?"
)

# ---------------------------------------------------------------------------
# Step 2 — The one thing I missed
# ---------------------------------------------------------------------------

st.header("2 · The basic thing I skipped the first time")
c1, c2 = st.columns([1, 1.3])
with c1:
    st.metric("Failures in the dataset", f"{n_fail:,} of {n_total:,}", f"{fail_rate*100:.2f}%")
    st.markdown(
        "Before touching a single model, the first question should always be: "
        "**how is the target variable distributed?** I didn't check that in the "
        "original pass. It's a basic step — and skipping it is a common landing-stage "
        "mistake — but it's the one that mattered most here."
    )
with c2:
    fig = px.pie(
        values=[n_total - n_fail, n_fail],
        names=["No failure", "Failure"],
        color_discrete_sequence=["#4c72b0", "#c44e52"],
        hole=0.55,
        title="Target variable balance",
    )
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

st.warning(
    f"A model that predicts **'no failure' every single time** scores "
    f"**{baseline_acc*100:.2f}% accuracy** — and catches **zero** real failures. "
    "That's the baseline every model here has to beat."
)

# ---------------------------------------------------------------------------
# Step 3 — What changed
# ---------------------------------------------------------------------------

st.header("3 · What I actually changed")
st.markdown(
    "Everything else in the analysis — the features, the preprocessing, the model "
    "families — stayed the same. The correction was entirely about **how the "
    "imbalance was handled**, in three places:"
)
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Stratified split")
    st.markdown(
        "Train and test sets now both keep the same ~3.5% failure rate as the full "
        "dataset, so the test set isn't accidentally easier or harder than reality."
    )
with col2:
    st.subheader("Class-weighted training")
    st.markdown(
        "`class_weight='balanced'` makes the model pay a real penalty for missing "
        "a failure, instead of treating it as a rounding error."
    )
with col3:
    st.subheader("The right scorecard")
    st.markdown(
        "Accuracy is replaced by **precision, recall, F1, and PR-AUC** — the metrics "
        "that don't collapse when one class is rare."
    )

# ---------------------------------------------------------------------------
# Step 4 — Results, before vs after
# ---------------------------------------------------------------------------

st.header("4 · The result — same accuracy, very different model")

display = results.copy()
display.index.name = "Model"
display = display[["Accuracy", "Precision", "Recall", "F1", "PR_AUC"]]
st.dataframe(
    display.style.format("{:.3f}").background_gradient(
        subset=["Recall", "F1", "PR_AUC"], cmap="Greens"
    ),
    use_container_width=True,
)

fig2 = go.Figure()
for metric in ["Accuracy", "Recall", "F1"]:
    fig2.add_trace(go.Bar(name=metric, x=display.index, y=display[metric]))
fig2.update_layout(
    barmode="group",
    title="Accuracy looks similar across models — recall does not",
    yaxis_title="Score",
    legend_title="Metric",
)
st.plotly_chart(fig2, use_container_width=True)

rf_orig = results.loc["RF (original)"]
st.markdown(
    f"**The trap, made concrete:** the original Random Forest reports "
    f"**{rf_orig['Accuracy']*100:.1f}% accuracy** — only "
    f"{(rf_orig['Accuracy']-baseline_acc)*100:.1f} points above the do-nothing "
    f"baseline — while its **recall is only {rf_orig['Recall']:.3f}**. It is "
    "catching roughly 1 in 5 real failures, and the accuracy number alone would "
    "never have told me that."
)

# ---------------------------------------------------------------------------
# Step 5 — Confusion matrix, in plain language
# ---------------------------------------------------------------------------

st.header("5 · What the numbers mean on the shop floor")

best_name = results["F1"].idxmax()
best_model = fitted[best_name]
pred = best_model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

st.markdown(f"Best model by F1: **{best_name}**")
m1, m2, m3 = st.columns(3)
m1.metric("Real failures in test set", f"{tp + fn}")
m2.metric("✅ Caught (true positives)", f"{tp}")
m3.metric("❌ Missed (false negatives)", f"{fn}")
st.metric("⚠️ False alarms (unnecessary inspections)", f"{fp}")
st.markdown(
    f"**Catch rate: {tp/(tp+fn)*100:.1f}%.** An F1 score of {results.loc[best_name,'F1']:.3f} "
    "means nothing to an engineer on the floor. *\"We missed "
    f"{fn} of {tp+fn} failures\"* does."
)

# ---------------------------------------------------------------------------
# Step 6 — Feature importance
# ---------------------------------------------------------------------------

st.header("6 · What actually drives failure")
imp = pd.Series(
    best_model.feature_importances_ if hasattr(best_model, "feature_importances_") else None,
    index=X.columns,
) if hasattr(best_model, "feature_importances_") else None

if imp is not None:
    imp_pct = (imp / imp.sum() * 100).sort_values(ascending=True)
    fig3 = px.bar(
        imp_pct, orientation="h",
        labels={"value": "Importance (%)", "index": "Feature"},
        title=f"Feature importance — {best_name}",
    )
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    top = imp_pct.idxmax()
    st.markdown(f"**{top}** is the dominant predictor, at **{imp_pct.max():.1f}%** — this held up unchanged from the original analysis.")

# ---------------------------------------------------------------------------
# Step 7 — The takeaway
# ---------------------------------------------------------------------------

st.header("7 · The lesson")
st.success(
    "**A model that's 97% \"accurate\" is not automatically a good model.** "
    "On imbalanced problems, accuracy can actively hide the exact thing you built "
    "the model to catch. Check the target variable's balance first — always — "
    "before you train anything."
)

st.divider()
st.markdown(
    f"\U0001F4D3 **Full notebook, all 8 corrections, and the code:** "
    f"[{NOTEBOOK_URL}]({NOTEBOOK_URL})\n\n"
    "Built by Ashok Medasani · M.S. Analytics, Saint Louis University"
)
