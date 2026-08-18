"""
TitanForge — Aerospace Component Failure Prediction
A 5-stage story: setup, the original (March 2025) analysis, the bias I missed,
the fix + corrected results, and the final conclusion.
Navigate with the Next / Back buttons. Charts are Chart.js (no Plotly).

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud, pointed at this repo's app.py
"""

import json
import itertools

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, roc_curve, confusion_matrix,
)
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="TitanForge — Failure Prediction, Corrected", page_icon="✈️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"]  { font-family: 'Manrope', -apple-system, sans-serif; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; font-weight: 800; letter-spacing: -0.01em; }
    h1 { font-size: 2.1rem !important; }
    h3 { color: #1f3a5f; }
    p, li, .stMarkdown { font-size: 0.98rem; line-height: 1.55; }
    code { font-family: 'JetBrains Mono', monospace; }

    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #1f3a5f; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #667; }

    .stButton>button { border-radius: 8px; font-weight: 600; border: 1px solid #d8dee8; }
    .stButton>button:hover { border-color: #1f3a5f; color: #1f3a5f; }

    div[data-testid="stProgress"] > div > div { background-color: #c44e52; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important; border-color: #eef1f6 !important;
        box-shadow: 0 1px 3px rgba(20,30,50,0.06);
    }

    @media (max-width: 640px) {
        h1 { font-size: 1.35rem !important; }
        h3 { font-size: 1.05rem !important; }
        p, li, .stMarkdown { font-size: 0.9rem !important; }
        div[data-testid="stMetricValue"] { font-size: 1.15rem !important; }
        div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
        .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; padding-top: 1rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chart.js helpers — every chart in this app is Chart.js, not Plotly.
# ---------------------------------------------------------------------------

PALETTE = ["#1f3a5f", "#c44e52", "#4c9a7d", "#e0a33e", "#7d5ba6", "#2c8fae"]
CHARTJS = "https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"
DATALABELS = "https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"
_id_counter = itertools.count()


def _wrap(canvas_id, config_json, height, use_datalabels=False, register_datalabels=False):
    plugin_script = f'<script src="{DATALABELS}"></script>' if use_datalabels else ""
    register = "Chart.register(ChartDataLabels);" if register_datalabels else ""
    return f"""
    <div style="font-family:'Manrope',-apple-system,sans-serif; background:#fff; border:1px solid #eef1f6;
                border-radius:14px; padding:14px 16px 6px 10px; box-shadow:0 1px 3px rgba(20,30,50,0.07);
                box-sizing:border-box; height:{height-16}px;">
      <canvas id="{canvas_id}"></canvas>
    </div>
    <script src="{CHARTJS}"></script>
    {plugin_script}
    <script>
      {register}
      const ctx_{canvas_id} = document.getElementById('{canvas_id}').getContext('2d');
      new Chart(ctx_{canvas_id}, {config_json});
    </script>
    """


def metric_line_compare(metric_names, series, title, height=380, y_min=None, y_max=None):
    """Straight-line comparison — metrics along the x-axis, one colored line per
    model/approach, y-axis auto-zoomed to the real spread so close scores
    (0.965 vs 0.981) are still visually distinguishable. Matches the
    'Model Performance Comparison' style from the original notebook."""
    cid = f"chart_{next(_id_counter)}"
    all_vals = [v for vals in series.values() for v in vals]
    if y_min is None:
        y_min = max(0, min(all_vals) - (max(all_vals) - min(all_vals)) * 0.25 - 0.01)
    if y_max is None:
        y_max = min(1.0, max(all_vals) + (max(all_vals) - min(all_vals)) * 0.25 + 0.01)
    datasets = []
    for i, (name, values) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        datasets.append({
            "label": name, "data": [round(v, 4) for v in values],
            "borderColor": color, "backgroundColor": color, "fill": False,
            "borderWidth": 3, "pointRadius": 5, "pointHoverRadius": 7, "tension": 0,
        })
    config = {
        "type": "line",
        "data": {"labels": metric_names, "datasets": datasets},
        "options": {
            "responsive": True, "maintainAspectRatio": False,
            "layout": {"padding": {"top": 8, "right": 16}},
            "plugins": {
                "title": {"display": True, "text": title, "font": {"size": 15, "weight": "700", "family": "Manrope"}, "color": "#1f3a5f", "padding": {"bottom": 14}},
                "legend": {"position": "right", "labels": {"font": {"family": "Manrope", "size": 11}, "boxWidth": 10, "usePointStyle": True, "padding": 14}},
                "tooltip": {"backgroundColor": "#1f3a5f", "cornerRadius": 8, "padding": 10, "callbacks": {"label": "__TIP__"}},
            },
            "scales": {
                "x": {"grid": {"color": "#f5f5f5"}, "ticks": {"font": {"family": "Manrope", "size": 11}}, "title": {"display": True, "text": "Evaluation metric", "font": {"family": "Manrope", "size": 11}}},
                "y": {"min": y_min, "max": y_max, "grid": {"color": "#f5f5f5"}, "ticks": {"font": {"family": "Manrope", "size": 10}}, "title": {"display": True, "text": "Score", "font": {"family": "Manrope", "size": 11}}},
            },
        },
    }
    cfg = json.dumps(config).replace('"__TIP__"', "(ctx) => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(3)")
    components.html(_wrap(cid, cfg, height), height=height + 10)


def hbar(labels, values, title, height=300, color=None, x_label="Importance (%)", is_percent=True):
    cid = f"chart_{next(_id_counter)}"
    order = np.argsort(values)  # ascending so the largest lands at the top after Chart.js reverse
    labels_o = [labels[i] for i in order][::-1]
    values_o = [round(float(values[i]), 2) for i in order][::-1]
    config = {
        "type": "bar",
        "data": {"labels": labels_o, "datasets": [{
            "data": values_o, "backgroundColor": color or PALETTE[0], "borderRadius": 5, "barPercentage": 0.7,
        }]},
        "options": {
            "indexAxis": "y", "responsive": True, "maintainAspectRatio": False,
            "layout": {"padding": {"top": 8, "right": 34}},
            "plugins": {
                "title": {"display": True, "text": title, "font": {"size": 15, "weight": "700", "family": "Manrope"}, "color": "#1f3a5f", "padding": {"bottom": 12}},
                "legend": {"display": False},
                "datalabels": {"anchor": "end", "align": "end", "color": "#33415c", "font": {"size": 10, "weight": "600", "family": "Manrope"}, "formatter": "__FMT__"},
                "tooltip": {"backgroundColor": "#1f3a5f", "cornerRadius": 8, "padding": 10},
            },
            "scales": {
                "x": {"grid": {"color": "#f2f2f2"}, "ticks": {"font": {"family": "Manrope", "size": 10}}, "title": {"display": bool(x_label), "text": x_label, "font": {"family": "Manrope", "size": 11}}},
                "y": {"grid": {"display": False}, "ticks": {"font": {"family": "Manrope", "size": 11.5, "weight": "600"}}},
            },
        },
    }
    fmt_js = "(v) => v.toFixed(1) + '%'" if is_percent else "(v) => v.toLocaleString()"
    cfg = json.dumps(config).replace('"__FMT__"', fmt_js)
    components.html(_wrap(cid, cfg, height, use_datalabels=True, register_datalabels=True), height=height + 10)


def line_curve(series, title, height=320, x_label="", y_label="", x_max=1.0, y_max=1.0):
    """series: list of {label, points:[{x,y},...], color, dash(optional bool)}"""
    cid = f"chart_{next(_id_counter)}"
    datasets = []
    for i, s in enumerate(series):
        color = s.get("color", PALETTE[i % len(PALETTE)])
        ds = {
            "label": s["label"], "data": s["points"], "borderColor": color,
            "backgroundColor": color + "22", "fill": s.get("fill", False),
            "borderWidth": 1 if s.get("dash") else 3, "pointRadius": 0, "tension": 0.15,
            "borderDash": [6, 4] if s.get("dash") else [],
        }
        datasets.append(ds)
    config = {
        "type": "line",
        "data": {"datasets": datasets},
        "options": {
            "responsive": True, "maintainAspectRatio": False, "parsing": False,
            "layout": {"padding": {"top": 4, "right": 12}},
            "plugins": {
                "title": {"display": True, "text": title, "font": {"size": 15, "weight": "700", "family": "Manrope"}, "color": "#1f3a5f", "padding": {"bottom": 12}},
                "legend": {"position": "bottom", "labels": {"font": {"family": "Manrope", "size": 11}, "boxWidth": 10, "usePointStyle": True, "padding": 12}},
                "tooltip": {"backgroundColor": "#1f3a5f", "cornerRadius": 8, "padding": 10},
            },
            "scales": {
                "x": {"type": "linear", "min": 0, "max": x_max, "grid": {"color": "#f2f2f2"}, "title": {"display": bool(x_label), "text": x_label, "font": {"family": "Manrope", "size": 11}}, "ticks": {"font": {"family": "Manrope", "size": 10}}},
                "y": {"min": 0, "max": y_max, "grid": {"color": "#f2f2f2"}, "title": {"display": bool(y_label), "text": y_label, "font": {"family": "Manrope", "size": 11}}, "ticks": {"font": {"family": "Manrope", "size": 10}}},
            },
        },
    }
    cfg = json.dumps(config)
    components.html(_wrap(cid, cfg, height), height=height + 10)


def doughnut(labels, values, colors, title, height=320):
    cid = f"chart_{next(_id_counter)}"
    config = {
        "type": "doughnut",
        "data": {"labels": labels, "datasets": [{"data": values, "backgroundColor": colors, "borderWidth": 3, "borderColor": "#fff"}]},
        "options": {
            "responsive": True, "maintainAspectRatio": False, "cutout": "62%",
            "plugins": {
                "title": {"display": True, "text": title, "font": {"size": 15, "weight": "700", "family": "Manrope"}, "color": "#1f3a5f", "padding": {"bottom": 14}},
                "legend": {"position": "bottom", "labels": {"font": {"family": "Manrope", "size": 11.5}, "usePointStyle": True, "padding": 14}},
                "tooltip": {"backgroundColor": "#1f3a5f", "cornerRadius": 8, "padding": 10, "callbacks": {"label": "__TIP__"}},
            },
        },
    }
    cfg = json.dumps(config).replace(
        '"__TIP__"',
        "(ctx) => ctx.label + ': ' + ctx.parsed.toLocaleString() + ' (' + (ctx.parsed / ctx.dataset.data.reduce((a,b)=>a+b,0) * 100).toFixed(1) + '%)'",
    )
    components.html(_wrap(cid, cfg, height), height=height + 10)


def confusion_matrix_card(cm, title, class_labels=("No failure", "Failure")):
    """Hand-built HTML/CSS confusion matrix — big, legible, fully responsive,
    no Plotly aspect-ratio issues."""
    tn, fp, fn, tp = cm.ravel()
    vmax = max(tn, fp, fn, tp) or 1

    def cell(value, is_correct):
        intensity = 0.15 + 0.65 * (value / vmax)
        bg = f"rgba(31,58,95,{intensity:.2f})" if is_correct else f"rgba(196,78,82,{intensity:.2f})"
        text_color = "#fff" if intensity > 0.45 else "#1f2937"
        return (
            f'<div style="background:{bg};color:{text_color};border-radius:10px;'
            f'padding:18px 10px;text-align:center;font-weight:700;font-size:1.6rem;">{value:,}'
            f'<div style="font-size:0.7rem;font-weight:500;opacity:0.85;margin-top:4px;">'
            f'{"correct" if is_correct else "error"}</div></div>'
        )

    html = f"""
    <div style="font-family:'Manrope',-apple-system,sans-serif; background:#fff; border:1px solid #eef1f6;
                border-radius:14px; padding:16px 18px; box-shadow:0 1px 3px rgba(20,30,50,0.07); box-sizing:border-box;">
      <div style="font-weight:700; color:#1f3a5f; font-size:0.95rem; margin-bottom:12px;">{title}</div>
      <div style="display:grid; grid-template-columns: 90px 1fr 1fr; grid-template-rows: auto 1fr 1fr; gap:6px; align-items:center;">
        <div></div>
        <div style="text-align:center; font-size:0.75rem; color:#667; font-weight:600;">Predicted: {class_labels[0]}</div>
        <div style="text-align:center; font-size:0.75rem; color:#667; font-weight:600;">Predicted: {class_labels[1]}</div>
        <div style="font-size:0.75rem; color:#667; font-weight:600;">Actual: {class_labels[0]}</div>
        {cell(tn, True)}
        {cell(fp, False)}
        <div style="font-size:0.75rem; color:#667; font-weight:600;">Actual: {class_labels[1]}</div>
        {cell(fn, False)}
        {cell(tp, True)}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data + both pipelines (mirrors both notebooks exactly)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_excel("TitanForge_dataset.xlsx")


@st.cache_data
def build_features(df):
    work = df.drop(["UDI", "Product ID"], axis=1)
    work = pd.get_dummies(work, columns=["Type"], prefix="Type")
    return work


def score_row(name, model, X_eval, y_eval, X_train_eval, y_train_eval):
    pred = model.predict(X_eval)
    proba = model.predict_proba(X_eval)[:, 1]
    return dict(
        Model=name,
        Accuracy=accuracy_score(y_eval, pred),
        Precision=precision_score(y_eval, pred, zero_division=0),
        Recall=recall_score(y_eval, pred),
        F1=f1_score(y_eval, pred),
        ROC_AUC=roc_auc_score(y_eval, proba),
        PR_AUC=average_precision_score(y_eval, proba),
        Train_acc=accuracy_score(y_train_eval, model.predict(X_train_eval)),
    )


@st.cache_resource
def run_original_analysis(work):
    """Reproduces the March 2025 exam notebook exactly: raw (unflipped) target,
    plain train_test_split (no stratify), weighted-average metrics."""
    X = work.drop(["Failure_Status"], axis=1)
    y = work["Failure_Status"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    rf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=5, random_state=1)
    rf.fit(x_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, loss="log_loss", random_state=1)
    gb.fit(x_train, y_train)

    def weighted_row(name, model):
        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)[:, 1]
        return dict(
            Model=name,
            Accuracy=accuracy_score(y_test, pred),
            Recall_weighted=recall_score(y_test, pred, average="weighted"),
            Precision_weighted=precision_score(y_test, pred, average="weighted"),
            F1_weighted=f1_score(y_test, pred, average="weighted"),
            ROC_AUC=roc_auc_score(y_test, proba),
            Recall_rare_class=recall_score(y_test, pred, pos_label=0),
            Precision_rare_class=precision_score(y_test, pred, pos_label=0, zero_division=0),
        )

    rows = [weighted_row("Random Forest", rf), weighted_row("Gradient Boosting", gb)]
    results = pd.DataFrame(rows).set_index("Model")

    rf_imp = pd.Series(rf.feature_importances_, index=X.columns)
    gb_imp = pd.Series(gb.feature_importances_, index=X.columns)

    return dict(
        X=X, y=y, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
        rf=rf, gb=gb, results=results, rf_imp=rf_imp, gb_imp=gb_imp,
        rare_class_rate=(y == 0).mean(),
    )


@st.cache_resource
def run_corrected_analysis(work):
    """The reworked pipeline: relabel to FAILURE = 1 - Failure_Status (so 1 = the
    rare, real failure event), stratified split, and three ways of handling the
    imbalance: plain, class-weighted, and SMOTE data-level balancing."""
    df2 = work.copy()
    df2["FAILURE"] = 1 - df2["Failure_Status"]
    X = df2.drop(["Failure_Status", "FAILURE"], axis=1)
    y = df2["FAILURE"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    rows, fitted = [], {}

    rf_plain = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    rf_plain.fit(x_train, y_train)
    fitted["RF (relabeled, no balancing)"] = rf_plain
    rows.append(score_row("RF (relabeled, no balancing)", rf_plain, x_test, y_test, x_train, y_train))

    rf_w = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1, class_weight="balanced")
    rf_w.fit(x_train, y_train)
    fitted["RF class-weighted"] = rf_w
    rows.append(score_row("RF class-weighted", rf_w, x_test, y_test, x_train, y_train))

    sm = SMOTE(random_state=1)
    x_res, y_res = sm.fit_resample(x_train, y_train)
    rf_sm = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    rf_sm.fit(x_res, y_res)
    fitted["RF SMOTE-balanced"] = rf_sm
    rows.append(score_row("RF SMOTE-balanced", rf_sm, x_test, y_test, x_res, y_res))

    gb_tuned = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=1)
    gb_tuned.fit(x_train, y_train)
    fitted["GB tuned"] = gb_tuned
    rows.append(score_row("GB tuned", gb_tuned, x_test, y_test, x_train, y_train))

    results = pd.DataFrame(rows).set_index("Model")
    baseline_acc = 1 - y_test.mean()
    smote_dist = {"before": y_train.value_counts().to_dict(), "after": y_res.value_counts().to_dict()}

    return dict(
        X=X, y=y, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
        fitted=fitted, results=results, baseline_acc=baseline_acc, smote_dist=smote_dist,
        n_total=len(df2), n_fail=int(y.sum()),
    )


df_raw = load_data()
work = build_features(df_raw)
orig = run_original_analysis(work)
corr = run_corrected_analysis(work)

# ---------------------------------------------------------------------------
# Page framework — Next / Back navigation + Home
# ---------------------------------------------------------------------------

STEPS = [
    "1 · Setup — the question & the data",
    "2 · A year ago — results & my conclusion",
    "3 · The bias I missed",
    "4 · The fix & the corrected results",
    "5 · Final conclusion",
]

GITHUB_URL = "https://github.com/ashokmedasani/Titan-Forge"
NOTEBOOK_URL_ORIGINAL = f"{GITHUB_URL}/blob/main/Titan_Forge.ipynb"
NOTEBOOK_URL_CORRECTED = f"{GITHUB_URL}/blob/main/Titan_Forge_Corrected.ipynb"

if "step" not in st.session_state:
    st.session_state.step = 0
# Clamp in case a stale session (e.g. from a previous version of this app with
# a different number of stages) left an out-of-range value behind.
st.session_state.step = max(0, min(st.session_state.step, len(STEPS) - 1))

title_col, home_col = st.columns([5, 1])
with title_col:
    st.title("✈️ TitanForge — Aerospace Component Failure Prediction")
    st.caption("A step-by-step walkthrough: the original analysis, the bias I missed, and the corrected, balanced result.")
with home_col:
    st.write("")
    if st.button("🏠 Home", use_container_width=True, disabled=st.session_state.step == 0):
        st.session_state.step = 0
        st.rerun()

st.markdown(
    f"**Original notebook:** [{NOTEBOOK_URL_ORIGINAL}]({NOTEBOOK_URL_ORIGINAL}) · "
    f"**Corrected notebook:** [{NOTEBOOK_URL_CORRECTED}]({NOTEBOOK_URL_CORRECTED})"
)

st.progress((st.session_state.step + 1) / len(STEPS))
st.markdown(f"### {STEPS[st.session_state.step]}")
st.divider()

step = st.session_state.step

# ---------------------------------------------------------------------------
# Step 0 — Setup: the research question + data & analysis overview
# ---------------------------------------------------------------------------
if step == 0:
    st.markdown(
        "TitanForge Industries manufactures forged aerospace components. Machine failures during "
        "the forging process are expensive and dangerous: production halts, emergency repair costs, "
        "wasted material, and safety risk to workers near the equipment."
    )
    st.markdown(
        "**The question:** using real-time sensor readings — air temperature, process temperature, "
        "rotational speed, torque, and tool wear — can we predict which machine runs are heading "
        "toward failure, early enough to act on it? And once we build that model, how do we know "
        "if the number it reports can actually be trusted?"
    )

    st.markdown("**A quick, plain-language look at what I was working with** — not the full detail, just the shape of it.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (machine runs)", f"{len(df_raw):,}")
    c2.metric("Sensor features used", "5")
    c3.metric("Machine types", df_raw["Type"].nunique())
    st.markdown(
        "**What the analysis did, in short:**\n\n"
        "- Dropped ID columns that carry no predictive signal (row ID, product ID)\n"
        "- One-hot encoded the machine `Type` (L / M / H) so models can use it\n"
        "- Checked for missing values and outliers across all sensor columns — dataset was clean, "
        "no missing values\n"
        "- Looked at correlations between sensors and the failure outcome\n"
        "- Split the data into a training set and a held-out test set, then trained two model "
        "families — Random Forest and Gradient Boosting — and compared them"
    )
    type_counts = df_raw["Type"].value_counts()
    hbar(list(type_counts.index), list(type_counts.values), "Machine type distribution",
         height=260, color=PALETTE[0], x_label="Count", is_percent=False)

# ---------------------------------------------------------------------------
# Step 1 — A year ago: results (detailed) + my conclusion at the time
# ---------------------------------------------------------------------------
elif step == 1:
    st.markdown(
        "This is exactly what I trained and reported in March 2025: a Random Forest and a Gradient "
        "Boosting classifier, evaluated with **weighted-average** accuracy, precision, recall, and F1 "
        "— plus ROC-AUC, computed here for the first time to show what the weighted metrics were hiding."
    )

    st.markdown("**Full scorecard, both models:**")
    sc1, sc2 = st.columns(2)
    for col, name in [(sc1, "Random Forest"), (sc2, "Gradient Boosting")]:
        rr = orig["results"].loc[name]
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                m1, m2 = st.columns(2)
                m1.metric("Accuracy", f"{rr['Accuracy']:.3f}")
                m2.metric("ROC-AUC", f"{rr['ROC_AUC']:.3f}")
                m3, m4 = st.columns(2)
                m3.metric("Precision (weighted)", f"{rr['Precision_weighted']:.3f}")
                m4.metric("Recall (weighted)", f"{rr['Recall_weighted']:.3f}")
                st.metric("F1 (weighted)", f"{rr['F1_weighted']:.3f}")

    st.write("")
    r = orig["results"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    series = {
        "Random Forest": [r.loc["Random Forest", "Accuracy"], r.loc["Random Forest", "Precision_weighted"],
                           r.loc["Random Forest", "Recall_weighted"], r.loc["Random Forest", "F1_weighted"],
                           r.loc["Random Forest", "ROC_AUC"]],
        "Gradient Boosting": [r.loc["Gradient Boosting", "Accuracy"], r.loc["Gradient Boosting", "Precision_weighted"],
                               r.loc["Gradient Boosting", "Recall_weighted"], r.loc["Gradient Boosting", "F1_weighted"],
                               r.loc["Gradient Boosting", "ROC_AUC"]],
    }
    metric_line_compare(metric_names, series, "Model Performance Comparison", height=360)
    st.caption("The y-axis is zoomed into the real spread (not 0–1) — the values are close (0.94–0.98), which is exactly why a quick glance at accuracy alone was so easy to trust.")

    c1, c2 = st.columns(2)
    with c1:
        rf_fpr, rf_tpr, _ = roc_curve(orig["y_test"], orig["rf"].predict_proba(orig["x_test"])[:, 1])
        gb_fpr, gb_tpr, _ = roc_curve(orig["y_test"], orig["gb"].predict_proba(orig["x_test"])[:, 1])
        rf_auc = orig["results"].loc["Random Forest", "ROC_AUC"]
        gb_auc = orig["results"].loc["Gradient Boosting", "ROC_AUC"]
        line_curve(
            [
                {"label": f"Random Forest (AUC={rf_auc:.3f})", "points": [{"x": float(x), "y": float(y)} for x, y in zip(rf_fpr, rf_tpr)], "color": PALETTE[0]},
                {"label": f"Gradient Boosting (AUC={gb_auc:.3f})", "points": [{"x": float(x), "y": float(y)} for x, y in zip(gb_fpr, gb_tpr)], "color": PALETTE[1]},
                {"label": "Random guess", "points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}], "color": "#999", "dash": True},
            ],
            "ROC curve — test set", height=300, x_label="False positive rate", y_label="True positive rate",
        )
    with c2:
        imp = orig["rf_imp"] / orig["rf_imp"].sum() * 100
        hbar(list(imp.index), list(imp.values), "Feature importance — Random Forest", height=300, color=PALETTE[0])

    c1, c2 = st.columns(2)
    with c1:
        pred_rf = orig["rf"].predict(orig["x_test"])
        confusion_matrix_card(confusion_matrix(orig["y_test"], pred_rf), "Random Forest — confusion matrix",
                               class_labels=("0", "1"))
    with c2:
        pred_gb = orig["gb"].predict(orig["x_test"])
        confusion_matrix_card(confusion_matrix(orig["y_test"], pred_gb), "Gradient Boosting — confusion matrix",
                               class_labels=("0", "1"))
    st.caption("Torque came out as the dominant driver in both models — that held up in the corrected analysis too.")

    st.divider()
    st.markdown(
        f"**My conclusion at the time:** Gradient Boosting was the better model — higher accuracy "
        f"({r.loc['Gradient Boosting','Accuracy']*100:.1f}% vs {r.loc['Random Forest','Accuracy']*100:.1f}%), "
        f"higher weighted recall, precision, and F1 across the board. I recommended it as the model "
        "TitanForge should deploy, and proposed two preventive actions based on the feature importance:"
    )
    st.markdown(
        "1. **Real-time torque monitoring and control** — torque was the single most influential "
        "feature in both models, so catching abnormal torque early was the highest-leverage fix.\n"
        "2. **Air temperature regulation + a tool-wear-based maintenance schedule** — to reduce "
        "failures tied to ambient conditions and worn tooling."
    )
    st.info(
        "At the time this felt like a complete analysis: two models compared, a clear winner, "
        "concrete recommendations. What I didn't do — and didn't think to do — was check how the "
        "outcome variable itself was distributed before trusting any of those numbers."
    )

# ---------------------------------------------------------------------------
# Step 2 — The bias I missed
# ---------------------------------------------------------------------------
elif step == 2:
    rare_rate = orig["rare_class_rate"]
    st.markdown(
        "Revisiting this recently, I went back to the most basic question in any classification "
        "problem — one I skipped entirely the first time: **how is the target variable actually "
        "distributed?**"
    )
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True, height=460):
            st.metric("Rare class in the data", f"{int(rare_rate*len(df_raw)):,} of {len(df_raw):,} rows", f"{rare_rate*100:.2f}%")
            st.markdown(
                "Only 3.48% of rows belong to the rare class — the one that actually represents the "
                "real machine failures I needed the model to catch. I never checked this ratio in the "
                "original pass. It's a basic step, and skipping it is a common mistake at the landing "
                "stage — but here, it was the one that mattered most."
            )
            rf_r = orig["results"].loc["Random Forest"]
            gb_r = orig["results"].loc["Gradient Boosting"]
            st.warning(
                f"Weighted recall was {rf_r['Recall_weighted']:.3f} (RF) and {gb_r['Recall_weighted']:.3f} "
                f"(GB) — both looked excellent. But recall on the rare, real-failure class alone was only "
                f"**{rf_r['Recall_rare_class']:.3f} for Random Forest** and "
                f"**{gb_r['Recall_rare_class']:.3f} for Gradient Boosting**. The weighted average was "
                "dominated by the 96.5% majority class and hid exactly the number I needed to see."
            )
    with c2:
        with st.container(border=True, height=460):
            doughnut(
                ["Majority class", "Rare class (real failures)"],
                [int(len(df_raw)*(1-rare_rate)), int(len(df_raw)*rare_rate)],
                ["#4c72b0", "#c44e52"],
                "Target variable balance — the check I skipped", height=390,
            )

# ---------------------------------------------------------------------------
# Step 3 — How I balanced & reworked it + the corrected results
# ---------------------------------------------------------------------------
elif step == 3:
    st.markdown("Once I saw the real class balance, I reworked the analysis in three concrete ways:")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1 · Stratified split")
        st.markdown("Train and test sets now both keep the true ~3.5% failure rate, so the test set isn't accidentally easier than reality.")
    with c2:
        st.subheader("2 · Class-weighted training")
        st.markdown("`class_weight='balanced'` — the data stays imbalanced, but the model pays a real penalty for missing a failure during training.")
    with c3:
        st.subheader("3 · Real data-level balancing")
        st.markdown("**SMOTE** oversampling on the training set only — synthetic failure examples generated until the training data is 50/50, tested against the untouched real test set.")

    before, after = corr["smote_dist"]["before"], corr["smote_dist"]["after"]
    st.markdown(
        f"**SMOTE, concretely:** training set went from **{before.get(0,0):,} no-failure / "
        f"{before.get(1,0):,} failure** to **{after.get(0,0):,} / {after.get(1,0):,}** — an actual "
        "50/50 balance in the data the model learns from, not just a reweighted loss function."
    )
    st.markdown("I also replaced the scorecard itself: instead of weighted accuracy/precision/recall/F1, every model below is judged on **precision, recall, F1, ROC-AUC, and PR-AUC computed on the real failure class specifically.**")

    st.divider()
    r = corr["results"]
    st.markdown(
        "**The corrected results** — same depth as the original analysis, mirrored: a full "
        "scorecard for every approach, the same metric comparison style, and a confusion matrix "
        "for each one, so the two analyses sit side by side on equal footing."
    )

    st.markdown("**Full scorecard, every approach:**")
    sc_cols = st.columns(2)
    for i, name in enumerate(r.index):
        rr = r.loc[name]
        with sc_cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                m1, m2 = st.columns(2)
                m1.metric("Accuracy", f"{rr['Accuracy']:.3f}")
                m2.metric("ROC-AUC", f"{rr['ROC_AUC']:.3f}")
                m3, m4 = st.columns(2)
                m3.metric("Precision", f"{rr['Precision']:.3f}")
                m4.metric("Recall", f"{rr['Recall']:.3f}")
                m5, m6 = st.columns(2)
                m5.metric("F1", f"{rr['F1']:.3f}")
                m6.metric("PR-AUC", f"{rr['PR_AUC']:.3f}")

    st.write("")
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    series = {
        name: [r.loc[name, "Accuracy"], r.loc[name, "Precision"], r.loc[name, "Recall"],
               r.loc[name, "F1"], r.loc[name, "ROC_AUC"]]
        for name in r.index
    }
    metric_line_compare(metric_names, series, "Every corrected approach catches far more real failures", height=380)

    orig_rare_recall = orig["results"].loc["Random Forest", "Recall_rare_class"]
    st.markdown(
        f"**The jump, made concrete:** a year ago, Random Forest caught only "
        f"{orig_rare_recall:.3f} recall on the real failure class. After relabeling, stratifying, "
        f"and balancing: **class-weighting reaches {r.loc['RF class-weighted','Recall']:.3f} recall**, "
        f"**SMOTE reaches {r.loc['RF SMOTE-balanced','Recall']:.3f} recall** — both real, verified "
        f"improvements, with different precision trade-offs "
        f"({r.loc['RF class-weighted','Precision']:.3f} vs {r.loc['RF SMOTE-balanced','Precision']:.3f})."
    )

    best_name = r["F1"].idxmax()
    st.markdown(f"**Confusion matrix, every approach** (best overall by F1: **{best_name}**):")
    cm_cols = st.columns(2)
    for i, name in enumerate(r.index):
        model = corr["fitted"][name]
        pred = model.predict(corr["x_test"])
        cm = confusion_matrix(corr["y_test"], pred)
        with cm_cols[i % 2]:
            label = f"{name} ⭐ best by F1" if name == best_name else name
            confusion_matrix_card(cm, label, class_labels=("No failure", "Failure"))

    tn, fp, fn, tp = confusion_matrix(corr["y_test"], corr["fitted"][best_name].predict(corr["x_test"])).ravel()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Real failures in test set", f"{tp+fn}")
    m2.metric("✅ Caught", f"{tp}")
    m3.metric("❌ Missed", f"{fn}")
    m4.metric("⚠️ False alarms", f"{fp}")

    best_model = corr["fitted"][best_name]
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=corr["X"].columns)
        imp_pct = imp / imp.sum() * 100
        hbar(list(imp_pct.index), list(imp_pct.values), f"Feature importance — {best_name}", height=300, color=PALETTE[2])

# ---------------------------------------------------------------------------
# Step 4 — Final conclusion
# ---------------------------------------------------------------------------
elif step == 4:
    r = corr["results"]
    best_name = r["F1"].idxmax()
    st.markdown(
        f"**Final recommendation:** {best_name} — it posts the best F1 "
        f"({r.loc[best_name,'F1']:.3f}) of every approach tested, balancing catching real failures "
        "against not overwhelming the floor with false alarms. Torque, rotational speed, and tool "
        "wear remain the dominant drivers, unchanged from the original analysis — that part of the "
        "original conclusion held up."
    )
    st.success(
        "**The lesson:** a model that's 97% \"accurate\" is not automatically a good model. On "
        "imbalanced problems, weighted metrics can hide the exact thing you built the model to "
        "catch. Once you know the data is imbalanced, there's more than one honest fix — "
        "reweighting the loss function and rebalancing the data itself both work — and it's worth "
        "verifying both instead of assuming either one automatically wins. The single highest-"
        "leverage step in this whole project was also the simplest: check how the target variable "
        "is distributed before you train anything."
    )
    st.markdown(
        f"📓 **Original notebook (March 2025):** [{NOTEBOOK_URL_ORIGINAL}]({NOTEBOOK_URL_ORIGINAL})\n\n"
        f"📓 **Corrected notebook, all steps:** [{NOTEBOOK_URL_CORRECTED}]({NOTEBOOK_URL_CORRECTED})\n\n"
        "Built by Ashok Medasani · M.S. Analytics, Saint Louis University"
    )

st.write("")
st.divider()
nav1, nav2, nav3 = st.columns([1, 4, 1])
with nav1:
    if st.button("← Back", disabled=st.session_state.step == 0, use_container_width=True):
        st.session_state.step -= 1
        st.rerun()
with nav3:
    if st.button("Next →", disabled=st.session_state.step == len(STEPS) - 1, use_container_width=True):
        st.session_state.step += 1
        st.rerun()
