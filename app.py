"""
Fraud Risk Console
A Streamlit UI for the Hybrid Fraud Risk Scoring Framework
(Isolation Forest + Autoencoder) built in financial_transaction_fraud_detection.ipynb

Run:
    streamlit run app.py

Expects the artifacts produced by the notebook's "Save Models" section:
    saved_models/isolation_forest.pkl
    saved_models/autoencoder.keras
    saved_models/scaler.pkl
    saved_models/risk_configuration.pkl

Optionally, for extra pages, point the sidebar at:
    model_comparison.csv         (from Section 13 of the notebook)
    high_risk_transactions.csv   (from Section 17 of the notebook)
    creditcard.csv               (the original dataset, for sampling real transactions)
"""

import glob
import math
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fraud Risk Console",
    page_icon="\u25c9",
    layout="wide",
    initial_sidebar_state="expanded",
)

TF_AVAILABLE = True
try:
    from tensorflow import keras
except Exception:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Design system — colors, type, and the signature gauge
# ---------------------------------------------------------------------------

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #0B1220;
    --surface: #131B2E;
    --surface-2: #1A2338;
    --border: #263045;
    --text: #E8ECF4;
    --text-muted: #8A94A8;
    --low: #35C482;
    --mid: #F0A93D;
    --high: #E5484D;
    --info: #5B8DEF;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp {
    background: var(--bg);
    color: var(--text);
}

h1, h2, h3, .app-title { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }

[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.app-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 22px;
}
.app-title { font-size: 26px; font-weight: 700; color: var(--text); margin: 0; }
.app-subtitle { font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--text-muted); }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.card-label {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.card-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 700;
    color: var(--text);
}
.card-sub { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 4px 12px;
    border-radius: 999px;
    text-transform: uppercase;
}
.badge-low  { background: rgba(53,196,130,0.15); color: var(--low); border: 1px solid rgba(53,196,130,0.4); }
.badge-mid  { background: rgba(240,169,61,0.15); color: var(--mid); border: 1px solid rgba(240,169,61,0.4); }
.badge-high { background: rgba(229,72,77,0.15); color: var(--high); border: 1px solid rgba(229,72,77,0.4); }

.signal-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: var(--text);
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
}
.signal-row:last-child { border-bottom: none; }

.stButton>button {
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}
.stButton>button:hover { border-color: var(--info); color: var(--info); }

[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; color: var(--text); }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

.stDataFrame { border: 1px solid var(--border); border-radius: 8px; }

hr { border-color: var(--border); }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def risk_color(category: str) -> str:
    return {"Low Risk": "#35C482", "Medium Risk": "#F0A93D", "High Risk": "#E5484D"}[category]


def risk_badge_class(category: str) -> str:
    return {"Low Risk": "badge-low", "Medium Risk": "badge-mid", "High Risk": "badge-high"}[category]


def risk_category(score: float) -> str:
    if score <= 30:
        return "Low Risk"
    elif score <= 70:
        return "Medium Risk"
    return "High Risk"


def polar_to_xy(cx, cy, r, theta_deg):
    theta_rad = math.radians(theta_deg)
    return cx + r * math.cos(theta_rad), cy - r * math.sin(theta_rad)


def arc_path(cx, cy, r, theta_start, theta_end):
    x1, y1 = polar_to_xy(cx, cy, r, theta_start)
    x2, y2 = polar_to_xy(cx, cy, r, theta_end)
    large_arc = 1 if abs(theta_start - theta_end) > 180 else 0
    return f"M {x1:.2f} {y1:.2f} A {r} {r} 0 {large_arc} 1 {x2:.2f} {y2:.2f}"


def render_gauge(score: float, threshold: float, size: int = 260) -> str:
    """The signature element: a calibrated three-band risk dial. The three arc
    segments encode the Low / Medium / High risk bands (0-30 / 31-70 / 71-100),
    and the needle marks both the current score and, as a tick, the adaptive
    decision threshold learned from the data."""
    score = max(0.0, min(100.0, float(score)))
    cx, cy, r = 150, 150, 108
    band_low = arc_path(cx, cy, r, 180, 126)
    band_mid = arc_path(cx, cy, r, 126, 54)
    band_high = arc_path(cx, cy, r, 54, 0)

    needle_theta = 180 - (score / 100) * 180
    nx, ny = polar_to_xy(cx, cy, r - 22, needle_theta)

    thr_theta = 180 - (max(0.0, min(100.0, threshold)) / 100) * 180
    tx1, ty1 = polar_to_xy(cx, cy, r + 10, thr_theta)
    tx2, ty2 = polar_to_xy(cx, cy, r - 10, thr_theta)

    category = risk_category(score)
    color = risk_color(category)

    return f"""
    <svg viewBox="0 0 300 190" width="100%" height="{size}" xmlns="http://www.w3.org/2000/svg">
      <path d="{band_low}" fill="none" stroke="#35C482" stroke-width="16" stroke-linecap="butt" opacity="0.85"/>
      <path d="{band_mid}" fill="none" stroke="#F0A93D" stroke-width="16" stroke-linecap="butt" opacity="0.85"/>
      <path d="{band_high}" fill="none" stroke="#E5484D" stroke-width="16" stroke-linecap="butt" opacity="0.85"/>
      <line x1="{tx1:.2f}" y1="{ty1:.2f}" x2="{tx2:.2f}" y2="{ty2:.2f}" stroke="#E8ECF4" stroke-width="2" opacity="0.6"/>
      <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="{color}" stroke-width="3.5" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="7" fill="{color}"/>
      <circle cx="{cx}" cy="{cy}" r="3" fill="#0B1220"/>
      <text x="150" y="150" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="0" fill="none"></text>
      <text x="34" y="168" font-family="IBM Plex Mono, monospace" font-size="11" fill="#8A94A8">0</text>
      <text x="140" y="35" font-family="IBM Plex Mono, monospace" font-size="11" fill="#8A94A8">50</text>
      <text x="252" y="168" font-family="IBM Plex Mono, monospace" font-size="11" fill="#8A94A8">100</text>
      <text x="150" y="150" text-anchor="middle" dy="45" font-family="Space Grotesk, sans-serif" font-size="34" font-weight="700" fill="{color}">{score:.1f}</text>
      <text x="150" y="150" text-anchor="middle" dy="65" font-family="IBM Plex Mono, monospace" font-size="11" fill="#8A94A8">FRAUD RISK SCORE</text>
    </svg>
    """


# ---------------------------------------------------------------------------
# Scoring core — mirrors the notebook's hybrid framework exactly, but is
# self-contained so this app never depends on the notebook's kernel state.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained models...")
def load_models(models_dir: str):
    iso_path = os.path.join(models_dir, "isolation_forest.pkl")
    ae_path = os.path.join(models_dir, "autoencoder.keras")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    config_path = os.path.join(models_dir, "risk_configuration.pkl")

    missing = [p for p in [iso_path, ae_path, scaler_path, config_path] if not os.path.exists(p)]
    if missing:
        return None, f"Missing artifact(s): {', '.join(os.path.basename(m) for m in missing)}"

    if not TF_AVAILABLE:
        return None, "TensorFlow is not installed in this environment. Run: pip install tensorflow"

    try:
        iso_forest = joblib.load(iso_path)
        autoencoder = keras.models.load_model(ae_path)
        scaler = joblib.load(scaler_path)
        config = joblib.load(config_path)
    except Exception as e:
        return None, f"Failed to load artifacts: {e}"

    bundle = {
        "iso_forest": iso_forest,
        "autoencoder": autoencoder,
        "scaler": scaler,
        "config": config,
    }
    return bundle, None


def isoforest_raw_anomaly_score(model, X_scaled):
    return -model.score_samples(X_scaled)


def normalize_score(raw_scores, low, high):
    denom = (high - low) if (high - low) > 1e-12 else 1e-12
    return np.clip((raw_scores - low) / denom, 0, 1)


def reconstruction_error(model, X_scaled_values):
    reconstructed = model.predict(X_scaled_values, verbose=0)
    return np.mean(np.square(X_scaled_values - reconstructed), axis=1)


def score_dataframe(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full hybrid framework on a dataframe of raw transactions
    (Time, V1-V28, Amount) and returns a dataframe of per-transaction scores."""
    cfg = bundle["config"]
    feature_cols = cfg["FEATURE_COLUMNS"]

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required columns: {missing_cols}")

    X = df[feature_cols].copy()
    scaled = bundle["scaler"].transform(X)
    scaled_df = pd.DataFrame(scaled, columns=feature_cols)

    iso_raw = isoforest_raw_anomaly_score(bundle["iso_forest"], scaled_df)
    iso_norm = normalize_score(iso_raw, cfg["iso_low"], cfg["iso_high"])

    ae_err = reconstruction_error(bundle["autoencoder"], scaled_df.values)
    ae_norm = normalize_score(ae_err, cfg["ae_low"], cfg["ae_high"])

    hybrid = (cfg["ISO_WEIGHT"] * iso_norm + cfg["AE_WEIGHT"] * ae_norm) * 100
    categories = np.array([risk_category(s) for s in hybrid])
    decisions = np.where(hybrid >= cfg["ADAPTIVE_THRESHOLD"], "FRAUD", "LEGITIMATE")

    out = df.copy().reset_index(drop=True)
    out["isolation_forest_score"] = np.round(iso_norm, 4)
    out["autoencoder_score"] = np.round(ae_norm, 4)
    out["hybrid_risk_score"] = np.round(hybrid, 2)
    out["risk_level"] = categories
    out["decision"] = decisions
    return out


def explain_row(row: pd.Series, amount_high_cutoff: float, threshold: float) -> list:
    signals = []
    if row["isolation_forest_score"] > 0.7:
        signals.append(f"High Isolation Forest anomaly score ({row['isolation_forest_score']:.2f})")
    if row["autoencoder_score"] > 0.7:
        signals.append(f"High Autoencoder reconstruction error ({row['autoencoder_score']:.2f})")
    if "Amount" in row and row["Amount"] > amount_high_cutoff:
        signals.append(f"Unusually high transaction amount (${row['Amount']:.2f})")
    if row["hybrid_risk_score"] >= threshold:
        signals.append(f"Hybrid risk score ({row['hybrid_risk_score']:.1f}) above adaptive threshold ({threshold:.1f})")
    if not signals:
        signals.append("No strong individual anomaly signals detected; risk score within normal range")
    return signals


@st.cache_data(show_spinner="Reading reference dataset...")
def load_reference_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def find_local_creditcard_csv() -> str:
    candidates = glob.glob(os.path.expanduser("~/.cache/kagglehub/**/creditcard.csv"), recursive=True)
    candidates += glob.glob("**/creditcard.csv", recursive=True)
    return candidates[0] if candidates else ""


# ---------------------------------------------------------------------------
# Sidebar — model + data source configuration and navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="app-title" style="font-size:19px;">\u25c9 Fraud Risk Console</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">hybrid isolation forest + autoencoder</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview", "Score a Transaction", "Batch Scoring", "Model Comparison", "About & Limitations"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="card-label">Model artifacts</div>', unsafe_allow_html=True)
    models_dir = st.text_input("saved_models folder", value="saved_models")

    st.markdown('<div class="card-label" style="margin-top:14px;">Reference dataset (optional)</div>', unsafe_allow_html=True)
    default_csv = find_local_creditcard_csv()
    csv_path = st.text_input("creditcard.csv path", value=default_csv, placeholder="path/to/creditcard.csv")
    uploaded_reference = st.file_uploader("...or upload it", type="csv", key="ref_upload")

    st.markdown('<div class="card-label" style="margin-top:14px;">Notebook exports (optional)</div>', unsafe_allow_html=True)
    comparison_csv_path = st.text_input("model_comparison.csv", value="model_comparison.csv")
    high_risk_csv_path = st.text_input("high_risk_transactions.csv", value="high_risk_transactions.csv")

bundle, load_error = load_models(models_dir)

reference_df = None
if uploaded_reference is not None:
    reference_df = pd.read_csv(uploaded_reference)
elif csv_path and os.path.exists(csv_path):
    try:
        reference_df = load_reference_csv(csv_path)
    except Exception:
        reference_df = None

st.markdown(
    '<div class="app-header"><span class="app-title">Fraud Risk Console</span>'
    '<span class="app-subtitle">hybrid anomaly-based fraud risk scoring</span></div>',
    unsafe_allow_html=True,
)

if load_error:
    st.error(
        f"**Models not loaded:** {load_error}\n\n"
        f"Run `financial_transaction_fraud_detection.ipynb` end-to-end first — its "
        f"Section 19 saves the four required files into a `saved_models/` folder. "
        f"Then point the sidebar field at that folder (default: `saved_models`, "
        f"relative to wherever you launch `streamlit run app.py`)."
    )
    st.stop()

cfg = bundle["config"]
amount_high_cutoff = None
if reference_df is not None and "Amount" in reference_df.columns:
    amount_high_cutoff = float(reference_df["Amount"].quantile(0.99))
else:
    amount_high_cutoff = float("inf")


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="card"><div class="card-label">Adaptive threshold</div>'
            f'<div class="card-value">{cfg["ADAPTIVE_THRESHOLD"]:.1f}</div>'
            f'<div class="card-sub">{cfg["RISK_PERCENTILE"]}th pct. of legit validation scores</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="card"><div class="card-label">Score weighting</div>'
            f'<div class="card-value">{cfg["ISO_WEIGHT"]:.2f} / {cfg["AE_WEIGHT"]:.2f}</div>'
            f'<div class="card-sub">isolation forest / autoencoder</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="card"><div class="card-label">Feature count</div>'
            f'<div class="card-value">{len(cfg["FEATURE_COLUMNS"])}</div>'
            f'<div class="card-sub">Time, V1&ndash;V28, Amount</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        status = "loaded" if reference_df is not None else "not loaded"
        st.markdown(
            f'<div class="card"><div class="card-label">Reference dataset</div>'
            f'<div class="card-value" style="font-size:20px;">{status}</div>'
            f'<div class="card-sub">{len(reference_df) if reference_df is not None else 0} rows</div></div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Signature instrument &mdash; the risk dial</div>', unsafe_allow_html=True)
        demo_score = st.slider("Preview score", 0, 100, 42, label_visibility="collapsed")
        st.markdown(render_gauge(demo_score, cfg["ADAPTIVE_THRESHOLD"]), unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center;"><span class="badge {risk_badge_class(risk_category(demo_score))}">'
            f'{risk_category(demo_score)}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">How a score is built</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="signal-row">1&nbsp;&nbsp; Isolation Forest isolates the transaction &rarr; anomaly score (0&ndash;1)</div>
<div class="signal-row">2&nbsp;&nbsp; Autoencoder reconstructs the transaction &rarr; reconstruction error (0&ndash;1)</div>
<div class="signal-row">3&nbsp;&nbsp; Weighted fusion: 0.5&times;IF + 0.5&times;AE &rarr; Hybrid Risk Score (0&ndash;100)</div>
<div class="signal-row">4&nbsp;&nbsp; Score compared to the adaptive threshold, learned from legitimate validation data</div>
<div class="signal-row">5&nbsp;&nbsp; Flagged transactions receive a plain-language explanation</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if reference_df is not None and "Class" in reference_df.columns:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Reference dataset class balance</div>', unsafe_allow_html=True)
            counts = reference_df["Class"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
            st.bar_chart(counts, color="#5B8DEF")
            st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "The Fraud Risk Score is a weighted heuristic combining two anomaly-detection "
        "signals — it is not a statistically calibrated probability of fraud."
    )


# ---------------------------------------------------------------------------
# Page: Score a Transaction
# ---------------------------------------------------------------------------

elif page == "Score a Transaction":
    feature_cols = cfg["FEATURE_COLUMNS"]

    if "manual_row" not in st.session_state:
        if reference_df is not None:
            base = reference_df[reference_df.get("Class", 0) == 0][feature_cols].mean() \
                if "Class" in (reference_df.columns if reference_df is not None else []) \
                else reference_df[feature_cols].mean()
            st.session_state["manual_row"] = base.to_dict()
        else:
            st.session_state["manual_row"] = {c: 0.0 for c in feature_cols}

    controls_col, result_col = st.columns([1.1, 1])

    with controls_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Transaction input</div>', unsafe_allow_html=True)

        btn_cols = st.columns(3)
        with btn_cols[0]:
            random_disabled = reference_df is None
            if st.button("\U0001F3B2 Random sample", disabled=random_disabled, use_container_width=True):
                sampled = reference_df.sample(1, random_state=None).iloc[0]
                st.session_state["manual_row"] = {c: float(sampled[c]) for c in feature_cols}
                st.rerun()
        with btn_cols[1]:
            fraud_disabled = reference_df is None or "Class" not in (reference_df.columns if reference_df is not None else [])
            if st.button("\U0001F6A9 Known fraud sample", disabled=fraud_disabled, use_container_width=True):
                fraud_rows = reference_df[reference_df["Class"] == 1]
                if len(fraud_rows) > 0:
                    sampled = fraud_rows.sample(1).iloc[0]
                    st.session_state["manual_row"] = {c: float(sampled[c]) for c in feature_cols}
                    st.rerun()
        with btn_cols[2]:
            if st.button("\u21ba Reset to zero", use_container_width=True):
                st.session_state["manual_row"] = {c: 0.0 for c in feature_cols}
                st.rerun()

        if random_disabled:
            st.caption("Load a reference `creditcard.csv` in the sidebar to sample real transactions.")

        row_df = pd.DataFrame([st.session_state["manual_row"]])[feature_cols]
        edited = st.data_editor(
            row_df,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="editor",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    try:
        scored = score_dataframe(bundle, edited)
        row = scored.iloc[0]
        error = None
    except Exception as e:
        row, error = None, str(e)

    with result_col:
        st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
        if error:
            st.error(f"Could not score this transaction: {error}")
        else:
            st.markdown(render_gauge(row["hybrid_risk_score"], cfg["ADAPTIVE_THRESHOLD"]), unsafe_allow_html=True)
            st.markdown(
                f'<span class="badge {risk_badge_class(row["risk_level"])}">{row["risk_level"]}</span>'
                f'&nbsp;&nbsp;<span class="badge" style="background:rgba(91,141,239,0.15);'
                f'color:#5B8DEF;border:1px solid rgba(91,141,239,0.4);">{row["decision"]}</span>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        if not error:
            m1, m2 = st.columns(2)
            m1.metric("Isolation Forest score", f"{row['isolation_forest_score']:.3f}")
            m2.metric("Autoencoder score", f"{row['autoencoder_score']:.3f}")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Explanation</div>', unsafe_allow_html=True)
            for s in explain_row(row, amount_high_cutoff, cfg["ADAPTIVE_THRESHOLD"]):
                st.markdown(f'<div class="signal-row">&bull;&nbsp; {s}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Batch Scoring
# ---------------------------------------------------------------------------

elif page == "Batch Scoring":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Upload transactions</div>', unsafe_allow_html=True)
    st.caption(f"CSV must contain: {', '.join(cfg['FEATURE_COLUMNS'])}")
    batch_file = st.file_uploader("CSV file", type="csv", key="batch_upload")
    st.markdown("</div>", unsafe_allow_html=True)

    if batch_file is not None:
        raw_df = pd.read_csv(batch_file)
        try:
            scored_df = score_dataframe(bundle, raw_df)
        except Exception as e:
            st.error(f"Could not score this file: {e}")
            scored_df = None

        if scored_df is not None:
            total = len(scored_df)
            counts = scored_df["risk_level"].value_counts()
            flagged = int((scored_df["decision"] == "FRAUD").sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transactions scored", f"{total:,}")
            c2.metric("Flagged as fraud", f"{flagged:,}", f"{flagged/total*100:.2f}%")
            c3.metric("High risk", f"{int(counts.get('High Risk', 0)):,}")
            c4.metric("Avg. risk score", f"{scored_df['hybrid_risk_score'].mean():.1f}")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Risk level distribution</div>', unsafe_allow_html=True)
            dist = scored_df["risk_level"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
            st.bar_chart(dist, color="#F0A93D")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Highest-risk transactions</div>', unsafe_allow_html=True)
            top = scored_df.sort_values("hybrid_risk_score", ascending=False).head(50)
            st.dataframe(top, use_container_width=True, height=360)
            st.download_button(
                "Download full scored results (CSV)",
                scored_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_transactions.csv",
                mime="text/csv",
            )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Upload a CSV of transactions (Time, V1-V28, Amount) to score them in bulk.")


# ---------------------------------------------------------------------------
# Page: Model Comparison
# ---------------------------------------------------------------------------

elif page == "Model Comparison":
    if os.path.exists(comparison_csv_path):
        comp_df = pd.read_csv(comparison_csv_path)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Model comparison (from notebook Section 13)</div>', unsafe_allow_html=True)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        metric_cols = [c for c in ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"] if c in comp_df.columns]
        if metric_cols and "Model" in comp_df.columns:
            chart_df = comp_df.set_index("Model")[metric_cols]
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Metric comparison</div>', unsafe_allow_html=True)
            st.bar_chart(chart_df, color=["#35C482", "#5B8DEF", "#F0A93D", "#E5484D", "#8A94A8"][: len(metric_cols)])
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(
            f"`{comparison_csv_path}` not found. Re-run Section 13 of the notebook "
            f"(it writes `model_comparison.csv`), or point the sidebar at its location."
        )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">High-risk transactions (from notebook Section 17)</div>', unsafe_allow_html=True)
    if os.path.exists(high_risk_csv_path):
        hr_df = pd.read_csv(high_risk_csv_path)
        st.dataframe(hr_df, use_container_width=True, height=360)
    else:
        st.warning(
            f"`{high_risk_csv_path}` not found. Re-run Section 17 of the notebook "
            f"(it writes `high_risk_transactions.csv`), or point the sidebar at its location."
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: About & Limitations
# ---------------------------------------------------------------------------

else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Framework</div>', unsafe_allow_html=True)
    st.markdown(
        "A hybrid anomaly-based fraud risk scoring framework combining Isolation "
        "Forest anomaly scores and Autoencoder reconstruction errors, with adaptive "
        "risk thresholding and explainable transaction-level risk reporting. This is "
        "a project-level engineering framework, not a claim of a novel research "
        "algorithm — Isolation Forest and Autoencoders are established techniques."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        """
- Built on the anonymized Kaggle `mlg-ulb/creditcardfraud` dataset — `V1`&ndash;`V28` are PCA components with no recoverable meaning.
- No location, merchant category, customer ID, or device data exists in this dataset or this UI.
- Scoring here is single-transaction / batch-file, not a live streaming pipeline (e.g. Kafka).
- Real-world fraud labels are often delayed; this system is evaluated on a static, pre-labeled dataset.
- Fraud patterns drift over time; this model reflects the patterns present when it was trained.
- The Hybrid Fraud Risk Score is a weighted heuristic, not a statistically calibrated probability.
- Strong benchmark performance does not guarantee equivalent production performance.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
