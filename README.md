# Fraud Risk Console

A Streamlit UI for the hybrid Isolation Forest + Autoencoder fraud risk scoring
system built in `financial_transaction_fraud_detection.ipynb`. It loads your
**actual trained models** — nothing here is mocked.

## 1. Run the notebook first

This app is a UI on top of the notebook's output, not a replacement for it. Run
`financial_transaction_fraud_detection.ipynb` from top to bottom at least once. That
produces a `saved_models/` folder next to the notebook containing:

```
saved_models/isolation_forest.pkl
saved_models/autoencoder.keras
saved_models/scaler.pkl
saved_models/risk_configuration.pkl
```

It also writes `model_comparison.csv` and `high_risk_transactions.csv` in the same
directory — the app's "Model Comparison" page will pick those up automatically if
present.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Put the app next to your artifacts

Copy `app.py` (and `requirements.txt`) into the same folder that contains
`saved_models/` (i.e. wherever you ran the notebook), or just note the path — you can
point the app at any location from its sidebar.

## 4. Run it

```bash
streamlit run app.py
```

It opens at `http://localhost:8501`.

## 5. Point it at your files

In the sidebar:

- **saved_models folder** — defaults to `saved_models` (relative to where you launch
  Streamlit). Change it if your folder lives elsewhere.
- **creditcard.csv path** (optional) — lets you sample real transactions on the
  "Score a Transaction" page instead of typing 30 numbers by hand, and unlocks the
  class-balance chart on the Overview page. The app also auto-searches
  `~/.cache/kagglehub/` for a copy `kagglehub` already downloaded. You can also just
  upload the CSV directly instead of typing a path.
- **model_comparison.csv / high_risk_transactions.csv** (optional) — feeds the Model
  Comparison page.

## What's in the app

- **Overview** — status of loaded artifacts, the adaptive threshold, and the
  signature risk dial.
- **Score a Transaction** — edit a single transaction's fields directly (or pull a
  random / known-fraud sample from your reference dataset), see its Fraud Risk Score
  on the dial, and read the plain-language explanation.
- **Batch Scoring** — upload a CSV of transactions and score all of them at once;
  view the risk-level breakdown, the highest-risk rows, and download the full scored
  file.
- **Model Comparison** — the Logistic Regression / Isolation Forest / Autoencoder /
  Hybrid comparison table and chart from the notebook, plus the saved high-risk
  transaction table.
- **About & Limitations** — the same limitations documented in the notebook (no
  location/merchant/device data, not a calibrated probability, static dataset, etc.),
  kept in the UI so nobody mistakes the score for more than it is.
