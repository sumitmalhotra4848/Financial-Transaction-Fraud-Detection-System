# Financial Transaction Fraud Detection

FastAPI backend for the fraud-detection model from the supplied notebook.

## Model artifacts

Put these files inside `saved_models/`:

```text
saved_models/
├── isolation_forest.pkl
├── autoencoder.keras
├── scaler.pkl
└── risk_configuration.pkl
```

The notebook already contains the model-saving step for these artifacts.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

The interactive Swagger UI lets you test every endpoint.

## Endpoints

### Health

```http
GET /health
```

### Model information

```http
GET /model-info
```

### Single prediction

```http
POST /predict
```

Example JSON:

```json
{
  "Time": 1000,
  "Amount": 250.50,
  "V1": 0,
  "V2": 0,
  "V3": 0,
  "V4": 0,
  "V5": 0,
  "V6": 0,
  "V7": 0,
  "V8": 0,
  "V9": 0,
  "V10": 0,
  "V11": 0,
  "V12": 0,
  "V13": 0,
  "V14": 0,
  "V15": 0,
  "V16": 0,
  "V17": 0,
  "V18": 0,
  "V19": 0,
  "V20": 0,
  "V21": 0,
  "V22": 0,
  "V23": 0,
  "V24": 0,
  "V25": 0,
  "V26": 0,
  "V27": 0,
  "V28": 0
}
```

### Batch JSON

```http
POST /predict/batch
```

Send an array of transactions.

### CSV

```http
POST /predict/csv
```

Upload a CSV containing:

```text
Time,V1,V2,...,V28,Amount
```

A `Class` column may also exist and is ignored for inference.

## Deployment on Render

The included `render.yaml` configures:

```bash
pip install -r requirements.txt
```

and:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

After deployment, FastAPI's documentation will be available at:

```text
https://YOUR-RENDER-DOMAIN/docs
```

## Inference logic

The API follows the notebook's inference flow:

1. Select `Time`, `V1`–`V28`, and `Amount`.
2. Scale only `Time` and `Amount`.
3. Calculate the Isolation Forest anomaly score.
4. Calculate Autoencoder reconstruction error.
5. Normalize both scores.
6. Combine them with the saved ISO/AE weights.
7. Convert the hybrid score to a 0–100 risk score.
8. Assign Low/Medium/High Risk.
9. Compare the score with the saved adaptive threshold.
10. Return `FRAUD` or `LEGITIMATE`.

The API does not retrain the models during prediction.
