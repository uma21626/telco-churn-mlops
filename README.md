# 📡 Telco Customer Churn Prediction — End-to-End MLOps Project

A production-ready Machine Learning system that predicts customer churn for telecom companies. Built with a full MLOps pipeline including data validation, feature engineering, model training, experiment tracking, REST API serving, Docker containerization, CI/CD, and AWS deployment.

---

## 🎯 Purpose

Build and ship a full machine-learning solution for predicting customer churn in a telecom setting — from data prep and modeling to an API + web UI deployed on AWS.

### Problem Solved & Benefits

- **Faster decisions** — Predicts which customers are likely to churn so teams can act before they leave.
- **Operationalized ML** — Model is accessible via a REST API and a simple UI; anyone can test it without notebooks.
- **Repeatable delivery** — CI/CD + containers mean every change can be rebuilt, tested, and redeployed in a consistent way.
- **Traceable experiments** — MLflow tracks runs, metrics, and artifacts for reproducibility and auditing.

---

## 🏗️ Project Architecture

```
telco-churn-mlops/
├── src/
│   ├── app/
│   │   ├── main.py              # FastAPI + Gradio serving app
│   │   └── app.py
│   ├── data/
│   │   ├── load_data.py         # Data loading
│   │   └── preprocess.py        # Data cleaning & encoding
│   ├── features/
│   │   └── build_features.py    # Feature engineering
│   ├── models/
│   │   ├── train.py             # Model training
│   │   ├── evaluate.py          # Model evaluation
│   │   └── tune.py              # Hyperparameter tuning
│   ├── serving/
│   │   └── model/
│   │       └── inference.py     # Inference pipeline
│   └── utils/
│       ├── utils.py             # Helper functions
│       └── validate_data.py     # Data validation
├── notebooks/                   # EDA & experimentation
├── scripts/                     # Pipeline test scripts
├── .github/workflows/           # CI/CD GitHub Actions
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| ML Model | XGBoost |
| Experiment Tracking | MLflow |
| Data Validation | Great Expectations |
| API Framework | FastAPI |
| Web UI | Gradio |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | AWS ECS Fargate + ALB |
| Observability | AWS CloudWatch |
| Language | Python 3.11 |

---

## 📦 Dataset

This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.

**Features include:**
- Customer demographics (gender, partner, dependents)
- Phone & internet services
- Contract type and billing information
- Tenure and monthly charges

**Target:** Whether the customer churned (`Yes` / `No`)

---

## ⚙️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/uma21626/telco-churn-mlops.git
cd telco-churn-mlops
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
# Step 1 - Validate data
python src/utils/validate_data.py

# Step 2 - Preprocess
python src/data/preprocess.py

# Step 3 - Build features
python src/features/build_features.py

# Step 4 - Tune hyperparameters (optional)
python src/models/tune.py

# Step 5 - Train model
python src/models/train.py

# Step 6 - Evaluate model
python src/models/evaluate.py

# Step 7 - Run inference
python src/serving/model/inference.py
```

### 5. Launch the app
```bash
python -m uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

| URL | What you'll see |
|-----|----------------|
| `http://localhost:8000` | `{"status": "ok"}` health check |
| `http://localhost:8000/ui` | Gradio prediction UI |
| `http://localhost:8000/docs` | Swagger API docs |

---

## 🐳 Run with Docker

### Build the image
```bash
docker build -t telco-churn-app .
```

### Run the container
```bash
docker run -p 8000:8000 telco-churn-app
```

---

## 🔌 API Usage

### Health Check
```http
GET http://localhost:8000
```
```json
{"status": "ok"}
```

### Predict Churn
```http
POST http://localhost:8000/predict
Content-Type: application/json
```

```json
{
  "gender": "Female",
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "tenure": 1,
  "MonthlyCharges": 85.0,
  "TotalCharges": 85.0
}
```

**Response:**
```json
{"prediction": "Likely to churn"}
```

---

## 📊 Model Performance

The XGBoost model was trained and tracked using MLflow.

| Metric | Score |
|--------|-------|
| F1 Score | 61.52% |
| ROC-AUC | 83.87% |
| Precision | 48.82% |
| Recall | 83.16% |

> Run `python src/models/evaluate.py` to see the full evaluation report.

### MLflow Experiment Tracking

```bash
mlflow ui
```
Open: `http://localhost:5000`

---

## 🚀 Deployment — AWS ECS Fargate

### Architecture

```
Internet
   │
   ▼
Application Load Balancer (HTTP:80)
   │
   ▼
Target Group (HTTP:8000)
   │
   ▼
ECS Fargate Task (Docker container)
   │
   ▼
CloudWatch Logs
```

### Deployment Flow

1. **Push to main** → GitHub Actions builds the Docker image and pushes to Docker Hub.
2. **ECS service** is updated (manually or via workflow) to force a new deployment.
3. **ALB health checks** hit `GET /` on port 8000; once healthy, traffic is routed to the new task.
4. **Users** call `POST /predict` or open the Gradio UI at `/ui` via the ALB DNS.

### Security Groups

| Component | Rule |
|-----------|------|
| ALB | Inbound HTTP:80 from `0.0.0.0/0` |
| ECS Task | Inbound HTTP:8000 from ALB Security Group only |
| Both | Outbound open |

---

## 🔄 CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/`):

1. On push to `main` → Build Docker image
2. Push image to Docker Hub
3. Optionally trigger ECS service update for zero-downtime redeploy

---

## 🔍 Key Design Decisions

- **Training/Serving Consistency** — The same feature transformations used during training are applied at inference time via `_serve_transform()` to prevent train/serve skew.
- **Docker-ready paths** — `inference.py` auto-detects whether it's running in Docker (`/app/model`) or locally (`mlruns/`) and adjusts paths accordingly.
- **MLflow tracking** — All experiments, parameters, and metrics are logged to MLflow for full reproducibility.
- **PYTHONPATH** — Set to `/app` in the Dockerfile so all modules import correctly inside the container.

---

## 🐛 Roadblocks & How We Solved Them

### Unhealthy targets behind ALB
- **Cause:** App didn't respond at the health-check path; listener/target port mismatches.
- **Fix:** Added `GET /` health endpoint; confirmed ALB listener on 80 forwards to TG on 8000; TG health check path set to `/`.

### ModuleNotFoundError in container
- **Cause:** Python path in the image didn't include `src/`.
- **Fix:** Set `PYTHONPATH=/app` in the Dockerfile; corrected uvicorn app path to `src.app.main:app`.

### ALB DNS timing out
- **Cause:** Security group rules not aligned with traffic flow.
- **Fix:** ALB SG allows inbound 80 from `0.0.0.0/0`; task SG allows inbound 8000 from the ALB SG only; outbound open.

### ECS redeploy not picking up new image
- **Cause:** Service still running previous task definition.
- **Fix:** Force new deployment (CLI or console) after pushing the new image; optional step added to CI.

### Gradio UI error ("No runs found in experiment")
- **Cause:** Inference expected an MLflow-logged model but couldn't resolve a run.
- **Fix:** Standardized MLflow experiment name and model logging in training; inference loads the logged model consistently.

### Local vs production path differences
- **Cause:** MLflow artifact URIs differ locally vs. in container.
- **Fix:** For local dev, load via `./mlruns/.../artifacts/model`; in prod, container loads the packaged model path used at build time.

### Windows symlinks breaking Docker build
- **Cause:** Files cloned as symlinks on Windows couldn't be copied by Docker.
- **Fix:** Replaced symlinks with real file copies using PowerShell before building.

### Windows-only packages in requirements.txt
- **Cause:** `pywin32` and `audioop-lts` are Windows-only and fail in Linux containers.
- **Fix:** Removed Windows-only packages from `requirements.txt` before building the Docker image.

---

## 👤 Author

**Uma** — [github.com/uma21626](https://github.com/uma21626)

---

## 📄 License

This project is for educational purposes. Dataset credit: [IBM/Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
