<<<<<<< HEAD
#  Telco Customer Churn Prediction — MLOps Project

A production-ready Machine Learning system that predicts customer churn for telecom companies. Built with a full MLOps pipeline including data validation, feature engineering, model training, experiment tracking, REST API serving, and Docker containerization.

---

##  Live Demo

| Interface | URL |
|-----------|-----|
| Gradio Web UI | `http://localhost:8000/ui` |
| REST API | `http://localhost:8000/predict` |
| API Docs (Swagger) | `http://localhost:8000/docs` |

---

##  Project Architecture

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
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
└── README.md
```

---

##  Tech Stack

| Category | Tools |
|----------|-------|
| ML Model | XGBoost |
| Experiment Tracking | MLflow |
| Data Validation | Great Expectations |
| API Framework | FastAPI |
| Web UI | Gradio |
| Containerization | Docker |
| Language | Python 3.11 |

---

##  Dataset

This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.

**Features include:**
- Customer demographics (gender, partner, dependents)
- Phone & internet services
- Contract type and billing information
- Tenure and monthly charges

**Target:** Whether the customer churned (`Yes` / `No`)

---

##  Run Locally

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

### 4. Run the pipeline

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

---

##  Run with Docker

### Build the image
```bash
docker build -t telco-churn-app .
```

### Run the container
```bash
docker run -p 8000:8000 telco-churn-app
```

### Open in browser
```
http://localhost:8000/ui    → Gradio Web UI
http://localhost:8000/docs  → Swagger API Docs
```

---

##  API Usage

### Health Check
```bash
GET http://localhost:8000
```
**Response:**
```json
{"status": "ok"}
```

### Predict Churn
```bash
POST http://localhost:8000/predict
Content-Type: application/json
```

**Request Body:**
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

##  Model Performance

The XGBoost model was trained and tracked using MLflow.

| Metric | Score |
|--------|-------|
| Accuracy | ~82% |
| ROC-AUC | ~85% |
| Precision | ~80% |
| Recall | ~78% |

> Run `python src/models/evaluate.py` to see the full evaluation report.

---

##  MLflow Experiment Tracking

Start the MLflow UI to view experiment runs:
```bash
mlflow ui
```
Then open: `http://localhost:5000`

---

##  Key Design Decisions

- **Training/Serving Consistency**: The same feature transformations used during training are applied at inference time via `_serve_transform()` to prevent train/serve skew.
- **Docker-ready paths**: `inference.py` auto-detects whether it's running in Docker (`/app/model`) or locally (`mlruns/`) and adjusts paths accordingly.
- **MLflow tracking**: All experiments, parameters, and metrics are logged to MLflow for full reproducibility.

---

##  Author

**Uma** — [github.com/uma21626](https://github.com/uma21626)

---

##  License

This project is for educational purposes. Dataset credit: [IBM/Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
=======

>>>>>>> 989fad7ded05ec91e490b5df03ab0075305249c1
