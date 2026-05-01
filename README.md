# 🚀 MLOps Phase 1: Sentiment Analysis Pipeline

This project demonstrates a **complete end-to-end MLOps pipeline (Phase 1)** for a Sentiment Analysis model using:

* MLflow (Experiment Tracking + Model Registry)
* FastAPI (Model Serving)
* Docker (Containerization)
* GitHub Actions (CI)

---

## 🧠 Architecture Overview

```text
GitHub Push
   ↓
CI (GitHub Actions)
   ↓
Train Model + Log to MLflow
   ↓
Model Registered in MLflow
   ↓
Auto Promotion (Staging → Production)
   ↓
FastAPI loads Production Model
```

---

## 📁 Project Structure

```bash
mlops-phase1/
│
├── data/
│   └── binary_class.csv
│
├── src/
│   ├── train.py          # Training + MLflow logging
│   ├── predict.py        # Load model from registry
│   |── registry.py       # Promotion logic
│   └── config.py         # Configurations (MLflow URI, model name, etc.)
├── app/
│   └── main.py           # FastAPI app
│
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci.yaml
└── README.md
```

---

## ⚙️ Features

### ✅ Experiment Tracking

* Logs parameters, metrics, and models using MLflow

### ✅ Model Registry

* Registers model versions
* Maintains lifecycle:

  * None → Staging → Production

### ✅ Auto Promotion Logic

* Model is promoted to **Production** if:

  * Accuracy improves over current production
  * OR passes threshold

### ✅ API Deployment

* FastAPI serves model from:

```text
models:/SentimentModel/Production
```

### ✅ CI Pipeline

* Triggered on GitHub push
* Trains model
* Registers model
* Builds Docker image

---

## 🧪 Dataset

Simple sentiment dataset:

```csv
text,sentiment
"I love this movie",1
"Worst experience ever",-1
```

---

## 🚀 Setup Instructions

---

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Start MLflow Server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

Open:

```
http://localhost:5000
```

---

### 3️⃣ Train Model

```bash
python src/train.py
```

This will:

* Train model
* Log metrics
* Register model
* Move to Staging
* Promote to Production (if better)

---

### 4️⃣ Run FastAPI

```bash
uvicorn app.main:app --reload
```

Open:

```
http://localhost:8000/docs
```

---

## 🐳 Docker Setup

### Build Image

```bash
docker build -t sentiment-app .
```

---

### Run Container

```bash
docker run -p 8000:8000 sentiment-app
```

---

### ⚠️ Important (MLflow inside Docker)

Update `config.py`:

```python
MLFLOW_TRACKING_URI = "http://host.docker.internal:5000"
```

---

## 🔁 CI/CD (GitHub Actions)

Pipeline runs on push:

```yaml
- Install dependencies
- Train model
- Register model
- Build Docker image
```

---

## 🧠 Key Concepts Implemented

### 🔹 Model Registry-based Deployment

API always loads:

```text
Production model
```

---

### 🔹 Automatic Model Promotion

* Based on performance
* No manual intervention

---

### 🔹 Versioned Models

```text
v1 → Production
v2 → Staging
v3 → Production (if better)
```

---

### 🔹 Rollback Ready

Just change stage in MLflow UI or via code

---

## ⚠️ Limitations (Phase 1)

* No data and pipeline versioning (DVC not included yet)
* No cloud deployment (EC2/S3)
* Model loads at runtime (basic approach)
* No monitoring

---

## 🚀 Next Steps (Phase 2)

* Add DVC for data & pipeline versioning
* Store artifacts in S3
* Deploy on EC2
* Add load balancer & auto-scaling
* Add monitoring (Prometheus/Grafana)

---

## 🧠 Learnings

* Difference between Experiments vs Model Registry
* CI vs CD in MLOps
* Model lifecycle management
* Containerized ML deployment

---

## 📌 Author

Built as part of MLOps learning journey 🚀
