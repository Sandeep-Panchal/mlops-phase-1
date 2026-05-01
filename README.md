# 🚀 MLOps Phase 1: Sentiment Analysis Pipeline

This project demonstrates a **complete end-to-end MLOps pipeline (Phase 1)** for a Sentiment Analysis model using:

* MLflow (Experiment Tracking + Model Registry)
* FastAPI (Model Serving)
* Docker (Containerization)
* GitHub Actions (CI)

---

## 🧠 Architecture Overview

```text
Local / Docker Run
   ↓
Train Model + Log to MLflow
   ↓
Model Registered in MLflow
   ↓
Auto Promotion (Staging → Production)
   ↓
FastAPI loads Production Model
```

### CI Flow (GitHub Actions)

```text
Pull Request / Push
   ↓
CI Pipeline
   ↓
Code Validation + Training (No MLflow)
   ↓
Docker Image Build
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
│   ├── train.py
│   ├── predict.py
│   ├── registry.py
│   └── config.py
│
├── app/
│   └── main.py
│
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci.yaml
└── README.md
```

---

## ⚙️ Features

### ✅ Experiment Tracking

* Logs metrics and models using MLflow

### ✅ Model Registry

* Versioned models with lifecycle:

  ```text
  None → Staging → Production
  ```

### ✅ Auto Promotion Logic

* Promotes model to Production if:

  * Accuracy improves over current production
  * OR no production model exists

### ✅ API Deployment

FastAPI serves model from:

```text
models:/SentimentModel/Production
```

### ✅ CI Pipeline

* Triggered on PR / push
* Validates code
* Runs training in safe mode (no MLflow)
* Builds Docker image

---

## 🧪 Dataset

```csv
text,sentiment
"I love this movie",1
"Worst experience ever",-1
```

---

## 🌍 Environment Configuration

The pipeline behavior depends on:

```text
ENV = local | docker | ci
```

| ENV    | Behavior                  |
| ------ | ------------------------- |
| local  | Full MLflow + registry    |
| docker | Full MLflow (via host)    |
| ci     | Training only (no MLflow) |

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

### 3️⃣ Train Model (Local)

```powershell
$env:ENV="local"
python src/train.py
```

This will:

* Train model
* Log metrics
* Register model
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
docker run -e ENV=docker -p 8000:8000 sentiment-app
```

---

## 🔁 CI/CD (GitHub Actions)

### CI (Current Phase)

```text
PR / Push
   ↓
Install dependencies
   ↓
Run training (CI mode)
   ↓
Build Docker image
```

⚠️ Note:

* MLflow is NOT used in CI (Phase 1)
* Model registry happens locally or via Docker

---

## 🧠 Key Concepts Implemented

### 🔹 Model Registry-based Deployment

API always serves:

```text
Production model
```

---

### 🔹 Automatic Model Promotion

* Based on model performance
* Fully automated

---

### 🔹 Versioned Models

```text
v1 → Production
v2 → Staging
v3 → Production (if better)
```

---

### 🔹 Rollback Ready

* Change model stage in MLflow UI

---

## ⚠️ Limitations (Phase 1)

* CI does NOT perform real model registry
* No DVC (data/pipeline versioning)
* No cloud deployment
* MLflow is local (not remote)
* No monitoring

---

## 🚀 Next Steps (Phase 2)

* Run MLflow inside CI (true CD)
* Add DVC
* Store artifacts in S3
* Deploy on EC2
* Add load balancer
* Add monitoring

---

## 🧠 Learnings

* Experiment vs Model Registry separation
* CI vs CD in MLOps
* Model lifecycle management
* Containerized ML deployment

---

## 📌 Author

Built as part of MLOps learning journey 🚀