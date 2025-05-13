<!-- Badges -->
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![DVC Status](https://img.shields.io/badge/dvc-pipelines-green)](https://dvc.org/)
[![MLflow Runs](https://img.shields.io/badge/mlflow-tracked-orange)](https://mlflow.org/)
[![Build & Test](https://img.shields.io/github/actions/workflow/status/your-username/aml-course/ci.yml?branch=main)](https://github.com/your-username/aml-course/actions)
[![Coverage](https://img.shields.io/badge/coverage-⛰️-yellow)](https://coveralls.io/github/your-username/aml-course)

# 📘 Applied Machine Learning

Welcome to the **Applied Machine Learning** coursework repository!  
Here you’ll find four progressively advanced assignments that take you from building a simple prototype to deploying and continuously integrating an ML service. We leverage production-grade tools like DVC, MLflow, Flask, Docker, and Git hooks to mirror real-world ML lifecycle practices.

---

## 📦 Assignment 1: Prototype

**Objective:** Kick off with a minimal viable product—classify SMS as “spam” or “ham.”

- **Data Ingestion & Cleaning**  
  Retrieved the [UCI SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection), performed text normalization (lowercasing, punctuation removal), and handled missing values gracefully.  
- **Dataset Splitting**  
  Split into **train** / **validation** / **test** sets (80/10/10) and exported each to CSV with clear naming conventions (`train.csv`, `val.csv`, `test.csv`).  
- **Baseline Modeling**  
  Trained simple classifiers (Multinomial NB, Logistic Regression) and visualized performance metrics—accuracy, precision, recall, F1-score, and Area Under the Precision-Recall Curve—to pick a starting point.  
- **Model Selection**  
  Compared models via validation curves, then shortlisted the best performer based on balanced precision and recall.

---

## 🔍 Assignment 2: Experiment Tracking

**Objective:** Introduce rigorous versioning and experiment tracking to your workflow.

- **Data Versioning with DVC**  
  Initialized a DVC pipeline to snapshot **raw** and **processed** data at each stage. Ensured reproducibility by storing data checksums and pipeline stages in Git.  
- **Target Distribution Checks**  
  Automated statistical checks to compare label distributions across different random seed splits, guarding against data drift.  
- **MLflow Integration**  
  Logged every experiment—hyperparameters, code versions, metrics, and artifacts.  
- **Benchmark Suite**  
  Registered three distinct models (e.g., TF-IDF + SVM, Word2Vec + Logistic Regression, and Character-level CNN) to the MLflow registry and compared their AUCPR and inference latency in a unified dashboard.

---

## 🧪 Assignment 3: Testing & Model Serving

**Objective:** Ensure code quality with tests, then expose your model via an API.

- **Prediction Function**  
  Designed a `score(text: str) → dict` function that returns both the binary class and a spam probability score.  
- **Unit Testing with pytest**  
  Covered edge cases (empty strings, very long messages, non-ASCII text), type checking, and error handling. Achieved >90% code coverage.  
- **Flask Microservice**  
  Built a lightweight Flask app with a `/predict` endpoint. Input: JSON payload `{"text": "Hello!"}`; Output: `{"prediction": "ham", "probability": 0.03}`.  
- **API Integration Tests**  
  Employed `pytest` and `requests` to simulate client calls, verify HTTP status codes, and assert response schemas. Generated coverage and report badges.

---

## 🐳 Assignment 4: Containerization & CI/CD

**Objective:** Package your service in Docker and automate quality checks on every commit.

- **Docker Container**  
  Authored a multi-stage `Dockerfile` to build a lean image. Ensured dependencies (Flask, scikit-learn, joblib) are isolated and cached for fast rebuilds.  
- **Smoke Tests**  
  Wrote a Bash/Python script (`test_docker.sh`) that builds the image, spins up a container, hits the `/predict` endpoint with sample data, verifies output format, and tears down cleanly.  
- **Git Hooks & CI**  
  Configured a **pre-commit** hook to run `test_docker.sh` locally. Integrated the same script into a GitHub Actions workflow to gate PRs—failing builds will block merges if tests or linting rules fail.

---

## 🛠️ Tools & Technologies

- **Data Science & ML**: `scikit-learn` · `pandas` · `numpy` · `joblib`  
- **Versioning & Tracking**: `DVC` · `MLflow`  
- **API & Testing**: `Flask` · `pytest` · `requests`  
- **DevOps**: `Docker` · `pre-commit` · **GitHub Actions**

---

## 🚀 Getting Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/aml-course.git
   cd aml-course
