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


## 🧠 Assignment 5: Transfer Learning – Vision & Language

**Objective:** Leverage pre-trained deep learning models to solve computer vision and natural language understanding tasks with minimal data and training time.

---

### 🐤 Vision Task: Duck vs. Chicken Classifier

- **Dataset Preparation**  
  Collected ~100 images each for **ducks** and **chickens** from web sources. Organized them into `train`, `val`, and `test` folders following PyTorch's expected directory structure. Used data augmentation to improve generalization.
  
- **Model & Training**  
  Used a **pre-trained CNN model** (ResNet18 / MobileNetV2) from `torchvision.models`. Replaced the final fully connected layer to adapt it to binary classification (duck vs. chicken). Fine-tuned the model using transfer learning techniques on Google Colab.

- **Evaluation**  
  Evaluated model predictions on the test set using a **classification report** with metrics like precision, recall, and F1-score.
---

### 📝 NLP Task: Sentiment Classification Using Transformers

- **Dataset**  
  Downloaded the [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset), which contains text samples labeled as **positive**, **neutral**, or **negative**.

- **Model & Approach**  
  Leveraged a **pre-trained Transformer model** (like `distilBERT`, `BERT-base-uncased`) from HuggingFace's `transformers` library. Fine-tuned the model on the labeled dataset using a Colab notebook with GPU acceleration.

- **Output**  
  After training, evaluated the performance with a **classification report** showing precision, recall, and F1-score across all three sentiment classes.

---


## 🛠️ Tools & Technologies

- **Data Science & ML**: `scikit-learn` · `pandas` · `numpy` · `joblib`  
- **Versioning & Tracking**: `DVC` · `MLflow`  
- **API & Testing**: `Flask` · `pytest` · `requests`  
- **DevOps**: `Docker` · `pre-commit` · **GitHub Actions**

---

## 📚 Additional Resources

- 📦 HuggingFace Transformers: [https://huggingface.co/transformers](https://huggingface.co/transformers)  
- 🔗 PyTorch Docs: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)  
- 🎓 Google Colab: [https://colab.research.google.com](https://colab.research.google.com)  
- 🧰 scikit-learn Metrics: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)

---


