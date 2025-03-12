import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, precision_score, 
    recall_score, roc_auc_score, precision_recall_curve, auc
)


# Set MLflow tracking URI (optional - use if you want to store experiments in a specific location)
# mlflow.set_tracking_uri("file:///path/to/mlruns")

def load_data(data_path):
    """Load data from the specified path."""
    data = pd.read_csv(data_path, sep='\t')
    X = data['preprocessed_message'].fillna('')
    y = data['label']
    return X, y

def preprocess_data(X_train, X_val=None, X_test=None):
    """Convert text data to TF-IDF features."""
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    result = {'tfidf_vectorizer': tfidf, 'X_train_tfidf': X_train_tfidf}
    
    if X_val is not None:
        result['X_val_tfidf'] = tfidf.transform(X_val)
    
    if X_test is not None:
        result['X_test_tfidf'] = tfidf.transform(X_test)
    
    return result

def fit_model(model, X_train, y_train):
    """Fit a model on training data."""
    model.fit(X_train, y_train)
    return model

def score_model(model, X):
    """Score a model on given data."""
    return model.predict(X)

def evaluate_model(y_true, y_pred, model=None, X=None):
    """Evaluate model predictions and return metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    # Calculate precision-recall AUC if model and X are provided
    pr_auc = None
    if model is not None and X is not None:
        if hasattr(model, "decision_function"):
            y_scores = model.decision_function(X)
        elif hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X)[:, 1]
        else:
            y_scores = y_pred  # Fall back to binary predictions
            
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'specificity': specificity,
        'roc_auc': roc_auc_score(y_true, y_pred)
    }
    
    if pr_auc is not None:
        metrics['pr_auc'] = pr_auc
    
    return metrics

def validate_model(model, X_train, y_train, X_val, y_val):
    """Validate a model by fitting on train and evaluating on train and validation."""
    # Fit on train
    model = fit_model(model, X_train, y_train)
    
    # Score on train and validation
    y_train_pred = score_model(model, X_train)
    y_val_pred = score_model(model, X_val)
    
    # Evaluate on train and validation
    train_metrics = evaluate_model(y_train, y_train_pred, model, X_train)
    val_metrics = evaluate_model(y_val, y_val_pred, model, X_val)
    
    return model, train_metrics, val_metrics

def run_experiment(model_name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Run an experiment with MLflow tracking."""
    with mlflow.start_run(run_name=model_name):
        # Log model parameters
        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)
        
        # Validate model
        model, train_metrics, val_metrics = validate_model(
            model, X_train, y_train, X_val, y_val
        )
        
        # Log training metrics
        for metric_name, metric_value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", metric_value)
        
        # Log validation metrics
        for metric_name, metric_value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)
        
        # Score on test data
        y_test_pred = score_model(model, X_test)
        test_metrics = evaluate_model(y_test, y_test_pred, model, X_test)
        
        # Log test metrics
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        # Create an input example for model signature
        input_example = X_train[:5]  # Use first 5 samples as example
        
        # Log the model with signature
        mlflow.sklearn.log_model(
            model, 
            model_name,
            input_example=input_example
        )
        
        return model, test_metrics
