import pytest
import joblib
from score import score
import requests
import subprocess
import time
import socket

@pytest.fixture
def model():
    return joblib.load('best_model.pkl')

@pytest.fixture
def vectorizer():
    return joblib.load('vectorizer.pkl')

def test_score(model, vectorizer):
    prediction, propensity = score("Test message", model, vectorizer, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)
    
    # Format test - check output types
    prediction, propensity = score("Another test", model, vectorizer, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)
    
    # Test propensity bounds
    _, propensity = score("Test message", model, vectorizer, 0.5)
    assert 0 <= propensity <= 1
    
    # Test threshold = 0 (should always predict spam)
    prediction, _ = score("Any message", model, vectorizer, 0)
    assert prediction == True
    
    # Test threshold = 1 (should always predict non-spam)
    prediction, _ = score("Any message", model, vectorizer, 1)
    assert prediction == False
    
    # Test obvious spam
    obvious_spam = """WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."""
    prediction, propensity = score(obvious_spam, model, vectorizer, 0.5)
    assert prediction == True
    assert propensity > 0.5
    
    # Test obvious non-spam
    obvious_ham = "Hi John, Can we meet tomorrow at 2pm to discuss the project?"
    prediction, propensity = score(obvious_ham, model, vectorizer, 0.5)
    assert prediction == False
    assert propensity < 0.5


def test_flask():
    try:
        process = subprocess.Popen(
            ["uv", "run", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for Flask to actually be ready
        time.sleep(20)
        
        response = requests.post(
            'http://localhost:5000/score',
            json={'text': 'Test message'}
        )
        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert 'propensity' in data
        assert isinstance(data['prediction'], bool)
        assert isinstance(data['propensity'], float)
        assert 0 <= data['propensity'] <= 1

        # Test obvious spam
        response = requests.post(
            'http://localhost:5000/score',
            json={'text': """WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."""}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['prediction'] == True
        assert data['propensity'] > 0.5

        # Test obvious ham
        response = requests.post(
            'http://localhost:5000/score',
            json={'text': 'Hi John, Can we meet tomorrow at 2pm?'}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['prediction'] == False
        assert data['propensity'] < 0.5

        # Test invalid request (missing text)
        response = requests.post(
            'http://localhost:5000/score',
            json={}
        )
        assert response.status_code == 400
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise
    finally:
        if 'process' in locals():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

def test_docker():
    try:
        # Build the Docker image
        subprocess.run(["docker", "build", "-t", "spam-classifier", "."], check=True)

        # Run the Docker container
        container = subprocess.Popen(
            ["docker", "run", "-p", "5000:5000", "spam-classifier"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for Flask to actually be ready
        time.sleep(20)
        
        # Test the Docker container
        response = requests.post(
            'http://localhost:5000/score',
            json={'text': 'Test message'}
        )
        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert 'propensity' in data
        assert isinstance(data['prediction'], bool)
        assert isinstance(data['propensity'], float)
        assert 0 <= data['propensity'] <= 1 
        
    except Exception as e:
        print(f"Docker test failed with error: {str(e)}")
        raise
    finally:
        if 'container' in locals():
            container.terminate()
            try:
                container.wait(timeout=5)
            except subprocess.TimeoutExpired:
                container.kill()
