import sklearn
from typing import Tuple

def score(text: str, 
         model: sklearn.base.BaseEstimator,
         vectorizer: sklearn.feature_extraction.text.CountVectorizer,  # or TfidfVectorizer
         threshold: float) -> Tuple[bool, float]:
    X = vectorizer.transform([text])
    
    propensity = model.predict_proba(X)[0][1]    
    prediction = bool(propensity >= threshold)
    
    return prediction, propensity 