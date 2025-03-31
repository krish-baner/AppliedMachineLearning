# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 18:43:56 2025

@author: KRISHANU
"""

# score.py
from typing import Tuple
import numpy as np
import sklearn.base  # for type hinting (if desired)
import joblib


def score(text, model, threshold=0.5):
    # Get the probability of the positive class (assumed at index 1)
    pred_proba = model.predict_proba([text])[0, 1]

    # Validate inputs
    assert isinstance(text, str)
    assert (isinstance(threshold, float) or isinstance(threshold, int)) and (0 <= threshold <= 1)

    # Return boolean prediction
    prediction = pred_proba >= threshold
    return prediction, pred_proba
