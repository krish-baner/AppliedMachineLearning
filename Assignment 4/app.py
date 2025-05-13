from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/score', methods=['POST'])
def score_endpoint():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        prediction, propensity = score(text, model, vectorizer, threshold=0.5)
        return jsonify({
            'prediction': bool(prediction),
            'propensity': float(propensity)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)