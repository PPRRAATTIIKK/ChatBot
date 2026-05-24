from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

model = None
tokenizer = None
le = None
responses = {}
load_error = "No error"

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'app/model.h5')
    
    model = load_model(model_path, compile=False)
    tokenizer = joblib.load(os.path.join(base_dir, 'app/tokenizer.joblib'))
    le = joblib.load(os.path.join(base_dir, 'app/label_encoder.joblib'))
    
    with open(os.path.join(base_dir, 'app/responses.json'), 'r') as f:
        responses = json.load(f)
        
    print("✅ SUCCESS: Model loaded!")
except Exception as e:
    load_error = str(e)
    print("❌ FAILED:", load_error)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route("/chat", methods=["POST"])
def chat():
    if not model:
        return jsonify({
            "response": "Model is not loaded properly.",
            "debug": load_error
        })

    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"response": "Please describe your symptoms."})

        cleaned = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=20)

        prediction = model.predict(padded, verbose=0)
        tag_index = np.argmax(prediction)
        tag = le.inverse_transform([tag_index])[0]

        response_list = responses.get(tag, ["I'm sorry, I don't understand your symptoms. Please describe them more clearly."])
        response = response_list[0] if isinstance(response_list, list) else str(response_list)

        return jsonify({
            "response": response,
            "tag": tag,
            "confidence": float(prediction[0][tag_index])
        })

    except Exception as e:
        return jsonify({"response": "Sorry, I had an error processing your request."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)