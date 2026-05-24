from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os

app = Flask(__name__)
CORS(app)

# Load model and tools
model = load_model('app/model.h5')
tokenizer = joblib.load('app/tokenizer.joblib')
le = joblib.load('app/label_encoder.joblib')

with open('app/responses.json', 'r') as f:
    responses = json.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route("/")
def home():
    return jsonify({
        "status": "Medical Chatbot API is running",
        "model_loaded": True
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"response": "Please type something!"})

        # Preprocess and predict
        cleaned = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=20)   # Change 20 if your model uses different maxlen

        prediction = model.predict(padded, verbose=0)
        tag_index = np.argmax(prediction)
        tag = le.inverse_transform([tag_index])[0]

        response = responses.get(tag, ["I'm sorry, I don't understand."])[0]

        return jsonify({
            "response": response,
            "tag": tag,
            "confidence": float(prediction[0][tag_index])
        })

    except Exception as e:
        return jsonify({"response": "Sorry, something went wrong. Please try again."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)