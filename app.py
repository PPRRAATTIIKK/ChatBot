from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Improved keyword-based medical chatbot
def get_medical_response(user_input):
    text = user_input.lower()
    
    if any(word in text for word in ['headache', 'migraine', 'head pain']):
        return "Headaches can be caused by stress, dehydration, or lack of sleep. Drink water, rest, and consider paracetamol. If severe or persistent, consult a doctor."
    
    elif any(word in text for word in ['fever', 'temperature', 'hot']):
        return "You likely have a fever. Rest, drink plenty of fluids, and take paracetamol if needed. Monitor your temperature. Seek medical help if it exceeds 39.5°C or lasts more than 3 days."
    
    elif any(word in text for word in ['stomach', 'pain', 'acidity', 'gas', 'vomiting']):
        return "Stomach pain can be due to acidity or indigestion. Try antacids or ginger tea. Avoid spicy/oily food. If pain is severe or you have vomiting with blood, see a doctor immediately."
    
    elif any(word in text for word in ['cough', 'cold', 'sore throat', 'running nose']):
        return "You seem to have a common cold. Rest, drink warm fluids, and take steam. Use honey for cough. If symptoms last more than 7 days or you have high fever, consult a physician."
    
    elif any(word in text for word in ['hello', 'hi', 'hey', 'namaste']):
        return "Hello! I'm NEXUS, your medical AI assistant. How are you feeling today? Please describe your symptoms."
    
    else:
        return "I'm here to help with medical queries. Please describe your symptoms clearly (e.g., headache, fever, stomach pain, cough, etc.)."

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        
        if not message:
            return jsonify({"response": "Please type a message!"})
        
        response = get_medical_response(message)
        
        return jsonify({
            "response": response,
            "confidence": 85.0
        })
        
    except Exception as e:
        return jsonify({"response": "Sorry, I encountered an error. Please try again."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)from flask import Flask, request, jsonify, send_from_directory
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

# Load model
model = None
tokenizer = None
le = None
responses = {}

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = load_model(os.path.join(base_dir, 'app/model.h5'), compile=False)
    tokenizer = joblib.load(os.path.join(base_dir, 'app/tokenizer.joblib'))
    le = joblib.load(os.path.join(base_dir, 'app/label_encoder.joblib'))
    
    with open(os.path.join(base_dir, 'app/responses.json'), 'r') as f:
        responses = json.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Model loading failed:", str(e))

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
        return jsonify({"response": "Model is not loaded properly."})

    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"response": "Please type a message!"})

        cleaned = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=20)

        prediction = model.predict(padded, verbose=0)
        tag_index = np.argmax(prediction)
        tag = le.inverse_transform([tag_index])[0]

        response_list = responses.get(tag, ["I'm sorry, I don't understand your query."])
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