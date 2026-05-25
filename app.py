from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import re
import os
import random

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)['intents']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def get_response(message):
    text = clean_text(message)
    
    for intent in intents:
        for pattern in intent['patterns']:
            if any(word in text for word in pattern.lower().split()):
                return random.choice(intent['responses'])
    
    return "I'm sorry, I didn't understand that. Can you please rephrase your symptoms or question?"

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({"response": "Please type a message!"})
    
    response = get_response(message)
    
    return jsonify({
        "response": response,
        "confidence": 75.0
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)