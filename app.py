from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Expanded medical keyword responses
def get_response(user_input):
    text = user_input.lower()
    
    if any(k in text for k in ['headache', 'migraine', 'head pain', 'head hurts']):
        return "Headaches are often caused by stress, dehydration, or lack of sleep. Drink plenty of water, rest in a dark room, and consider paracetamol. If the pain is severe or lasts more than 2 days, please consult a doctor."
    
    elif any(k in text for k in ['fever', 'temperature', 'hot', 'chills']):
        return "You appear to have a fever. Rest, stay hydrated, and monitor your temperature. Take paracetamol if needed. Seek medical attention if fever exceeds 39.5°C or lasts more than 3 days."
    
    elif any(k in text for k in ['stomach', 'acidity', 'gas', 'pain in stomach', 'vomiting']):
        return "Stomach pain can be due to acidity or indigestion. Avoid spicy and oily food. Try antacids or ginger tea. If you have severe pain, vomiting with blood, or it persists, see a doctor immediately."
    
    elif any(k in text for k in ['cough', 'cold', 'sore throat', 'running nose', 'sneezing']):
        return "You seem to have a common cold or cough. Take steam, drink warm fluids, and rest. Honey with ginger can help with cough. If symptoms last more than a week or you have high fever, consult a physician."
    
    elif any(k in text for k in ['hello', 'hi', 'hey', 'namaste', 'greetings']):
        return "Hello! I am NEXUS, your medical AI assistant. How are you feeling today? Please describe your symptoms."
    
    else:
        return "I'm here to help with common medical queries. Please describe your symptoms clearly (headache, fever, stomach pain, cough, etc.)."

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
        
        response = get_response(message)
        
        return jsonify({
            "response": response,
            "confidence": 75.0
        })
        
    except Exception as e:
        return jsonify({"response": "Sorry, I encountered an error. Please try again later."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)