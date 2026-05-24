from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import json
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

print("Current Directory:", os.getcwd())
print("Files in root:", os.listdir('.'))
if os.path.exists('app'):
    print("Files in app folder:", os.listdir('app'))
else:
    print("ERROR: 'app' folder not found!")

model = None
tokenizer = None
le = None
responses = {}
load_error = "No error yet"

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("Base directory:", base_dir)
    
    model_path = os.path.join(base_dir, 'app/model.h5')
    print("Looking for model at:", model_path)
    print("Model file exists:", os.path.exists(model_path))
    
    model = load_model(model_path, compile=False)
    tokenizer = joblib.load(os.path.join(base_dir, 'app/tokenizer.joblib'))
    le = joblib.load(os.path.join(base_dir, 'app/label_encoder.joblib'))
    
    with open(os.path.join(base_dir, 'app/responses.json'), 'r') as f:
        responses = json.load(f)
        
    print("✅ SUCCESS: Model loaded!")
except Exception as e:
    load_error = str(e)
    print("❌ FAILED to load model:", load_error)

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

    # ... rest of your chat function (we'll add later)

    return jsonify({"response": "Model is working but chat logic not yet connected."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)