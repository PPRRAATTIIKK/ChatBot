# 🏥 Medical Assistant Chatbot

> An NLP-powered chatbot for healthcare navigation — built with LSTM neural networks and TensorFlow/Keras.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-LSTM-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

---

## 📌 Overview

The **Medical Assistant Chatbot** is an intent-based conversational AI that helps users navigate key healthcare modules — from adverse drug reaction queries to hospital and pharmacy lookups. It uses a lightweight LSTM model trained on a custom `intents.json` dataset, with multi-turn context handling for complex interactions.

```
User: "Find me a pharmacy nearby"
Bot:  "Please provide pharmacy name"
User: "Apollo"
Bot:  "Loading pharmacy details..."
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **LSTM Intent Classification** | Sequence model with token embeddings for robust intent prediction |
| 🔄 **Multi-turn Context** | Remembers conversation state for pharmacy, hospital, and BP search flows |
| 💊 **Adverse Drug Reactions** | Query and navigate drug safety information |
| 🩺 **Blood Pressure Tracking** | Log and search patient BP records by ID |
| 🏨 **Hospital Lookup** | Search hospitals by name, location, and type |
| 💊 **Pharmacy Search** | Find pharmacies by name |
| 🛑 **Out-of-scope Fallback** | Confidence thresholding returns a safe "noanswer" response |
| 📊 **Training Visualizations** | Accuracy/loss curves and intent distribution charts |

---

## 🏗️ Model Architecture

```
Input (padded token sequence)
        │
  Embedding Layer
  (vocab_size+1 → 10 dims)
        │
  LSTM (10 units, return_sequences=True)
        │
    Flatten
        │
  Dense + Softmax
  (13 output intent classes)
```

**Training config:**
- Optimizer: `Adam`
- Loss: `Sparse Categorical Crossentropy`
- Early stopping: `patience=10` on accuracy
- Max epochs: `200`, batch size: `8`

---

## 🎯 Supported Intents

```
greeting              · goodbye              · thanks
options               · noanswer (fallback)
adverse_drug          · blood_pressure       · blood_pressure_search
search_blood_pressure_by_patient_id
pharmacy_search       · search_pharmacy_by_name
hospital_search       · search_hospital_by_params · search_hospital_by_type
```

---

## 📁 Project Structure

```
medical-chatbot/
│
├── medical_chatbot.ipynb       # Main notebook (train + inference + chat loop)
├── intents.json                # Intent patterns, responses & context definitions
│
├── medical_chatbot_model.h5    # Saved Keras model (generated after training)
├── tokenizer.pkl               # Fitted tokenizer (generated after training)
├── label_encoder.pkl           # Label encoder (generated after training)
│
├── training_curves.png         # Accuracy & loss plots (generated)
└── intent_distribution.png     # Pattern count per intent (generated)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

### Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot

# Launch the notebook
jupyter notebook medical_chatbot.ipynb
```

### Run on Google Colab

1. Upload `medical_chatbot.ipynb` and `intents.json` to Colab.
2. Uncomment the file upload cell at the top of Section 2.
3. Run all cells (`Runtime → Run all`).

---

## 💬 Usage

After training, the chatbot exposes three simple functions:

```python
# Predict the intent of any message
tag = predict_intent("Find me a pharmacy nearby")
# → "pharmacy_search"

# Get a random response for an intent
response = get_response("pharmacy_search")
# → "Please provide pharmacy name"

# Full pipeline in one call
reply = chat("I want to log blood pressure results")
# → "[Intent: blood_pressure]  Navigating to Blood Pressure module"
```

**Interactive loop** — run Section 13 in the notebook for a live terminal chat session. Type `quit` to exit.

---

## 🔧 Extending the Bot

Adding new capabilities is straightforward — no architecture changes needed:

1. Open `intents.json`
2. Add a new intent block:
   ```json
   {
     "tag": "appointment_booking",
     "patterns": ["Book an appointment", "Schedule a visit", "I need to see a doctor"],
     "responses": ["Navigating to appointment booking module"],
     "context": [""]
   }
   ```
3. Re-run the notebook from **Section 2** onwards.

---

## 📊 Training Results

The model trains in under a minute on CPU and typically converges to **>95% accuracy** on the training set within 100 epochs, with early stopping preventing overfitting.

Charts generated automatically during training:
- `training_curves.png` — accuracy and loss over epochs
- `intent_distribution.png` — pattern count per intent tag

---

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow / Keras** — LSTM model building & training
- **NumPy / Pandas** — data manipulation
- **scikit-learn** — label encoding
- **Matplotlib** — training visualizations
- **Jupyter Notebook** — interactive development environment

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change. Make sure to update `intents.json` and re-run training if you modify intent definitions.

---

<p align="center">Made with ❤️ for healthcare accessibility</p>
