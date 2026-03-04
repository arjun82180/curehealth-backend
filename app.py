"""
CureHealth AI — Flask Backend
═══════════════════════════════
AI Chat  : Groq API (LLaMA 3.3 70B) — Ultra fast, FREE
ML Model : Random Forest + Gradient Boosting Ensemble
═══════════════════════════════

Endpoints:
  GET  /           → API info
  GET  /health     → health check
  GET  /symptoms   → all symptoms list
  POST /predict    → disease prediction from symptoms
  POST /chat       → Groq AI health chatbot
  POST /report     → generate report data

Setup:
  pip install -r requirements.txt

  Windows:
    set GROQ_API_KEY=gsk_xxxxxxxxxxxx

  Linux / Mac:
    export GROQ_API_KEY=gsk_xxxxxxxxxxxx

  Run:
    python app.py

Free Groq API Key → https://console.groq.com
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import os
import traceback
from groq import Groq
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────────────────────
@app.route("/")
def root():
    return jsonify({
        "app":       "CureHealth AI Backend",
        "version":   "2.0.0",
        "status":    "running ✅",
        "ai_engine": "Groq — llama-3.3-70b-versatile",
        "endpoints": {
            "GET  /health":   "API health check",
            "GET  /symptoms": "Symptoms list",
            "POST /predict":  "Disease prediction",
            "POST /chat":     "Groq AI chatbot",
            "POST /report":   "Report data",
        }
    })


# ─────────────────────────────────────────────────────────────
# ML MODELS
# ─────────────────────────────────────────────────────────────
MODEL_DIR = "models"
rf_model = gb_model = label_encoder = metadata = None

def load_models():
    global rf_model, gb_model, label_encoder, metadata
    try:
        rf_model      = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
        gb_model      = joblib.load(f"{MODEL_DIR}/gb_model.pkl")
        label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
        with open(f"{MODEL_DIR}/metadata.json") as f:
            metadata = json.load(f)
        print("✅ ML Models loaded")
    except Exception as e:
        print(f"⚠️  ML Models not found — run train_model.py first\n   {e}")

load_models()


# ─────────────────────────────────────────────────────────────
# GROQ CLIENT
# Get free API key at: https://console.groq.com
# ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client  = Groq(api_key=GROQ_API_KEY)

# Model options (pick one):
#  "llama-3.3-70b-versatile"   ← Best quality  ✅ (recommended)
#  "llama3-8b-8192"            ← Fastest / lightest
#  "mixtral-8x7b-32768"        ← Good balance
GROQ_MODEL = "llama-3.3-70b-versatile"


# ─────────────────────────────────────────────────────────────
# DISEASE DATABASE
# ─────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Flu (Influenza)": {
        "severity": "medium",
        "description": "A viral respiratory illness caused by influenza viruses affecting the nose, throat, and lungs.",
        "common_medicines": ["Paracetamol", "Ibuprofen", "ORS", "Oseltamivir (Tamiflu — prescription)"],
        "doctors": ["General Physician"],
        "prevention": ["Annual flu vaccine", "Frequent hand washing", "Avoid close contact with sick people", "Cover coughs and sneezes"],
        "emergency": False,
    },
    "COVID-19": {
        "severity": "high",
        "description": "Coronavirus disease caused by SARS-CoV-2, affecting the respiratory system.",
        "common_medicines": ["Paracetamol", "ORS", "Vitamin C & D", "Doctor-prescribed antivirals"],
        "doctors": ["General Physician", "Pulmonologist"],
        "prevention": ["COVID vaccination", "Mask in crowded places", "Hand sanitization", "Ventilated spaces"],
        "emergency": True,
    },
    "Common Cold": {
        "severity": "low",
        "description": "Mild viral infection of the upper respiratory tract, usually harmless.",
        "common_medicines": ["Paracetamol", "Antihistamine", "Cough syrup", "Vitamin C", "Steam inhalation"],
        "doctors": ["General Physician"],
        "prevention": ["Regular hand washing", "Avoid touching face", "Adequate sleep", "Stay hydrated"],
        "emergency": False,
    },
    "Malaria": {
        "severity": "high",
        "description": "Parasitic infection transmitted by the bite of infected Anopheles mosquitoes.",
        "common_medicines": ["Chloroquine (prescription)", "Artemisinin-based combination therapy (prescription)"],
        "doctors": ["General Physician", "Infectious Disease Specialist"],
        "prevention": ["Sleep under mosquito nets", "Use repellents (DEET)", "Prophylactic antimalarials when traveling", "Drain stagnant water"],
        "emergency": True,
    },
    "Dengue Fever": {
        "severity": "high",
        "description": "Viral infection spread by Aedes mosquitoes, causing severe flu-like illness.",
        "common_medicines": ["Paracetamol ONLY — NEVER use ibuprofen or aspirin", "ORS for hydration", "Papaya leaf extract"],
        "doctors": ["General Physician", "Infectious Disease Specialist"],
        "prevention": ["Eliminate stagnant water around home", "Use mosquito repellent", "Wear long sleeves", "Use mosquito nets"],
        "emergency": True,
    },
    "Typhoid": {
        "severity": "high",
        "description": "Bacterial infection caused by Salmonella typhi, spread through contaminated food and water.",
        "common_medicines": ["Antibiotics — Ciprofloxacin or Azithromycin (prescription only)", "ORS", "Paracetamol for fever"],
        "doctors": ["General Physician", "Gastroenterologist"],
        "prevention": ["Typhoid vaccine", "Drink only safe/boiled water", "Avoid street food", "Proper hand hygiene"],
        "emergency": True,
    },
    "Pneumonia": {
        "severity": "high",
        "description": "Infection that inflames air sacs in lungs, causing them to fill with fluid.",
        "common_medicines": ["Antibiotics (prescription)", "Paracetamol", "Cough expectorant", "Rest and hydration"],
        "doctors": ["Pulmonologist", "General Physician"],
        "prevention": ["Pneumococcal vaccine", "No smoking", "Good nutrition", "Avoid cold and damp environments"],
        "emergency": True,
    },
    "Tuberculosis": {
        "severity": "high",
        "description": "Serious bacterial infection primarily affecting the lungs, spread through air.",
        "common_medicines": ["DOTS therapy — 6 month course minimum (prescription only — never self-medicate)"],
        "doctors": ["Pulmonologist", "Infectious Disease Specialist"],
        "prevention": ["BCG vaccine at birth", "Ensure good ventilation", "Avoid prolonged contact with TB patients", "Test and treat contacts"],
        "emergency": True,
    },
    "Diabetes": {
        "severity": "high",
        "description": "Chronic metabolic disorder where the body cannot properly regulate blood glucose.",
        "common_medicines": ["Metformin (prescription)", "Insulin (prescription)", "Diet control", "Regular blood glucose monitoring"],
        "doctors": ["Endocrinologist", "General Physician", "Diabetologist"],
        "prevention": ["Healthy balanced diet", "Regular physical exercise", "Maintain healthy weight", "Avoid sugary beverages"],
        "emergency": False,
    },
    "Hypertension": {
        "severity": "high",
        "description": "Persistently high blood pressure that can lead to heart disease and stroke.",
        "common_medicines": ["Amlodipine (prescription)", "Losartan (prescription)", "ACE inhibitors (prescription)"],
        "doctors": ["Cardiologist", "General Physician"],
        "prevention": ["Low-sodium diet (DASH diet)", "Regular aerobic exercise", "Limit alcohol", "No smoking", "Stress management"],
        "emergency": False,
    },
    "Gastroenteritis": {
        "severity": "medium",
        "description": "Inflammation of the stomach and intestines, usually from viral or bacterial infection.",
        "common_medicines": ["ORS — most important", "Zinc supplements", "Probiotics", "Paracetamol for fever"],
        "doctors": ["Gastroenterologist", "General Physician"],
        "prevention": ["Thorough hand washing before eating", "Food safety practices", "Safe drinking water", "Rotavirus vaccine for children"],
        "emergency": False,
    },
    "Food Poisoning": {
        "severity": "medium",
        "description": "Illness caused by eating food contaminated with bacteria, viruses, or toxins.",
        "common_medicines": ["ORS — stay hydrated", "Paracetamol for fever", "Avoid antidiarrheal drugs unless prescribed"],
        "doctors": ["General Physician"],
        "prevention": ["Cook food thoroughly", "Refrigerate perishables promptly", "Avoid raw/undercooked meat and eggs", "Wash hands before cooking"],
        "emergency": False,
    },
    "Migraine": {
        "severity": "medium",
        "description": "Neurological condition characterized by intense, recurrent headaches often with nausea.",
        "common_medicines": ["Paracetamol or Ibuprofen at onset", "Sumatriptan (prescription)", "Anti-nausea medication", "Rest in dark quiet room"],
        "doctors": ["Neurologist"],
        "prevention": ["Identify and avoid personal triggers", "Maintain regular sleep schedule", "Stay hydrated", "Manage stress", "Limit caffeine"],
        "emergency": False,
    },
    "Asthma": {
        "severity": "high",
        "description": "Chronic condition causing airway inflammation and narrowing, making breathing difficult.",
        "common_medicines": ["Salbutamol reliever inhaler (prescription)", "Corticosteroid preventer inhaler (prescription)", "Never stop medication without doctor advice"],
        "doctors": ["Pulmonologist", "Allergist"],
        "prevention": ["Identify and avoid triggers (dust, pollen, smoke)", "Air purifier at home", "Strict medication adherence", "Annual flu vaccine"],
        "emergency": True,
    },
    "Sinusitis": {
        "severity": "low",
        "description": "Inflammation of the sinuses, often following a cold or allergic reaction.",
        "common_medicines": ["Saline nasal rinse (Neti pot)", "Decongestant spray (max 3 days)", "Paracetamol for pain", "Steam inhalation"],
        "doctors": ["ENT Specialist", "General Physician"],
        "prevention": ["Treat allergies promptly", "Use humidifier", "Avoid smoking and pollutants", "Stay hydrated"],
        "emergency": False,
    },
    "Urinary Tract Infection": {
        "severity": "medium",
        "description": "Bacterial infection in any part of the urinary system — kidneys, bladder, or urethra.",
        "common_medicines": ["Antibiotics — Nitrofurantoin or Trimethoprim (prescription)", "Drink 8+ glasses of water daily", "Cranberry juice may help"],
        "doctors": ["Urologist", "General Physician"],
        "prevention": ["Drink plenty of water", "Urinate frequently, don't hold", "Proper hygiene", "Urinate after intercourse"],
        "emergency": False,
    },
    "Jaundice": {
        "severity": "high",
        "description": "Yellowing of skin and eyes due to elevated bilirubin, indicating liver dysfunction.",
        "common_medicines": ["Treatment depends on cause — see doctor immediately", "Rest", "Adequate hydration", "Avoid alcohol completely"],
        "doctors": ["Gastroenterologist", "Hepatologist", "General Physician"],
        "prevention": ["Hepatitis A and B vaccination", "Avoid alcohol", "Safe food and drinking water", "Avoid sharing needles"],
        "emergency": True,
    },
    "Chickenpox": {
        "severity": "medium",
        "description": "Highly contagious viral infection causing an itchy blister rash all over the body.",
        "common_medicines": ["Calamine lotion for itching", "Antihistamine (Cetirizine)", "Paracetamol — AVOID ibuprofen", "Acyclovir for severe cases (prescription)"],
        "doctors": ["General Physician", "Dermatologist"],
        "prevention": ["Varicella vaccine (very effective)", "Avoid contact with infected individuals", "Good hygiene"],
        "emergency": False,
    },
    "Allergy": {
        "severity": "low",
        "description": "Immune system overreaction to normally harmless substances (allergens).",
        "common_medicines": ["Cetirizine or Loratadine (antihistamine)", "Nasal corticosteroid spray", "Eye drops for allergic conjunctivitis", "Epinephrine for severe reactions (prescription)"],
        "doctors": ["Allergist", "ENT Specialist", "Dermatologist"],
        "prevention": ["Identify specific allergens via testing", "HEPA air purifier", "Hypoallergenic bedding", "Avoid outdoor activity during high pollen"],
        "emergency": False,
    },
    "Arthritis": {
        "severity": "medium",
        "description": "Inflammation of one or more joints causing persistent pain, stiffness, and reduced mobility.",
        "common_medicines": ["Ibuprofen or Naproxen for pain", "Paracetamol", "DMARDs for rheumatoid arthritis (prescription)", "Topical pain relief gels"],
        "doctors": ["Rheumatologist", "Orthopedist"],
        "prevention": ["Regular low-impact exercise", "Maintain healthy weight", "Protect joints during activity", "Calcium and Vitamin D intake"],
        "emergency": False,
    },
    "Hypothyroidism": {
        "severity": "medium",
        "description": "Underactive thyroid gland that doesn't produce enough hormones, slowing metabolism.",
        "common_medicines": ["Levothyroxine (prescription — take consistently)", "Regular TSH monitoring"],
        "doctors": ["Endocrinologist", "General Physician"],
        "prevention": ["Adequate iodine in diet", "Regular thyroid screening especially for women over 35", "Avoid goitrogenic foods in excess"],
        "emergency": False,
    },
    "Anaemia": {
        "severity": "medium",
        "description": "Deficiency of red blood cells or haemoglobin, reducing oxygen delivery to body tissues.",
        "common_medicines": ["Iron supplements with Vitamin C (enhances absorption)", "Vitamin B12 injections if deficient", "Folic acid supplements"],
        "doctors": ["General Physician", "Haematologist"],
        "prevention": ["Iron-rich diet (leafy greens, meat, legumes)", "Vitamin C with meals", "Regular blood tests", "Treat underlying causes"],
        "emergency": False,
    },
    "Heart Disease": {
        "severity": "high",
        "description": "Umbrella term for conditions affecting the heart's structure and function.",
        "common_medicines": ["Medications vary by condition — follow cardiologist's prescription strictly", "NEVER self-medicate for heart conditions"],
        "doctors": ["Cardiologist", "Cardiac Surgeon"],
        "prevention": ["Heart-healthy diet (Mediterranean diet)", "Regular aerobic exercise", "No smoking", "Control blood pressure, cholesterol, and diabetes"],
        "emergency": True,
    },
    "Stroke": {
        "severity": "high",
        "description": "Medical emergency where blood supply to brain is cut off, causing brain cells to die rapidly.",
        "common_medicines": ["EMERGENCY — call 112 immediately", "tPA clot-busting drug (within 4.5 hours — hospital only)", "Aspirin after hospital assessment"],
        "doctors": ["Neurologist", "Emergency Medicine Specialist"],
        "prevention": ["Control hypertension and diabetes", "No smoking", "Regular exercise", "Healthy diet", "Limit alcohol"],
        "emergency": True,
    },
    "Meningitis": {
        "severity": "high",
        "description": "Serious inflammation of the membranes surrounding the brain and spinal cord.",
        "common_medicines": ["EMERGENCY IV antibiotics or antivirals (hospital only)", "Do not delay — seek emergency care immediately"],
        "doctors": ["Neurologist", "Infectious Disease Specialist", "Emergency Medicine"],
        "prevention": ["Meningococcal and pneumococcal vaccines", "Good respiratory hygiene", "Avoid sharing drinks/utensils"],
        "emergency": True,
    },
    "Depression": {
        "severity": "medium",
        "description": "Common mental health disorder causing persistent sadness, loss of interest, and difficulty functioning.",
        "common_medicines": ["SSRIs — Fluoxetine, Sertraline (prescription)", "SNRIs (prescription)", "Psychotherapy is equally important"],
        "doctors": ["Psychiatrist", "Psychologist", "Counselor"],
        "prevention": ["Maintain social connections", "Regular exercise (proven effective)", "Adequate sleep", "Mindfulness and therapy", "Limit alcohol"],
        "emergency": False,
    },
    "Appendicitis": {
        "severity": "high",
        "description": "Inflammation of the appendix requiring urgent surgical treatment.",
        "common_medicines": ["EMERGENCY — requires surgical removal (appendectomy)", "IV antibiotics pre-surgery"],
        "doctors": ["General Surgeon", "Emergency Medicine"],
        "prevention": ["High-fiber diet may reduce risk", "No proven prevention — seek care at first sign of severe right lower abdominal pain"],
        "emergency": True,
    },
    "Kidney Disease": {
        "severity": "high",
        "description": "Progressive loss of kidney function, affecting the body's ability to filter waste.",
        "common_medicines": ["Depends on cause — see nephrologist", "Blood pressure medications to slow progression", "Dietary restrictions"],
        "doctors": ["Nephrologist", "General Physician"],
        "prevention": ["Stay well hydrated", "Control blood pressure and diabetes strictly", "Avoid overuse of NSAIDs (ibuprofen)", "Regular kidney function tests"],
        "emergency": False,
    },
}


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_overall_severity(predictions):
    if not predictions:
        return "low"
    top = predictions[0]
    if top.get("severity") == "high" and top["probability"] > 40:
        return "high"
    if top.get("severity") in ["high", "medium"] and top["probability"] > 25:
        return "medium"
    return "low"


def rule_based_predict(symptoms):
    """Fallback when ML model is not loaded."""
    rules = {
        "Flu (Influenza)":     ["fever", "cough", "headache", "body aches", "fatigue", "chills", "sore throat"],
        "COVID-19":            ["fever", "cough", "fatigue", "shortness of breath", "loss of appetite", "body aches"],
        "Common Cold":         ["runny nose", "sore throat", "cough", "headache", "sneezing"],
        "Malaria":             ["fever", "chills", "headache", "sweating", "nausea", "body aches", "fatigue"],
        "Dengue Fever":        ["fever", "headache", "rash", "body aches", "nausea", "vomiting", "fatigue"],
        "Typhoid":             ["fever", "stomach pain", "headache", "fatigue", "loss of appetite", "nausea"],
        "Migraine":            ["headache", "nausea", "vomiting", "blurred vision", "dizziness"],
        "Gastroenteritis":     ["nausea", "vomiting", "diarrhea", "stomach pain", "fever", "dehydration"],
        "Food Poisoning":      ["nausea", "vomiting", "diarrhea", "stomach pain"],
        "Allergy":             ["itching", "runny nose", "rash", "sneezing", "swelling"],
        "Asthma":              ["shortness of breath", "cough", "chest pain", "wheezing"],
        "Urinary Tract Infection": ["frequent urination", "dark urine", "fever", "stomach pain"],
        "Anaemia":             ["fatigue", "dizziness", "shortness of breath", "muscle weakness"],
        "Sinusitis":           ["headache", "runny nose", "sore throat", "ear pain"],
        "Jaundice":            ["yellowing skin", "dark urine", "fatigue", "loss of appetite", "stomach pain"],
        "Hypertension":        ["headache", "dizziness", "palpitations", "blurred vision"],
        "Depression":          ["fatigue", "anxiety", "depression", "loss of appetite", "weight loss"],
    }
    syms_lower = {s.lower() for s in symptoms}
    scores = {}
    for disease, rule_syms in rules.items():
        match = sum(1 for s in rule_syms if any(s in sym for sym in syms_lower))
        if match > 0:
            scores[disease] = round((match / len(rule_syms)) * 100, 1)

    predictions = []
    for disease, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        if score < 5:
            continue
        info = DISEASE_INFO.get(disease, {})
        predictions.append({
            "disease":          disease,
            "probability":      round(min(score, 90), 1),
            "severity":         info.get("severity", "medium"),
            "description":      info.get("description", ""),
            "common_medicines": info.get("common_medicines", []),
            "doctors":          info.get("doctors", ["General Physician"]),
            "prevention":       info.get("prevention", []),
            "emergency":        info.get("emergency", False),
        })
    return predictions


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":       "ok ✅",
        "ai_engine":    f"Groq — {GROQ_MODEL}",
        "models_loaded": rf_model is not None,
        "groq_key_set": GROQ_API_KEY != "YOUR_GROQ_API_KEY",
        "diseases":     len(DISEASE_INFO),
        "timestamp":    datetime.now().isoformat(),
    })


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    symptoms = [
        {"id": i, "name": n, "icon": ic, "category": c}
        for i, (n, ic, c) in enumerate([
            ("Fever",               "🌡️", "general"),
            ("Headache",            "🤕", "neuro"),
            ("Cough",               "😷", "respiratory"),
            ("Sore Throat",         "🤧", "respiratory"),
            ("Runny Nose",          "👃", "respiratory"),
            ("Fatigue",             "😴", "general"),
            ("Body Aches",          "🦴", "general"),
            ("Nausea",              "🤢", "digestive"),
            ("Vomiting",            "🤮", "digestive"),
            ("Diarrhea",            "💊", "digestive"),
            ("Chest Pain",          "💗", "cardiac"),
            ("Shortness of Breath", "😤", "respiratory"),
            ("Dizziness",           "😵", "neuro"),
            ("Rash",                "🔴", "skin"),
            ("Itching",             "🖐️", "skin"),
            ("Stomach Pain",        "🫃", "digestive"),
            ("Loss of Appetite",    "🍽️", "digestive"),
            ("Chills",              "🥶", "general"),
            ("Sweating",            "💦", "general"),
            ("Blurred Vision",      "👁️", "neuro"),
            ("Ear Pain",            "👂", "general"),
            ("Weight Loss",         "⚖️", "general"),
            ("Swelling",            "🫸", "general"),
            ("Palpitations",        "💓", "cardiac"),
            ("Anxiety",             "😰", "mental"),
            ("Constipation",        "😣", "digestive"),
            ("Dark Urine",          "🫙", "urinary"),
            ("Yellowing Skin",      "🟡", "skin"),
            ("Neck Stiffness",      "🧍", "neuro"),
            ("Acidity",             "🔥", "digestive"),
            ("Dehydration",         "💧", "general"),
            ("Frequent Urination",  "🚽", "urinary"),
            ("Muscle Weakness",     "💪", "general"),
            ("Joint Stiffness",     "🦵", "general"),
            ("Skin Peeling",        "🧴", "skin"),
            ("Depression",          "😔", "mental"),
        ])
    ]
    categories = {
        "general":     {"label": "General",       "color": "#00E5C3"},
        "respiratory": {"label": "Respiratory",   "color": "#4C9EFF"},
        "digestive":   {"label": "Digestive",     "color": "#00D4AA"},
        "neuro":       {"label": "Neurological",  "color": "#B06EFF"},
        "cardiac":     {"label": "Cardiac",       "color": "#FF3B3B"},
        "skin":        {"label": "Skin",          "color": "#F5A623"},
        "urinary":     {"label": "Urinary",       "color": "#00D4FF"},
        "mental":      {"label": "Mental Health", "color": "#FF6B9D"},
    }
    return jsonify({"symptoms": symptoms, "categories": categories, "total": len(symptoms)})


@app.route("/predict", methods=["POST"])
def predict_disease():
    """
    Predict diseases from selected symptoms.

    Request body:
    {
        "symptoms": ["Fever", "Cough", "Headache"],
        "age": 25,        (optional)
        "gender": "male"  (optional)
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        selected_symptoms = data.get("symptoms", [])
        if not selected_symptoms:
            return jsonify({"error": "No symptoms provided. Please select at least one symptom."}), 400

        # ── ML Model prediction ────────────────────────────
        if rf_model is not None:
            features       = metadata["features"]
            feature_vector = [1 if f in selected_symptoms else 0 for f in features]
            X              = np.array([feature_vector])
            rf_proba       = rf_model.predict_proba(X)[0]
            gb_proba       = gb_model.predict_proba(X)[0]
            ensemble_proba = 0.6 * rf_proba + 0.4 * gb_proba   # 60% RF + 40% GB
            top_indices    = np.argsort(ensemble_proba)[::-1][:5]
            diseases       = label_encoder.classes_

            predictions = []
            for idx in top_indices:
                disease = diseases[idx]
                prob    = float(ensemble_proba[idx])
                if prob < 0.03:
                    continue
                info = DISEASE_INFO.get(disease, {})
                predictions.append({
                    "disease":          disease,
                    "probability":      round(prob * 100, 1),
                    "severity":         info.get("severity", "medium"),
                    "description":      info.get("description", ""),
                    "common_medicines": info.get("common_medicines", []),
                    "doctors":          info.get("doctors", ["General Physician"]),
                    "prevention":       info.get("prevention", []),
                    "emergency":        info.get("emergency", False),
                })
        else:
            # ── Rule-based fallback ─────────────────────────
            predictions = rule_based_predict(selected_symptoms)

        if not predictions:
            predictions = [{
                "disease":          "Unspecified Condition",
                "probability":      50.0,
                "severity":         "medium",
                "description":      "Unable to determine specific condition. Please consult a doctor.",
                "common_medicines": ["Consult a doctor for appropriate medication"],
                "doctors":          ["General Physician"],
                "prevention":       ["Maintain healthy lifestyle", "Stay hydrated", "Get adequate rest"],
                "emergency":        False,
            }]

        severity     = get_overall_severity(predictions)
        all_doctors  = list({d for p in predictions[:3] for d in p.get("doctors", [])})
        all_meds     = list({m for p in predictions[:2] for m in p.get("common_medicines", [])})
        is_emergency = any(p.get("emergency") and p["probability"] > 30 for p in predictions[:2])

        return jsonify({
            "predictions":           predictions,
            "overall_severity":      severity,
            "recommended_doctors":   all_doctors[:4],
            "recommended_medicines": all_meds[:6],
            "is_emergency":          is_emergency,
            "symptoms_analyzed":     selected_symptoms,
            "model_used":            "ensemble_ml" if rf_model else "rule_based",
            "timestamp":             datetime.now().isoformat(),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Groq AI Health Chatbot — LLaMA 3.3 70B

    Request body:
    {
        "message": "What to do for fever?",
        "history": [
            {"role": "user",      "content": "I have a headache"},
            {"role": "assistant", "content": "..."}
        ],
        "context": {
            "symptoms":    ["Fever", "Cough"],
            "top_disease": "Flu"
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        user_message = data.get("message", "").strip()
        history      = data.get("history", [])
        context      = data.get("context", {})

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # ── System Prompt ──────────────────────────────────
        system_prompt = """You are CureHealth AI — a knowledgeable, empathetic medical health assistant.

Your responsibilities:
- Provide accurate, evidence-based health information
- Help users understand symptoms, conditions, medications, and wellness
- Always recommend consulting a qualified doctor for diagnosis or treatment
- For emergencies (chest pain, difficulty breathing, stroke symptoms, severe bleeding) IMMEDIATELY advise calling 112 (India) or 911 (US)
- Never give definitive medical diagnoses — use phrases like "may suggest", "could indicate", "commonly associated with"
- Keep responses clear, concise, and actionable
- Respond in the same language the user writes in (Hindi or English)

Response format (use emojis for clarity):
🔍 Key information
💊 Medication notes (general guidance only — always say "consult your doctor")
✅ Actionable steps / home care
⚠️ Warning signs — when to seek immediate care

Tone: Professional yet warm. Like a knowledgeable friend who happens to be a doctor."""

        # Add patient context if available
        if context.get("symptoms"):
            system_prompt += f"\n\n📋 Patient's reported symptoms: {', '.join(context['symptoms'])}"
        if context.get("top_disease"):
            system_prompt += f"\n🧬 AI's top predicted condition: {context['top_disease']}"

        # ── Build Messages (OpenAI-compatible format) ──────
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last 10 messages)
        for msg in history[-10:]:
            role = msg.get("role", "user")
            if role == "model":
                role = "assistant"      # Gemini format → OpenAI format
            if role not in ["user", "assistant"]:
                continue
            messages.append({"role": role, "content": msg.get("content", "")})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # ── Call Groq API ──────────────────────────────────
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=0.9,
        )

        reply = response.choices[0].message.content

        return jsonify({
            "reply":     reply,
            "model":     GROQ_MODEL,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        traceback.print_exc()
        # Check for common Groq errors
        err_str = str(e)
        if "401" in err_str or "invalid_api_key" in err_str.lower():
            return jsonify({"error": "Invalid Groq API key. Get free key at: https://console.groq.com"}), 401
        if "429" in err_str or "rate_limit" in err_str.lower():
            return jsonify({"error": "Rate limit reached. Please wait a moment and try again."}), 429
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500


@app.route("/report", methods=["POST"])
def generate_report():
    """
    Generate structured report data for PDF generation.

    Request body:
    {
        "symptoms":    ["Fever", "Cough"],
        "predictions": [...],
        "user_name":   "Arjun Singh",
        "age":         25,
        "gender":      "male"
    }
    """
    try:
        data = request.get_json() or {}
        return jsonify({
            "patient_name": data.get("user_name", "Patient"),
            "age":          data.get("age", "N/A"),
            "gender":       data.get("gender", "N/A"),
            "report_date":  datetime.now().strftime("%d %B %Y, %I:%M %p"),
            "report_id":    f"CH-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "symptoms":     data.get("symptoms", []),
            "predictions":  data.get("predictions", [])[:3],
            "disclaimer":   (
                "This AI-generated health report is for informational purposes only. "
                "It does NOT constitute a medical diagnosis or replace professional medical advice. "
                "Always consult a qualified healthcare professional for diagnosis and treatment."
            ),
            "emergency_note": "For life-threatening emergencies, call 112 (India) or 911 (US) immediately.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print("\n" + "═" * 50)
    print("   CureHealth AI — Backend Server")
    print("═" * 50)
    print(f"   AI Engine  : Groq ({GROQ_MODEL})")
    print(f"   Groq Key   : {'✅ SET' if GROQ_API_KEY != 'YOUR_GROQ_API_KEY' else '❌ NOT SET — get free key at console.groq.com'}")
    print(f"   ML Models  : {'✅ Loaded' if rf_model else '⚠️  Not found — run train_model.py'}")
    print(f"   Diseases   : {len(DISEASE_INFO)} in database")
    print(f"   Port       : {port}")
    print(f"   Debug Mode : {debug}")
    print("═" * 50 + "\n")

    app.run(host="0.0.0.0", port=port, debug=debug)