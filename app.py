
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import os
from groq import Groq
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)

# ─── Load ML Models ─────────────────────────────────────────
MODEL_DIR = "models"
rf_model = None
gb_model = None
label_encoder = None
metadata = None

def load_models():
    global rf_model, gb_model, label_encoder, metadata
    try:
        rf_model = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
        gb_model = joblib.load(f"{MODEL_DIR}/gb_model.pkl")
        label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
        with open(f"{MODEL_DIR}/metadata.json") as f:
            metadata = json.load(f)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"⚠️  Models not found, run train_model.py first: {e}")

load_models()

# ─── Gemini Client ───────────────────────────────────────────
# Get your FREE API key from: https://aistudio.google.com/app/apikey
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL  = "llama-3.3-70b-versatile"

# ─── Disease Info Database ───────────────────────────────────
DISEASE_INFO = {
    "Flu (Influenza)": {
        "severity": "medium",
        "description": "A viral respiratory illness caused by influenza viruses.",
        "common_medicines": ["Paracetamol", "Ibuprofen", "ORS", "Oseltamivir (Tamiflu)"],
        "doctors": ["General Physician"],
        "prevention": ["Annual flu vaccine", "Hand hygiene", "Avoid close contact with sick people"],
        "emergency": False,
    },
    "COVID-19": {
        "severity": "high",
        "description": "Coronavirus disease caused by SARS-CoV-2.",
        "common_medicines": ["Paracetamol", "ORS", "Vitamin C", "Doctor-prescribed antivirals"],
        "doctors": ["General Physician", "Pulmonologist"],
        "prevention": ["COVID vaccination", "Mask usage", "Hand sanitization", "Social distancing"],
        "emergency": True,
    },
    "Common Cold": {
        "severity": "low",
        "description": "Viral infection of the upper respiratory tract.",
        "common_medicines": ["Paracetamol", "Antihistamine", "Cough syrup", "Vitamin C"],
        "doctors": ["General Physician"],
        "prevention": ["Hand washing", "Avoid touching face", "Adequate sleep"],
        "emergency": False,
    },
    "Malaria": {
        "severity": "high",
        "description": "Parasitic infection transmitted through mosquito bites.",
        "common_medicines": ["Chloroquine", "Artemisinin-based therapy (prescription only)"],
        "doctors": ["General Physician", "Infectious Disease Specialist"],
        "prevention": ["Mosquito nets", "Repellents", "Prophylactic antimalarials when travelling"],
        "emergency": True,
    },
    "Dengue Fever": {
        "severity": "high",
        "description": "Viral infection spread by Aedes mosquitoes.",
        "common_medicines": ["Paracetamol (NOT ibuprofen/aspirin)", "ORS", "Papaya leaf extract"],
        "doctors": ["General Physician", "Infectious Disease Specialist"],
        "prevention": ["Eliminate stagnant water", "Use mosquito repellent", "Wear full sleeves"],
        "emergency": True,
    },
    "Typhoid": {
        "severity": "high",
        "description": "Bacterial infection caused by Salmonella typhi.",
        "common_medicines": ["Antibiotics (prescription)", "Paracetamol", "ORS"],
        "doctors": ["General Physician", "Gastroenterologist"],
        "prevention": ["Typhoid vaccine", "Safe food and water", "Hygiene"],
        "emergency": True,
    },
    "Pneumonia": {
        "severity": "high",
        "description": "Lung infection causing air sacs to fill with fluid.",
        "common_medicines": ["Antibiotics (prescription)", "Paracetamol", "Cough medicine"],
        "doctors": ["Pulmonologist", "General Physician"],
        "prevention": ["Pneumonia vaccine", "No smoking", "Good nutrition"],
        "emergency": True,
    },
    "Tuberculosis": {
        "severity": "high",
        "description": "Bacterial infection primarily affecting the lungs.",
        "common_medicines": ["DOTS therapy (prescription only — 6 month course)"],
        "doctors": ["Pulmonologist", "Infectious Disease Specialist"],
        "prevention": ["BCG vaccine", "Good ventilation", "Avoid prolonged contact with TB patients"],
        "emergency": True,
    },
    "Diabetes": {
        "severity": "high",
        "description": "Metabolic disorder affecting blood glucose regulation.",
        "common_medicines": ["Metformin (prescription)", "Insulin (prescription)", "Diet control"],
        "doctors": ["Endocrinologist", "General Physician"],
        "prevention": ["Healthy diet", "Regular exercise", "Maintain healthy weight"],
        "emergency": False,
    },
    "Hypertension": {
        "severity": "high",
        "description": "Persistently elevated blood pressure in arteries.",
        "common_medicines": ["Amlodipine (prescription)", "ACE inhibitors (prescription)"],
        "doctors": ["Cardiologist", "General Physician"],
        "prevention": ["Low-salt diet", "Exercise", "Stress management", "Limit alcohol"],
        "emergency": False,
    },
    "Gastroenteritis": {
        "severity": "medium",
        "description": "Inflammation of stomach and intestines, often from infection.",
        "common_medicines": ["ORS", "Zinc supplements", "Probiotics", "Paracetamol"],
        "doctors": ["Gastroenterologist", "General Physician"],
        "prevention": ["Proper hand washing", "Food hygiene", "Safe drinking water"],
        "emergency": False,
    },
    "Food Poisoning": {
        "severity": "medium",
        "description": "Illness caused by consuming contaminated food or water.",
        "common_medicines": ["ORS", "Antidiarrheal (adult only)", "Paracetamol"],
        "doctors": ["General Physician"],
        "prevention": ["Proper food handling", "Avoid raw/undercooked food", "Refrigerate food"],
        "emergency": False,
    },
    "Migraine": {
        "severity": "medium",
        "description": "Neurological condition causing severe recurring headaches.",
        "common_medicines": ["Paracetamol", "Ibuprofen", "Sumatriptan (prescription)", "Rest in dark room"],
        "doctors": ["Neurologist"],
        "prevention": ["Identify and avoid triggers", "Regular sleep", "Stress management"],
        "emergency": False,
    },
    "Asthma": {
        "severity": "high",
        "description": "Chronic condition causing airway inflammation and breathing difficulty.",
        "common_medicines": ["Salbutamol inhaler (prescription)", "Corticosteroids (prescription)"],
        "doctors": ["Pulmonologist", "Allergist"],
        "prevention": ["Avoid allergens", "Air purifier", "Medication compliance"],
        "emergency": True,
    },
    "Sinusitis": {
        "severity": "low",
        "description": "Inflammation of the sinuses, often following a cold.",
        "common_medicines": ["Decongestants", "Nasal saline rinse", "Paracetamol", "Antihistamine"],
        "doctors": ["ENT Specialist", "General Physician"],
        "prevention": ["Treat allergies promptly", "Humidifier use", "Avoid irritants"],
        "emergency": False,
    },
    "Urinary Tract Infection": {
        "severity": "medium",
        "description": "Bacterial infection in the urinary system.",
        "common_medicines": ["Antibiotics (prescription)", "Cranberry juice", "Increased water intake"],
        "doctors": ["Urologist", "General Physician"],
        "prevention": ["Stay hydrated", "Proper hygiene", "Urinate after intercourse"],
        "emergency": False,
    },
    "Jaundice": {
        "severity": "high",
        "description": "Yellowing of skin and eyes due to liver dysfunction.",
        "common_medicines": ["Rest", "Adequate hydration", "Liver support supplements (prescription)"],
        "doctors": ["Gastroenterologist", "Hepatologist"],
        "prevention": ["Hepatitis vaccination", "Avoid alcohol", "Safe food/water"],
        "emergency": True,
    },
    "Chickenpox": {
        "severity": "medium",
        "description": "Highly contagious viral infection causing itchy blister rash.",
        "common_medicines": ["Calamine lotion", "Antihistamine", "Paracetamol (NOT aspirin for children)"],
        "doctors": ["General Physician", "Dermatologist"],
        "prevention": ["Varicella vaccine", "Avoid contact with infected persons"],
        "emergency": False,
    },
    "Psoriasis": {
        "severity": "medium",
        "description": "Chronic skin condition causing scaly patches.",
        "common_medicines": ["Topical corticosteroids (prescription)", "Moisturizers", "Vitamin D creams"],
        "doctors": ["Dermatologist"],
        "prevention": ["Avoid triggers", "Stress management", "Moisturize regularly"],
        "emergency": False,
    },
    "Allergy": {
        "severity": "low",
        "description": "Immune system reaction to allergens (pollen, dust, food, etc.).",
        "common_medicines": ["Antihistamine (Cetirizine/Loratadine)", "Nasal steroids", "Eye drops"],
        "doctors": ["Allergist", "ENT Specialist"],
        "prevention": ["Identify and avoid allergens", "Air purifier", "HEPA filter"],
        "emergency": False,
    },
    "Arthritis": {
        "severity": "medium",
        "description": "Inflammation of one or more joints causing pain and stiffness.",
        "common_medicines": ["Ibuprofen", "Paracetamol", "DMARDs (prescription)"],
        "doctors": ["Rheumatologist", "Orthopedist"],
        "prevention": ["Regular exercise", "Maintain healthy weight", "Joint protection"],
        "emergency": False,
    },
    "Hypothyroidism": {
        "severity": "medium",
        "description": "Underactive thyroid gland not producing enough hormones.",
        "common_medicines": ["Levothyroxine (prescription)"],
        "doctors": ["Endocrinologist"],
        "prevention": ["Regular thyroid screening", "Adequate iodine intake"],
        "emergency": False,
    },
    "Anaemia": {
        "severity": "medium",
        "description": "Deficiency of red blood cells or haemoglobin.",
        "common_medicines": ["Iron supplements", "Vitamin B12", "Folic acid", "Diet rich in iron"],
        "doctors": ["General Physician", "Haematologist"],
        "prevention": ["Iron-rich diet", "Vitamin C with iron food", "Regular checkups"],
        "emergency": False,
    },
    "Heart Disease": {
        "severity": "high",
        "description": "Various conditions affecting heart structure and function.",
        "common_medicines": ["As prescribed by cardiologist — do not self-medicate"],
        "doctors": ["Cardiologist"],
        "prevention": ["Heart-healthy diet", "Regular exercise", "No smoking", "Stress management"],
        "emergency": True,
    },
    "Meningitis": {
        "severity": "high",
        "description": "Inflammation of the membranes surrounding the brain and spinal cord.",
        "common_medicines": ["Antibiotics/antivirals (prescription — emergency)", "Corticosteroids"],
        "doctors": ["Neurologist", "Infectious Disease Specialist"],
        "prevention": ["Meningitis vaccine", "Good hygiene", "Avoid sharing utensils"],
        "emergency": True,
    },
    "Stroke": {
        "severity": "high",
        "description": "Brain cell death due to interrupted blood supply — EMERGENCY.",
        "common_medicines": ["tPA (clot buster — emergency)", "Aspirin (as directed)"],
        "doctors": ["Neurologist", "Emergency Medicine"],
        "prevention": ["Control BP/cholesterol", "No smoking", "Regular exercise"],
        "emergency": True,
    },
    "Depression": {
        "severity": "medium",
        "description": "Mood disorder causing persistent sadness and loss of interest.",
        "common_medicines": ["SSRIs (prescription)", "Therapy", "Lifestyle changes"],
        "doctors": ["Psychiatrist", "Psychologist"],
        "prevention": ["Social connections", "Exercise", "Therapy", "Adequate sleep"],
        "emergency": False,
    },
    "Appendicitis": {
        "severity": "high",
        "description": "Inflammation of the appendix — may require surgery.",
        "common_medicines": ["Surgical intervention (appendectomy)", "IV antibiotics"],
        "doctors": ["General Surgeon", "Emergency Medicine"],
        "prevention": ["High-fiber diet"],
        "emergency": True,
    },
    "Kidney Disease": {
        "severity": "high",
        "description": "Progressive loss of kidney function.",
        "common_medicines": ["Medication depends on cause (prescription)", "Diet control"],
        "doctors": ["Nephrologist"],
        "prevention": ["Stay hydrated", "Control blood pressure/diabetes", "Avoid NSAIDs overuse"],
        "emergency": False,
    },
}

# ─── Routes ──────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "models_loaded": rf_model is not None,
        "diseases_count": len(DISEASE_INFO),
        "timestamp": datetime.now().isoformat()
    })


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    """Return all selectable symptoms grouped by category."""
    symptoms = [
        {"id": i, "name": name, "icon": icon, "category": cat}
        for i, (name, icon, cat) in enumerate([
            ("Fever", "🌡️", "general"), ("Headache", "🤕", "neuro"),
            ("Cough", "😷", "respiratory"), ("Sore Throat", "🤧", "respiratory"),
            ("Runny Nose", "👃", "respiratory"), ("Fatigue", "😴", "general"),
            ("Body Aches", "🦴", "general"), ("Nausea", "🤢", "digestive"),
            ("Vomiting", "🤮", "digestive"), ("Diarrhea", "💊", "digestive"),
            ("Chest Pain", "💗", "cardiac"), ("Shortness of Breath", "😤", "respiratory"),
            ("Dizziness", "😵", "neuro"), ("Rash", "🔴", "skin"),
            ("Itching", "🖐️", "skin"), ("Stomach Pain", "🫃", "digestive"),
            ("Loss of Appetite", "🍽️", "digestive"), ("Chills", "🥶", "general"),
            ("Sweating", "💦", "general"), ("Blurred Vision", "👁️", "neuro"),
            ("Ear Pain", "👂", "general"), ("Weight Loss", "⚖️", "general"),
            ("Swelling", "🫸", "general"), ("Palpitations", "💓", "cardiac"),
            ("Anxiety", "😰", "mental"), ("Constipation", "😣", "digestive"),
            ("Dark Urine", "🫙", "urinary"), ("Yellowing Skin", "🟡", "skin"),
            ("Neck Stiffness", "🧍", "neuro"), ("Acidity", "🔥", "digestive"),
            ("Dehydration", "💧", "general"), ("Frequent Urination", "🚽", "urinary"),
            ("Muscle Weakness", "💪", "general"), ("Joint Stiffness", "🦵", "general"),
            ("Skin Peeling", "🫁", "skin"), ("Depression", "😔", "mental"),
        ])
    ]
    categories = {
        "general": {"label": "General", "color": "#1B6CA8"},
        "respiratory": {"label": "Respiratory", "color": "#2D9CDB"},
        "digestive": {"label": "Digestive", "color": "#27AE60"},
        "neuro": {"label": "Neurological", "color": "#9B51E0"},
        "cardiac": {"label": "Cardiac", "color": "#EB5757"},
        "skin": {"label": "Skin", "color": "#F2994A"},
        "urinary": {"label": "Urinary", "color": "#F9C74F"},
        "mental": {"label": "Mental Health", "color": "#6C757D"},
    }
    return jsonify({"symptoms": symptoms, "categories": categories})


@app.route("/predict", methods=["POST"])
def predict_disease():
    """
    Predict diseases from selected symptoms.
    Body: { "symptoms": ["Fever", "Headache", "Cough"], "age": 30, "gender": "male" }
    """
    try:
        data = request.get_json()
        selected_symptoms = data.get("symptoms", [])
        age = data.get("age", 25)
        gender = data.get("gender", "unknown")

        if not selected_symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        if rf_model is None:
            # Fallback: rule-based prediction when model not trained
            return _rule_based_predict(selected_symptoms, age, gender)

        # Build feature vector
        features = metadata["features"]
        feature_vector = [1 if f in selected_symptoms else 0 for f in features]
        X = np.array([feature_vector])

        # Ensemble prediction
        rf_proba = rf_model.predict_proba(X)[0]
        gb_proba = gb_model.predict_proba(X)[0]
        ensemble_proba = (0.6 * rf_proba + 0.4 * gb_proba)

        # ── Smart Scoring: penalise rare/serious if few symptoms ──
        n_symptoms = len(selected_symptoms)
        disease_symptom_map = metadata.get("user_symptom_map", {})
        disease_meta = metadata.get("disease_metadata", {})

        # Build symptom→disease coverage map
        from train_model import DISEASE_SYMPTOM_MAP as DSM

        def coverage_score(disease_name, selected):
            """How many selected symptoms match this disease (normalised)."""
            all_syms = DSM.get(disease_name, [])
            sym_map  = metadata.get("user_symptom_map", {})
            matched  = 0
            for sel in selected:
                internal = sym_map.get(sel, [sel.lower()])
                if any(iv in all_syms for iv in internal):
                    matched += 1
            return matched / max(len(selected), 1)

        diseases = label_encoder.classes_
        top_indices = np.argsort(ensemble_proba)[::-1][:15]  # grab more candidates

        scored = []
        for idx in top_indices:
            disease = diseases[idx]
            ml_prob  = float(ensemble_proba[idx])
            if ml_prob < 0.01:
                continue

            cov  = coverage_score(disease, selected_symptoms)
            meta = disease_meta.get(disease, {})
            sev  = meta.get("severity", "medium")

            # Severity penalty: high-severity diseases need stronger symptom match
            sev_penalty = 1.0
            if sev == "high"   and cov < 0.5 and n_symptoms <= 3:
                sev_penalty = 0.35   # heavily penalise rare-disease if few symptoms
            elif sev == "high"  and cov < 0.4:
                sev_penalty = 0.55
            elif sev == "medium" and cov < 0.3:
                sev_penalty = 0.75

            # Common/low diseases get a small boost when symptoms fully match
            if sev == "low" and cov >= 0.6:
                sev_penalty = 1.15

            final_score = ml_prob * cov * sev_penalty
            scored.append((disease, ml_prob, cov, final_score, sev))

        # Sort by final smart score
        scored.sort(key=lambda x: x[3], reverse=True)

        predictions = []
        for disease, ml_prob, cov, final_score, sev in scored[:7]:
            if final_score < 0.005:
                continue
            # Display probability: blend ML prob with coverage for realism
            display_prob = round(min((ml_prob * 0.5 + cov * 0.5) * 100, 92), 1)
            if display_prob < 5:
                continue
            info = DISEASE_INFO.get(disease, {})
            dm   = disease_meta.get(disease, {})
            predictions.append({
                "disease":          disease,
                "probability":      display_prob,
                "severity":         dm.get("severity", info.get("severity", "medium")),
                "description":      info.get("description", dm.get("description", "")),
                "common_medicines": dm.get("medicines",   info.get("common_medicines", ["Consult doctor"])),
                "doctors":          dm.get("doctors",     info.get("doctors", ["General Physician"])),
                "prevention":       info.get("prevention", []),
                "emergency":        info.get("emergency", sev == "high"),
                "symptom_match":    round(cov * 100),
            })

        # Determine overall severity
        severity = _get_overall_severity(predictions)

        # Get all recommended doctors (deduped)
        all_doctors = []
        for p in predictions[:3]:
            for d in p.get("doctors", []):
                if d not in all_doctors:
                    all_doctors.append(d)

        # Get all medicines (deduped, top 3 predictions)
        all_medicines = []
        for p in predictions[:2]:
            for m in p.get("common_medicines", []):
                if m not in all_medicines:
                    all_medicines.append(m)

        # Emergency flag
        is_emergency = any(p.get("emergency", False) and p["probability"] > 30 for p in predictions[:2])

        return jsonify({
            "predictions": predictions,
            "overall_severity": severity,
            "recommended_doctors": all_doctors[:4],
            "recommended_medicines": all_medicines[:6],
            "is_emergency": is_emergency,
            "symptoms_analyzed": selected_symptoms,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _rule_based_predict(symptoms, age, gender):
    """Smarter rule-based prediction — common diseases ranked first for few symptoms."""
    symptom_set = set(s.lower() for s in symptoms)
    n = len(symptoms)

    # (disease, required_symptoms, bonus_symptoms, severity, common?)
    # common=True → gets priority when symptom count is low
    disease_rules = {
        # ── Very common, low-severity ──────────────────────────
        "Common Cold":          (["runny nose","cough","sore throat"],          ["headache","fatigue"],              "low",  True),
        "Allergic Rhinitis":    (["runny nose","sneezing"],                     ["itching","watery eyes"],           "low",  True),
        "Pharyngitis":          (["sore throat"],                               ["fever","headache"],                "low",  True),
        "Sinusitis":            (["sinus pressure","nasal congestion"],         ["headache","runny nose"],           "low",  True),
        "Acid Reflux / GERD":   (["acidity","heartburn"],                       ["chest pain","nausea"],             "low",  True),
        "Gastritis":            (["stomach pain","nausea"],                     ["vomiting","loss of appetite"],     "low",  True),
        "Tension Headache":     (["headache"],                                  ["neck stiffness","fatigue"],        "low",  True),
        "Viral Fever":          (["fever","fatigue"],                           ["chills","body aches"],             "low",  True),
        "Constipation":         (["constipation"],                              ["stomach pain","bloating"],         "low",  True),
        "Gastroenteritis":      (["nausea","vomiting","diarrhea"],              ["stomach pain","fever"],            "low",  True),
        "Food Poisoning":       (["vomiting","diarrhea","stomach pain"],        ["nausea","fever"],                  "low",  True),
        "UTI":                  (["frequent urination","burning urination"],    ["fever","flank pain"],              "low",  True),
        "Anemia":               (["fatigue","pallor"],                          ["dizziness","weakness"],            "medium",True),
        "Vitamin Deficiency":   (["fatigue","weakness"],                        ["bone pain","hair loss"],           "low",  True),
        "Dehydration":          (["dehydration","dizziness"],                   ["fatigue","dry mouth"],             "low",  True),
        "Migraine":             (["headache","sensitivity to light"],           ["nausea","dizziness"],              "medium",True),
        "Allergic Reaction":    (["itching","skin rash"],                       ["hives","swelling"],                "low",  True),
        "Fungal Infection":     (["itching","skin rash"],                       ["skin peeling"],                   "low",  True),
        "GERD":                 (["heartburn","acidity / heartburn"],           ["chest pain"],                     "low",  True),
        "IBS":                  (["stomach pain","bloating","diarrhea"],        ["constipation"],                   "low",  True),
        # ── Medium severity ───────────────────────────────────
        "Flu (Influenza)":      (["fever","cough","body aches"],                ["chills","fatigue","headache"],     "medium",True),
        "COVID-19":             (["fever","cough","loss of smell"],             ["fatigue","shortness of breath"],  "medium",False),
        "Malaria":              (["fever","chills","sweating"],                 ["headache","nausea","joint pain"], "high", False),
        "Dengue Fever":         (["fever","severe headache","joint pain"],      ["skin rash","nausea"],             "high", False),
        "Typhoid Fever":        (["fever","stomach pain","constipation"],       ["headache","loss of appetite"],    "high", False),
        "Bronchitis":           (["cough","phlegm"],                            ["shortness of breath","fatigue"],  "medium",True),
        "Pneumonia":            (["cough","fever","shortness of breath"],       ["chest pain","chills"],            "high", False),
        "Asthma":               (["shortness of breath","wheezing","cough"],   ["chest tightness"],                "high", False),
        "Hypertension":         (["high blood pressure","headache"],            ["dizziness","palpitations"],       "medium",False),
        "Diabetes":             (["excessive thirst","frequent urination"],     ["fatigue","blurred vision"],       "medium",False),
        "Anxiety Disorder":     (["anxiety","palpitations"],                    ["sleep problems","irritability"],  "medium",True),
        "Depression":           (["depression","sleep problems"],               ["fatigue","lack of concentration"],"medium",True),
        "Hypothyroidism":       (["fatigue","weight gain","cold intolerance"],  ["hair loss","constipation"],       "medium",False),
        # ── High severity — need more symptoms ────────────────
        "Heart Attack":         (["chest pain","shortness of breath","sweating"],["palpitations","arm pain"],      "high", False),
        "Tuberculosis":         (["coughing blood","night sweats","weight loss"],["low grade fever","fatigue"],    "high", False),
        "Meningitis":           (["neck stiffness","fever","severe headache"],  ["seizures","confusion"],          "high", False),
        "Stroke":               (["paralysis","slurred speech","confusion"],    ["severe headache","vision loss"], "high", False),
        "Appendicitis":         (["lower abdominal pain","fever","nausea"],     ["vomiting","loss of appetite"],   "high", False),
    }

    scores = {}
    for disease, (required, bonus, sev, common) in disease_rules.items():
        req_match  = sum(1 for s in required if any(s in sym.lower() for sym in symptom_set))
        bon_match  = sum(1 for s in bonus    if any(s in sym.lower() for sym in symptom_set))
        if req_match == 0:
            continue
        req_ratio  = req_match / len(required)
        bon_ratio  = bon_match / max(len(bonus), 1)
        base_score = (req_ratio * 70) + (bon_ratio * 30)

        # Penalise high-severity diseases when only 1-2 symptoms given
        if sev == "high" and n <= 2:
            base_score *= 0.3
        elif sev == "high" and n <= 3:
            base_score *= 0.5
        elif sev == "medium" and n <= 1:
            base_score *= 0.6

        # Boost common diseases
        if common:
            base_score *= 1.2

        scores[disease] = (base_score, sev)

    sorted_diseases = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:6]

    predictions = []
    for disease, (score, sev) in sorted_diseases:
        if score < 5:
            continue
        info = DISEASE_INFO.get(disease, {})
        predictions.append({
            "disease": disease,
            "probability": round(min(score, 88), 1),
            "severity": sev,
            "description": info.get("description", ""),
            "common_medicines": info.get("common_medicines", ["Consult a doctor"]),
            "doctors": info.get("doctors", ["General Physician"]),
            "prevention": info.get("prevention", []),
            "emergency": info.get("emergency", sev == "high"),
        })

    severity = _get_overall_severity(predictions)
    all_doctors = list({d for p in predictions[:3] for d in p.get("doctors", [])})
    all_medicines = list({m for p in predictions[:2] for m in p.get("common_medicines", [])})
    is_emergency = any(p.get("emergency", False) for p in predictions[:2])

    return jsonify({
        "predictions": predictions,
        "overall_severity": severity,
        "recommended_doctors": all_doctors[:4],
        "recommended_medicines": all_medicines[:6],
        "is_emergency": is_emergency,
        "symptoms_analyzed": symptoms,
        "model": "rule_based",
        "timestamp": datetime.now().isoformat(),
    })


def _get_overall_severity(predictions):
    if not predictions:
        return "low"
    top_pred = predictions[0]
    if top_pred.get("severity") == "high" and top_pred["probability"] > 40:
        return "high"
    if top_pred.get("severity") in ["high", "medium"] and top_pred["probability"] > 25:
        return "medium"
    return "low"


@app.route("/chat", methods=["POST"])
def chat():
    """
    Gemini AI chatbot for health questions.
    Body: { "message": "What should I do for fever?", "history": [...], "context": {...} }
    """
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        history = data.get("history", [])  # [{"role": "user/assistant", "content": "..."}]
        context = data.get("context", {})  # optional: current symptoms/predictions

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Build system prompt
        system_prompt = """You are an expert AI Health Assistant integrated into a medical app.
        
Your role:
- Provide accurate, empathetic, evidence-based health information
- Answer questions about symptoms, diseases, medications, diet, and wellness
- Always recommend consulting a qualified doctor for diagnosis/treatment
- For emergencies (chest pain, difficulty breathing, stroke symptoms), IMMEDIATELY advise calling 112/911
- Keep answers concise (3-5 sentences for simple questions, up to 10 for complex ones)
- Use bullet points for lists
- Do NOT diagnose definitively — say "may indicate" or "could suggest"

Format responses clearly with:
🔍 Key information first
💊 Medication notes (general info only, always say "consult doctor")  
✅ Actionable tips
⚠️ Warning signs to watch for
"""

        if context.get("symptoms"):
            system_prompt += f"\n\nUser's current symptoms: {', '.join(context['symptoms'])}"
        if context.get("top_disease"):
            system_prompt += f"\nAI predicted: {context['top_disease']} ({context.get('top_probability', '')}%)"

        # Build Groq messages (OpenAI-compatible format)
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-10:]:
            role = msg.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"
            messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": user_message})

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
        )

        reply = response.choices[0].message.content
        return jsonify({
            "reply": reply,
            "model": GROQ_MODEL,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/report", methods=["POST"])
def generate_report_data():
    """Generate structured health report data."""
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", [])
        predictions = data.get("predictions", [])
        user_name = data.get("user_name", "Patient")
        age = data.get("age", "N/A")
        gender = data.get("gender", "N/A")

        report = {
            "patient_name": user_name,
            "age": age,
            "gender": gender,
            "report_date": datetime.now().strftime("%d %B %Y, %I:%M %p"),
            "symptoms": symptoms,
            "predictions": predictions[:3],
            "disclaimer": (
                "This AI-generated report is for informational purposes only. "
                "It does NOT constitute a medical diagnosis. "
                "Please consult a qualified healthcare professional for proper medical advice and treatment."
            ),
            "emergency_note": (
                "If you experience severe symptoms such as chest pain, difficulty breathing, "
                "loss of consciousness, or stroke signs, call 112 (India) / 911 (US) IMMEDIATELY."
            ),
        }

        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 Health Assistant API running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
