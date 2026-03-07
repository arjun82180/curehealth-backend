
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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
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

        # ══════════════════════════════════════════════════
        # IMPROVEMENT 1 — Symptom Weight Algorithm
        # Each symptom has importance weight per disease
        # Score = Sum(matched_weights) / total_possible_weight
        # ══════════════════════════════════════════════════
        SYMPTOM_WEIGHTS = {
            # Fever-related
            "Fever":0.8, "Chills":0.9, "Night Sweats":0.85, "Sweating":0.7,
            # Respiratory
            "Cough":0.75, "Shortness of Breath":0.9, "Wheezing":0.92,
            "Chest Tightness":0.85, "Phlegm":0.7, "Coughing Blood":0.95,
            # Neuro
            "Headache":0.6, "Neck Stiffness":0.95, "Seizures":0.98,
            "Confusion":0.9, "Paralysis":0.98, "Slurred Speech":0.97,
            # Cardiac
            "Chest Pain":0.92, "Palpitations":0.8, "Swollen Legs":0.8,
            "Cyanosis":0.95,
            # GI
            "Nausea":0.5, "Vomiting":0.65, "Diarrhea":0.7,
            "Bloody Stool":0.95, "Vomiting Blood":0.97, "Bloody Diarrhea":0.95,
            "Stomach Pain":0.65, "Yellowing Skin":0.92,
            # Urinary
            "Frequent Urination":0.8, "Burning Urination":0.85,
            "Blood in Urine":0.92, "Flank Pain":0.8,
            # Skin
            "Skin Rash":0.7, "Itching":0.5, "Blisters":0.85, "Petechiae":0.95,
            # General
            "Fatigue":0.4, "Weakness":0.45, "Weight Loss":0.75,
            "Loss of Appetite":0.5, "Joint Pain":0.65, "Muscle Pain":0.6,
            # High-specificity
            "Loss of Smell":0.88, "Loss of Taste":0.88,
        }
        DEFAULT_WEIGHT = 0.55

        n_symptoms    = len(selected_symptoms)
        disease_meta  = metadata.get("disease_metadata", {})
        sym_map       = metadata.get("user_symptom_map", {})

        # ══════════════════════════════════════════════════
        # IMPROVEMENT 3 — Location-based disease boost (India)
        # ══════════════════════════════════════════════════
        country = data.get("country", "India")
        LOCATION_BOOST = {}
        if country == "India":
            LOCATION_BOOST = {
                "Malaria": 12, "Dengue Fever": 12, "Typhoid Fever": 10,
                "Chikungunya": 10, "Leptospirosis": 8,
                "Influenza (Flu)": 5, "Tuberculosis (Pulmonary)": 8,
                "Kala-azar (Leishmaniasis)": 6, "Cholera": 5,
                "Viral Hepatitis A": 7, "Viral Hepatitis E": 7,
            }

        # ══════════════════════════════════════════════════
        # IMPROVEMENT 4 — Follow-up smart questions
        # ══════════════════════════════════════════════════
        FOLLOWUP_QUESTIONS = {
            "Malaria":           ["Have you been bitten by mosquitoes recently?",
                                  "Do you have cyclical (every 2-3 day) fever?",
                                  "Have you travelled to a forest or rural area?"],
            "Dengue Fever":      ["Do you have pain behind the eyes?",
                                  "Have you noticed red spots on your skin?",
                                  "Is there platelet drop in blood test?"],
            "Typhoid Fever":     ["How many days have you had fever (>5 days)?",
                                  "Do you have rose-colored spots on abdomen?",
                                  "Is your fever worse in the evening?"],
            "Tuberculosis (Pulmonary)":["Have you been coughing for more than 2 weeks?",
                                  "Do you have a TB-positive contact at home?",
                                  "Have you lost significant weight recently?"],
            "COVID-19":          ["Have you lost sense of smell or taste?",
                                  "Have you been in contact with a COVID patient?",
                                  "Do you have body aches and fatigue together?"],
            "Pneumonia (Bacterial)":["Is your cough producing yellow/green mucus?",
                                  "Do you feel sharp chest pain on breathing?",
                                  "Do you have high fever with shivering?"],
            "Asthma":            ["Do symptoms worsen at night or early morning?",
                                  "Do you have allergies (dust, pollen, pets)?",
                                  "Does exercise trigger your breathlessness?"],
            "Urinary Tract Infection":["Is there pain in lower abdomen or back?",
                                  "Is your urine cloudy or foul-smelling?",
                                  "Do you feel urgency to urinate frequently?"],
            "Appendicitis":      ["Does the pain start near navel and move to right side?",
                                  "Does the pain worsen on movement?",
                                  "Do you have fever with loss of appetite?"],
            "Heart Attack":      ["Is the pain spreading to left arm or jaw?",
                                  "Are you sweating profusely with chest pain?",
                                  "Do you have a history of heart disease?"],
        }

        # Build feature vector for ML
        features = metadata["features"]
        feature_vector = [1 if f in selected_symptoms else 0 for f in features]
        X = np.array([feature_vector])

        rf_proba  = rf_model.predict_proba(X)[0]
        gb_proba  = gb_model.predict_proba(X)[0]
        ensemble_proba = (0.6 * rf_proba + 0.4 * gb_proba)

        diseases    = label_encoder.classes_
        top_indices = np.argsort(ensemble_proba)[::-1][:20]

        try:
            from train_model import DISEASE_SYMPTOM_MAP as DSM
        except Exception:
            DSM = {}

        scored = []
        for idx in top_indices:
            disease  = diseases[idx]
            ml_prob  = float(ensemble_proba[idx])
            if ml_prob < 0.008:
                continue

            # ── Improvement 1: weighted coverage score ──
            disease_syms = DSM.get(disease, [])
            total_w = 0.0
            match_w = 0.0
            for sel in selected_symptoms:
                internal = sym_map.get(sel, [sel.lower().replace(" ","_")])
                w = SYMPTOM_WEIGHTS.get(sel, DEFAULT_WEIGHT)
                total_w += w
                if any(iv in disease_syms for iv in internal):
                    match_w += w
            cov = match_w / max(total_w, 0.01)

            meta = disease_meta.get(disease, {})
            sev  = meta.get("severity", "medium")

            # Severity penalty
            sev_penalty = 1.0
            if sev == "high"   and cov < 0.5 and n_symptoms <= 3:
                sev_penalty = 0.30
            elif sev == "high"  and cov < 0.4:
                sev_penalty = 0.50
            elif sev == "medium" and cov < 0.3:
                sev_penalty = 0.75
            if sev == "low" and cov >= 0.6:
                sev_penalty = 1.2

            final_score = ml_prob * cov * sev_penalty

            # ── Improvement 3: location boost ──
            loc_boost = LOCATION_BOOST.get(disease, 0) / 100.0
            final_score += loc_boost * ml_prob

            scored.append((disease, ml_prob, cov, final_score, sev))

        scored.sort(key=lambda x: x[3], reverse=True)

        predictions = []
        for disease, ml_prob, cov, final_score, sev in scored[:6]:
            if final_score < 0.004:
                continue
            display_prob = round(min((ml_prob * 0.45 + cov * 0.55) * 100, 91), 1)
            if display_prob < 6:
                continue
            info = DISEASE_INFO.get(disease, {})
            dm   = disease_meta.get(disease, {})
            predictions.append({
                "disease":          disease,
                "probability":      display_prob,
                "severity":         dm.get("severity", info.get("severity", "medium")),
                "description":      info.get("description", ""),
                "common_medicines": dm.get("medicines", info.get("common_medicines", ["Consult doctor"])),
                "doctors":          dm.get("doctors",   info.get("doctors", ["General Physician"])),
                "prevention":       info.get("prevention", []),
                "emergency":        info.get("emergency", sev == "high"),
                "symptom_match":    round(cov * 100),
                "followup_questions": FOLLOWUP_QUESTIONS.get(disease, []),
            })

        # ══════════════════════════════════════════════════
        # IMPROVEMENT 4 — Emergency severity detection
        # ══════════════════════════════════════════════════
        EMERGENCY_SYMPTOMS = {
            "Chest Pain", "Shortness of Breath", "Coughing Blood",
            "Vomiting Blood", "Bloody Stool", "Paralysis",
            "Slurred Speech", "Seizures", "Cyanosis", "Confusion",
        }
        has_emergency_symptom = bool(set(selected_symptoms) & EMERGENCY_SYMPTOMS)
        is_emergency = (
            has_emergency_symptom or
            any(p.get("emergency", False) and p["probability"] > 28 for p in predictions[:2])
        )

        severity = _get_overall_severity(predictions)
        if has_emergency_symptom:
            severity = "high"

        # ══════════════════════════════════════════════════
        # IMPROVEMENT 5 — Medicines only from TOP disease
        # ══════════════════════════════════════════════════
        all_doctors = []
        for p in predictions[:3]:
            for d in p.get("doctors", []):
                if d not in all_doctors:
                    all_doctors.append(d)

        # Only top disease medicines (not mixed from all)
        top_medicines = []
        if predictions:
            top_medicines = predictions[0].get("common_medicines", [])[:6]

        return jsonify({
            "predictions":           predictions,
            "overall_severity":      severity,
            "recommended_doctors":   all_doctors[:4],
            "recommended_medicines": top_medicines,
            "is_emergency":          is_emergency,
            "symptoms_analyzed":     selected_symptoms,
            "followup_questions":    predictions[0].get("followup_questions", []) if predictions else [],
            "top_disease":           predictions[0]["disease"] if predictions else "",
            "timestamp":             datetime.now().isoformat(),
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

    # Improvement 1: Symptom weights
    SYMPTOM_WEIGHTS = {
        "Fever":0.8,"Chills":0.9,"Sweating":0.7,"Cough":0.75,
        "Shortness of Breath":0.9,"Wheezing":0.92,"Chest Pain":0.92,
        "Neck Stiffness":0.95,"Seizures":0.98,"Headache":0.6,
        "Nausea":0.5,"Vomiting":0.65,"Diarrhea":0.7,
        "Bloody Stool":0.95,"Fatigue":0.4,"Weakness":0.45,
        "Joint Pain":0.65,"Skin Rash":0.7,"Itching":0.5,
        "Loss of Smell":0.88,"Loss of Taste":0.88,
    }
    DEF_W = 0.55

    # Improvement 3: India location boost
    country = getattr(gender, '__class__', None)  # placeholder
    INDIA_BOOST = {
        "Malaria":10,"Dengue Fever":10,"Typhoid Fever":8,
        "Flu (Influenza)":5,"Tuberculosis":7,
    }

    # Improvement 2: Follow-up questions
    FOLLOWUP = {
        "Malaria":           ["Have you been bitten by mosquitoes recently?",
                              "Do you have cyclical fever every 2-3 days?"],
        "Dengue Fever":      ["Do you have pain behind the eyes?",
                              "Have you noticed red spots on skin?"],
        "Typhoid Fever":     ["Have you had fever for more than 5 days?",
                              "Is your fever worse in the evening?"],
        "COVID-19":          ["Have you lost sense of smell or taste?",
                              "Have you been in contact with COVID patient?"],
        "Asthma":            ["Do symptoms worsen at night?",
                              "Do you have history of allergies?"],
        "UTI":               ["Is urine cloudy or foul-smelling?",
                              "Do you feel lower abdominal pain?"],
    }

    scores = {}
    for disease, (required, bonus, sev, common) in disease_rules.items():
        # Weighted matching
        req_w = sum(SYMPTOM_WEIGHTS.get(s.title(), DEF_W)
                    for s in required if any(s in sym.lower() for sym in symptom_set))
        bon_w = sum(SYMPTOM_WEIGHTS.get(s.title(), DEF_W)
                    for s in bonus    if any(s in sym.lower() for sym in symptom_set))
        total_req_w = sum(SYMPTOM_WEIGHTS.get(s.title(), DEF_W) for s in required)
        total_bon_w = sum(SYMPTOM_WEIGHTS.get(s.title(), DEF_W) for s in bonus) or 1.0

        if req_w == 0:
            continue

        req_ratio  = req_w / max(total_req_w, 0.01)
        bon_ratio  = bon_w / max(total_bon_w, 0.01)
        base_score = (req_ratio * 70) + (bon_ratio * 30)

        # Severity penalty
        if sev == "high" and n <= 2:
            base_score *= 0.3
        elif sev == "high" and n <= 3:
            base_score *= 0.5
        elif sev == "medium" and n <= 1:
            base_score *= 0.6

        if common:
            base_score *= 1.2

        # Location boost (India default)
        base_score += INDIA_BOOST.get(disease, 0) * req_ratio

        scores[disease] = (base_score, sev)

    sorted_diseases = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:6]

    # Emergency symptoms check (Improvement 4)
    EMERGENCY_SYMS = {"chest pain","shortness of breath","coughing blood",
                      "vomiting blood","bloody stool","paralysis","seizures","cyanosis"}
    has_emergency = bool(symptom_set & EMERGENCY_SYMS)

    predictions = []
    for disease, (score, sev) in sorted_diseases:
        if score < 5:
            continue
        info = DISEASE_INFO.get(disease, {})
        predictions.append({
            "disease":            disease,
            "probability":        round(min(score, 88), 1),
            "severity":           sev,
            "description":        info.get("description", ""),
            "common_medicines":   info.get("common_medicines", ["Consult a doctor"]),
            "doctors":            info.get("doctors", ["General Physician"]),
            "prevention":         info.get("prevention", []),
            "emergency":          info.get("emergency", sev == "high"),
            "followup_questions": FOLLOWUP.get(disease, []),
        })

    severity    = "high" if has_emergency else _get_overall_severity(predictions)
    all_doctors = list({d for p in predictions[:3] for d in p.get("doctors", [])})
    is_emergency = has_emergency or any(p.get("emergency") for p in predictions[:2])

    # Improvement 5: medicines only from top disease
    top_medicines = predictions[0].get("common_medicines", [])[:6] if predictions else []

    return jsonify({
        "predictions":           predictions,
        "overall_severity":      severity,
        "recommended_doctors":   all_doctors[:4],
        "recommended_medicines": top_medicines,
        "is_emergency":          is_emergency,
        "symptoms_analyzed":     symptoms,
        "followup_questions":    predictions[0].get("followup_questions", []) if predictions else [],
        "top_disease":           predictions[0]["disease"] if predictions else "",
        "model":                 "rule_based",
        "timestamp":             datetime.now().isoformat(),
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
