"""
AI Health Assistant - Disease Prediction Model Training
Uses Random Forest + Naive Bayes ensemble on symptom-disease dataset
Run this script once to train and save the model: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os

# ─────────────────────────────────────────────
# SYMPTOM-DISEASE DATASET
# 132 symptoms × 42 diseases (expanded)
# ─────────────────────────────────────────────

ALL_SYMPTOMS = [
    "itching","skin_rash","nodal_skin_eruptions","continuous_sneezing","shivering",
    "chills","joint_pain","stomach_pain","acidity","ulcers_on_tongue","muscle_wasting",
    "vomiting","burning_micturition","spotting_urination","fatigue","weight_gain",
    "anxiety","cold_hands_and_feets","mood_swings","weight_loss","restlessness",
    "lethargy","patches_in_throat","irregular_sugar_level","cough","high_fever",
    "sunken_eyes","breathlessness","sweating","dehydration","indigestion","headache",
    "yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes",
    "back_pain","constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine",
    "yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach",
    "swelled_lymph_nodes","malaise","blurred_and_distorted_vision","phlegm",
    "throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion",
    "chest_pain","weakness_in_limbs","fast_heart_rate","pain_during_bowel_movements",
    "pain_in_anal_region","bloody_stool","irritation_in_anus","neck_stiffness",
    "word_finding_difficulty","spinning_movements","loss_of_balance","unsteadiness",
    "weakness_of_one_body_side","loss_of_smell","bladder_discomfort","foul_smell_of_urine",
    "continuous_feel_of_urine","passage_of_gases","internal_itching","toxic_look_(typhos)",
    "depression","irritability","muscle_pain","altered_sensorium","red_spots_over_body",
    "belly_pain","abnormal_menstruation","dischromic_patches","watering_from_eyes",
    "increased_appetite","polyuria","family_history","mucoid_sputum","rusty_sputum",
    "lack_of_concentration","visual_disturbances","receiving_blood_transfusion",
    "receiving_unsterile_injections","coma","stomach_bleeding","distention_of_abdomen",
    "history_of_alcohol_consumption","fluid_overload_1","blood_in_sputum",
    "prominent_veins_on_calf","palpitations","painful_walking","pus_filled_pimples",
    "blackheads","scurring","skin_peeling","silver_like_dusting","small_dents_in_nails",
    "inflammatory_nails","blister","red_sore_around_nose","yellow_crust_ooze",
    "prognosis","muscle_weakness","stiff_neck","swelling_joints","movement_stiffness",
    "spinning_movements_2","loss_of_balance_2","unsteadiness_2","weakness_of_one_body_side_2",
    "slurred_speech","knee_pain","hip_joint_pain","swelling","bruising","obesity",
    "swollen_legs","swollen_blood_vessels","puffy_face_and_eyes","enlarged_thyroid",
    "brittle_nails","swollen_extremeties","excessive_hunger","extra_marital_contacts",
    "drying_and_tingling_lips","slurred_speech_2","knee_pain_2","hip_joint_pain_2",
    "loss_of_smell_2","throat_irritation_2","redness_of_eyes_2","sinus_pressure_2",
    "runny_nose_2","congestion_2"
]

# Simplified symptom list (user-friendly, maps to model features)
USER_SYMPTOMS = {
    "Fever": ["high_fever", "mild_fever"],
    "Headache": ["headache"],
    "Cough": ["cough", "mucoid_sputum", "rusty_sputum"],
    "Sore Throat": ["throat_irritation", "patches_in_throat"],
    "Runny Nose": ["runny_nose", "congestion"],
    "Fatigue": ["fatigue", "lethargy", "malaise"],
    "Body Aches": ["muscle_pain", "joint_pain", "back_pain"],
    "Nausea": ["nausea"],
    "Vomiting": ["vomiting"],
    "Diarrhea": ["diarrhoea"],
    "Chest Pain": ["chest_pain"],
    "Shortness of Breath": ["breathlessness"],
    "Dizziness": ["spinning_movements", "loss_of_balance", "unsteadiness"],
    "Rash": ["skin_rash", "red_spots_over_body", "blister"],
    "Itching": ["itching", "internal_itching"],
    "Stomach Pain": ["stomach_pain", "abdominal_pain", "belly_pain"],
    "Loss of Appetite": ["loss_of_appetite"],
    "Chills": ["chills", "shivering"],
    "Sweating": ["sweating"],
    "Blurred Vision": ["blurred_and_distorted_vision", "visual_disturbances"],
    "Ear Pain": ["pain_behind_the_eyes"],
    "Weight Loss": ["weight_loss"],
    "Swelling": ["swelling", "swelling_joints"],
    "Palpitations": ["palpitations", "fast_heart_rate"],
    "Anxiety": ["anxiety", "restlessness"],
    "Constipation": ["constipation"],
    "Dark Urine": ["dark_urine", "yellow_urine"],
    "Yellowing Skin": ["yellowish_skin", "yellowing_of_eyes"],
    "Neck Stiffness": ["stiff_neck", "neck_stiffness"],
    "Acidity": ["acidity", "indigestion"],
    "Dehydration": ["dehydration", "sunken_eyes"],
    "Frequent Urination": ["polyuria", "continuous_feel_of_urine"],
    "Muscle Weakness": ["muscle_weakness", "weakness_in_limbs"],
    "Joint Stiffness": ["movement_stiffness", "swelling_joints"],
    "Skin Peeling": ["skin_peeling", "silver_like_dusting"],
    "Depression": ["depression", "mood_swings", "irritability"],
}

# Disease → symptom mapping for synthetic dataset generation
DISEASE_SYMPTOM_MAP = {
    "Flu (Influenza)": [
        "high_fever","cough","headache","muscle_pain","fatigue","chills","shivering",
        "sore_throat","runny_nose","loss_of_appetite"
    ],
    "COVID-19": [
        "high_fever","cough","fatigue","loss_of_smell","breathlessness","headache",
        "muscle_pain","chills","throat_irritation","body_aches"
    ],
    "Common Cold": [
        "runny_nose","congestion","throat_irritation","mild_fever","cough","sneezing",
        "headache","fatigue","watering_from_eyes"
    ],
    "Malaria": [
        "high_fever","chills","shivering","headache","sweating","nausea","vomiting",
        "muscle_pain","fatigue","malaise"
    ],
    "Dengue Fever": [
        "high_fever","headache","joint_pain","muscle_pain","skin_rash","pain_behind_the_eyes",
        "nausea","vomiting","fatigue","red_spots_over_body"
    ],
    "Typhoid": [
        "high_fever","headache","stomach_pain","constipation","fatigue","loss_of_appetite",
        "nausea","vomiting","abdominal_pain","toxic_look_(typhos)"
    ],
    "Pneumonia": [
        "cough","high_fever","breathlessness","chest_pain","fatigue","chills",
        "rusty_sputum","mucoid_sputum","fast_heart_rate","sweating"
    ],
    "Tuberculosis": [
        "cough","blood_in_sputum","weight_loss","fatigue","high_fever","sweating",
        "chest_pain","breathlessness","loss_of_appetite","malaise"
    ],
    "Diabetes": [
        "polyuria","increased_appetite","weight_loss","fatigue","blurred_and_distorted_vision",
        "drying_and_tingling_lips","irregular_sugar_level","excessive_hunger","weakness_in_limbs"
    ],
    "Hypertension": [
        "headache","chest_pain","breathlessness","dizziness","palpitations","fast_heart_rate",
        "fatigue","blurred_and_distorted_vision","sweating"
    ],
    "Gastroenteritis": [
        "nausea","vomiting","diarrhoea","stomach_pain","high_fever","dehydration",
        "loss_of_appetite","fatigue","abdominal_pain","chills"
    ],
    "Food Poisoning": [
        "nausea","vomiting","diarrhoea","stomach_pain","mild_fever","fatigue",
        "dehydration","loss_of_appetite","chills","sweating"
    ],
    "Migraine": [
        "headache","nausea","vomiting","blurred_and_distorted_vision","fatigue",
        "visual_disturbances","spinning_movements","loss_of_balance","sensitivity_to_light"
    ],
    "Asthma": [
        "breathlessness","cough","chest_pain","fast_heart_rate","fatigue","anxiety",
        "phlegm","restlessness","mucoid_sputum"
    ],
    "Bronchitis": [
        "cough","phlegm","breathlessness","chest_pain","mild_fever","fatigue",
        "throat_irritation","mucoid_sputum","sinus_pressure"
    ],
    "Sinusitis": [
        "sinus_pressure","headache","runny_nose","congestion","throat_irritation",
        "mild_fever","fatigue","redness_of_eyes","watering_from_eyes"
    ],
    "Urinary Tract Infection": [
        "burning_micturition","foul_smell_of_urine","continuous_feel_of_urine",
        "bladder_discomfort","mild_fever","fatigue","abdominal_pain","dark_urine"
    ],
    "Jaundice": [
        "yellowish_skin","yellowing_of_eyes","dark_urine","fatigue","abdominal_pain",
        "nausea","loss_of_appetite","itching","high_fever","vomiting"
    ],
    "Hepatitis B": [
        "yellowish_skin","yellowing_of_eyes","dark_urine","fatigue","abdominal_pain",
        "nausea","loss_of_appetite","joint_pain","vomiting","acute_liver_failure"
    ],
    "Hepatitis C": [
        "fatigue","nausea","vomiting","dark_urine","abdominal_pain","yellowish_skin",
        "loss_of_appetite","joint_pain","muscle_pain"
    ],
    "Chickenpox": [
        "skin_rash","itching","high_fever","headache","fatigue","loss_of_appetite",
        "blister","red_spots_over_body","chills","sweating"
    ],
    "Measles": [
        "high_fever","skin_rash","runny_nose","cough","redness_of_eyes","headache",
        "fatigue","loss_of_appetite","watering_from_eyes"
    ],
    "Psoriasis": [
        "skin_rash","itching","skin_peeling","silver_like_dusting","small_dents_in_nails",
        "inflammatory_nails","joint_pain","red_spots_over_body"
    ],
    "Fungal Infection": [
        "itching","skin_rash","nodal_skin_eruptions","dischromic_patches","fatigue",
        "yellow_crust_ooze","red_sore_around_nose","skin_peeling"
    ],
    "Allergy": [
        "itching","runny_nose","redness_of_eyes","watering_from_eyes","sneezing",
        "congestion","skin_rash","throat_irritation","breathlessness"
    ],
    "Arthritis": [
        "joint_pain","swelling_joints","movement_stiffness","muscle_weakness",
        "fatigue","back_pain","knee_pain","hip_joint_pain","painful_walking"
    ],
    "Hypothyroidism": [
        "fatigue","weight_gain","cold_hands_and_feets","brittle_nails","enlarged_thyroid",
        "mood_swings","constipation","depression","puffy_face_and_eyes","lethargy"
    ],
    "Hyperthyroidism": [
        "weight_loss","fast_heart_rate","anxiety","sweating","palpitations","irritability",
        "restlessness","increased_appetite","fatigue","breathlessness"
    ],
    "Anaemia": [
        "fatigue","weakness_in_limbs","breathlessness","fast_heart_rate","cold_hands_and_feets",
        "headache","dizziness","lethargy","muscle_weakness","pallor"
    ],
    "Heart Disease": [
        "chest_pain","breathlessness","palpitations","fast_heart_rate","fatigue",
        "sweating","nausea","dizziness","swollen_legs","weakness_in_limbs"
    ],
    "Kidney Disease": [
        "fatigue","swelling","dark_urine","breathlessness","nausea","loss_of_appetite",
        "muscle_weakness","frequent_urination","back_pain","headache"
    ],
    "Appendicitis": [
        "abdominal_pain","nausea","vomiting","high_fever","loss_of_appetite",
        "constipation","fatigue","chills","sweating"
    ],
    "Peptic Ulcer": [
        "stomach_pain","acidity","nausea","vomiting","loss_of_appetite","bloody_stool",
        "burning_micturition","fatigue","weight_loss"
    ],
    "Dengue Hemorrhagic Fever": [
        "high_fever","skin_rash","red_spots_over_body","bloody_stool","joint_pain",
        "muscle_pain","nausea","vomiting","fatigue","bleeding"
    ],
    "Meningitis": [
        "stiff_neck","high_fever","headache","nausea","vomiting","altered_sensorium",
        "loss_of_balance","spinning_movements","redness_of_eyes","chills"
    ],
    "Stroke": [
        "weakness_of_one_body_side","slurred_speech","loss_of_balance","unsteadiness",
        "headache","blurred_and_distorted_vision","nausea","altered_sensorium"
    ],
    "Acne": [
        "pus_filled_pimples","blackheads","scurring","skin_rash","itching","redness",
        "skin_peeling","nodal_skin_eruptions"
    ],
    "Varicose Veins": [
        "swollen_blood_vessels","prominent_veins_on_calf","swelling","painful_walking",
        "fatigue","muscle_pain","swollen_legs","bruising"
    ],
    "Hypoglycemia": [
        "anxiety","restlessness","sweating","palpitations","fatigue","headache",
        "blurred_and_distorted_vision","irritability","weakness_in_limbs"
    ],
    "Panic Disorder": [
        "anxiety","palpitations","chest_pain","breathlessness","sweating","chills",
        "dizziness","fatigue","restlessness","fast_heart_rate"
    ],
    "Depression": [
        "depression","fatigue","loss_of_appetite","weight_loss","mood_swings",
        "irritability","lack_of_concentration","lethargy","restlessness","sleep_problems"
    ],
    "Vitamin D Deficiency": [
        "bone_pain","muscle_weakness","fatigue","depression","back_pain",
        "impaired_wound_healing","hair_loss","body_aches","lethargy"
    ],
}

def generate_dataset(n_samples=300):
    """Generate synthetic training dataset from disease-symptom mappings."""
    feature_cols = list(USER_SYMPTOMS.keys())
    rows = []
    labels = []
    
    for disease, disease_symptoms in DISEASE_SYMPTOM_MAP.items():
        for _ in range(n_samples):
            row = {}
            for user_sym, model_syms in USER_SYMPTOMS.items():
                # Check if any model symptom is in disease symptoms
                base_match = any(s in disease_symptoms for s in model_syms)
                if base_match:
                    # High probability of having this symptom if it's a disease symptom
                    row[user_sym] = 1 if np.random.random() > 0.2 else 0
                else:
                    # Low probability (noise/comorbidity)
                    row[user_sym] = 1 if np.random.random() < 0.08 else 0
            rows.append(row)
            labels.append(disease)
    
    df = pd.DataFrame(rows, columns=feature_cols)
    return df, labels

def train_and_save():
    print("🔬 Generating training dataset...")
    X, y = generate_dataset(n_samples=200)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("🌲 Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    print("📊 Training Gradient Boosting model...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    # Evaluate
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    gb_acc = accuracy_score(y_test, gb_model.predict(X_test))
    print(f"\n✅ Random Forest Accuracy: {rf_acc:.2%}")
    print(f"✅ Gradient Boosting Accuracy: {gb_acc:.2%}")
    
    # Save models and metadata
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/rf_model.pkl")
    joblib.dump(gb_model, "models/gb_model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    
    # Save feature names and disease info
    with open("models/metadata.json", "w") as f:
        json.dump({
            "features": list(USER_SYMPTOMS.keys()),
            "diseases": le.classes_.tolist(),
            "rf_accuracy": rf_acc,
            "gb_accuracy": gb_acc,
            "user_symptom_map": USER_SYMPTOMS,
        }, f, indent=2)
    
    print("\n💾 Models saved to models/ directory")
    print(f"📋 {len(DISEASE_SYMPTOM_MAP)} diseases | {len(USER_SYMPTOMS)} symptom features")
    return rf_model, gb_model, le

if __name__ == "__main__":
    train_and_save()
