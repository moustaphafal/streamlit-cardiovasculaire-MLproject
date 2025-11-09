import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Maladies Cardiovasculaires",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🫀 Système de Prédiction des Maladies Cardiovasculaires")
st.markdown("### Modèle de prédiction basé sur Régression Logistique")
st.markdown("---")

# Sidebar pour les informations
with st.sidebar:
    st.header("ℹ️ À propos")
    st.info(
        "Cette application utilise un modèle de **Régression Logistique** "
        "pour prédire le risque de maladies cardiovasculaires basé sur "
        "différents paramètres médicaux et résultats de tests d'effort."
    )
    st.markdown("---")
    st.header("📊 Instructions")
    st.markdown("""
    1. Remplissez les informations du patient
    2. Cliquez sur 'Prédire'
    3. Consultez les résultats
    """)
    st.markdown("---")
    st.header("🎯 Performance du Modèle")
    st.markdown("""
    - **Accuracy**: 88.6%
    - **F1-Score**: 90.0%
    - **Recall**: 92.2%
    - **ROC-AUC**: 93.4%
    """)

# Fonction pour charger le modèle et le scaler
@st.cache_resource
def load_model_and_scaler():
    """Charge le modèle ML et le scaler sauvegardés"""
    try:
        model_path = Path("models/heart_disease_model.pkl")
        scaler_path = Path("models/scaler.pkl")
        
        if model_path.exists() and scaler_path.exists():
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            with open(scaler_path, 'rb') as file:
                scaler = pickle.load(file)
            return model, scaler
        else:
            st.warning("⚠️ Modèle non trouvé. Veuillez entraîner et sauvegarder un modèle d'abord.")
            return None, None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None, None

# Interface de saisie des données
st.header("📝 Informations du Patient")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Données Démographiques")
    age = st.number_input("Âge (années)", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sexe", ["Homme", "Femme"])

with col2:
    st.subheader("Mesures Médicales")
    resting_bp = st.number_input("Pression artérielle au repos (mmHg)", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Cholestérol (mg/dL)", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox("Glycémie à jeun > 120 mg/dL", ["Non", "Oui"])
    max_hr = st.number_input("Fréquence cardiaque maximale", min_value=60, max_value=220, value=150)

with col3:
    st.subheader("Résultats Tests")
    chest_pain = st.selectbox("Type de douleur thoracique", 
                              ["ATA (Angine Atypique)", 
                               "NAP (Douleur Non-Angineuse)", 
                               "ASY (Asymptomatique)", 
                               "TA (Angine Typique)"])
    resting_ecg = st.selectbox("ECG au repos", 
                               ["Normal", "ST (Anomalie ST-T)", "LVH (Hypertrophie VG)"])
    exercise_angina = st.selectbox("Angine induite par l'exercice", ["Non", "Oui"])
    oldpeak = st.number_input("Oldpeak (Dépression ST)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
    st_slope = st.selectbox("Pente du segment ST", 
                           ["Up (Montante)", "Flat (Plate)", "Down (Descendante)"])

# Préparation des données
def prepare_input_data():
    """Convertit les entrées en format pour le modèle"""
    # Extraction des valeurs simples des selectbox
    chest_pain_map = {
        "ATA (Angine Atypique)": "ATA",
        "NAP (Douleur Non-Angineuse)": "NAP",
        "ASY (Asymptomatique)": "ASY",
        "TA (Angine Typique)": "TA"
    }
    
    resting_ecg_map = {
        "Normal": "Normal",
        "ST (Anomalie ST-T)": "ST",
        "LVH (Hypertrophie VG)": "LVH"
    }
    
    st_slope_map = {
        "Up (Montante)": "Up",
        "Flat (Plate)": "Flat",
        "Down (Descendante)": "Down"
    }
    
    # Création du DataFrame avec les données brutes
    data = {
        'Age': [age],
        'Sex': ['M' if sex == "Homme" else 'F'],
        'ChestPainType': [chest_pain_map[chest_pain]],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [1 if fasting_bs == "Oui" else 0],
        'RestingECG': [resting_ecg_map[resting_ecg]],
        'MaxHR': [max_hr],
        'ExerciseAngina': ['Y' if exercise_angina == "Oui" else 'N'],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_map[st_slope]]
    }
    
    df = pd.DataFrame(data)
    
    # Encodage des variables catégorielles (comme dans le notebook)
    df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)
    
    # Assurer que toutes les colonnes nécessaires existent
    required_columns = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    
    for col in required_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Réordonner les colonnes
    df_encoded = df_encoded[required_columns]
    
    return df_encoded

st.markdown("---")

# Bouton de prédiction
if st.button("🔮 Prédire le Risque Cardiovasculaire", type="primary", use_container_width=True):
    model, scaler = load_model_and_scaler()
    
    if model is not None and scaler is not None:
        # Préparation des données
        input_data = prepare_input_data()
        
        try:
            # Colonnes à standardiser (comme dans le notebook)
            col_standardise = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
            
            # Standardisation
            input_data[col_standardise] = scaler.transform(input_data[col_standardise])
            
            # Prédiction
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)
            
            st.markdown("---")
            st.header("📊 Résultats de la Prédiction")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction[0] == 1:
                    st.error("⚠️ RISQUE ÉLEVÉ de maladie cardiovasculaire détecté")
                    st.markdown("""
                    ### Recommandations:
                    - 🏥 **Consultez un cardiologue rapidement**
                    - 💊 Surveillez votre pression artérielle
                    - 🏃 Adoptez un mode de vie sain
                    - 📋 Suivez un traitement si prescrit
                    - 🥗 Adoptez une alimentation équilibrée
                    """)
                else:
                    st.success("✅ RISQUE FAIBLE de maladie cardiovasculaire")
                    st.markdown("""
                    ### Recommandations:
                    - 💪 Maintenez un mode de vie sain
                    - 📅 Faites des contrôles réguliers
                    - 🏃 Pratiquez une activité physique
                    - 🥗 Adoptez une alimentation équilibrée
                    - 😴 Dormez suffisamment
                    """)
            
            with col_res2:
                st.subheader("Probabilités")
                risk_low = probability[0][0] * 100
                risk_high = probability[0][1] * 100
                
                st.metric("Risque Faible", f"{risk_low:.1f}%")
                st.metric("Risque Élevé", f"{risk_high:.1f}%")
                
                # Visualisation
                chart_data = pd.DataFrame({
                    'Catégorie': ['Risque Faible', 'Risque Élevé'],
                    'Probabilité': [risk_low, risk_high]
                })
                st.bar_chart(chart_data.set_index('Catégorie'))
            
            # Affichage des facteurs de risque
            st.markdown("---")
            st.subheader("🔍 Facteurs de Risque Identifiés")
            
            risk_factors = []
            if oldpeak > 1.0:
                risk_factors.append("⚠️ Oldpeak élevé (dépression ST importante)")
            if max_hr < 120:
                risk_factors.append("⚠️ Fréquence cardiaque maximale faible")
            if exercise_angina == "Oui":
                risk_factors.append("⚠️ Angine induite par l'exercice")
            if "Flat" in st_slope:
                risk_factors.append("⚠️ Pente du segment ST plate")
            if age > 60:
                risk_factors.append("⚠️ Âge supérieur à 60 ans")
            if cholesterol > 240:
                risk_factors.append("⚠️ Cholestérol élevé")
                
            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.success("✅ Aucun facteur de risque majeur identifié")
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")
            st.error(f"Détails: {str(e)}")
    else:
        st.warning("⚠️ Veuillez d'abord entraîner et sauvegarder votre modèle en exécutant `python train_model.py`")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>💡 Cette application est à titre informatif uniquement. Consultez un professionnel de santé pour un diagnostic médical.</p>
    </div>
    """,
    unsafe_allow_html=True
)
