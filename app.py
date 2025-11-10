import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

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

# Création des onglets
tab1, tab2 = st.tabs(["🔮 Prédiction", "📊 Analyse & Modélisation"])

# ==================== ONGLET 1: PRÉDICTION ====================
with tab1:
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

# ==================== ONGLET 2: ANALYSE & MODÉLISATION ====================
with tab2:
    st.header("📊 Exploration des Données et Choix du Modèle")
    
    # Chargement des données
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv("data/heart.csv")
            return df
        except:
            st.error("⚠️ Fichier heart.csv non trouvé dans le dossier data/")
            return None
    
    df_original = load_data()
    
    if df_original is not None:
        # Section 1: Aperçu des données
        st.subheader("1️⃣ Aperçu du Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nombre de patients", df_original.shape[0])
        with col2:
            st.metric("Nombre de variables", df_original.shape[1])
        with col3:
            st.metric("Cas positifs", df_original['HeartDisease'].sum())
        with col4:
            st.metric("Cas négatifs", (df_original['HeartDisease'] == 0).sum())
        
        with st.expander("🔍 Voir les premières lignes du dataset"):
            st.dataframe(df_original.head(10))
        
        with st.expander("📋 Informations sur les colonnes"):
            st.write(df_original.describe())
        
        # Section 2: Prétraitement
        st.markdown("---")
        st.subheader("2️⃣ Prétraitement des Données")
        
        st.write("**Valeurs aberrantes détectées:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RestingBP = 0", (df_original['RestingBP'] == 0).sum())
        with col2:
            st.metric("Cholesterol = 0", (df_original['Cholesterol'] == 0).sum())
        with col3:
            st.metric("Oldpeak < 0", (df_original['Oldpeak'] < 0).sum())
        
        st.info("✅ Ces valeurs aberrantes sont remplacées par la médiane de la variable correspondante")
        
        # Appliquer le prétraitement
        df = df_original.copy()
        df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
        df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
        df.loc[df['Oldpeak'] < 0, 'Oldpeak'] = np.nan
        
        df['RestingBP'] = df['RestingBP'].fillna(df['RestingBP'].median())
        df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].median())
        df['Oldpeak'] = df['Oldpeak'].fillna(df['Oldpeak'].median())
        
        # Section 3: Corrélations
        st.markdown("---")
        st.subheader("3️⃣ Matrice de Corrélation")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        correlation_matrix = df.select_dtypes(include='number').corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        plt.title("Corrélations entre variables quantitatives")
        st.pyplot(fig)
        
        st.markdown("""
        **Observations clés:**
        - **Oldpeak** et **HeartDisease**: Corrélation positive modérée (0.42)
        - **MaxHR** et **HeartDisease**: Corrélation négative modérée (-0.40)
        - **FastingBS** et **HeartDisease**: Corrélation positive faible (0.27)
        """)
        
        # Section 4: Encodage
        st.markdown("---")
        st.subheader("4️⃣ Encodage et Standardisation")
        
        st.write("**Variables catégorielles encodées:**")
        st.write("- Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope")
        st.write("- Méthode: One-Hot Encoding (drop_first=True)")
        
        # Encodage
        df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)
        
        st.write(f"**Nombre de colonnes après encodage:** {df_encoded.shape[1]}")
        
        # Corrélation après encodage
        with st.expander("🔍 Voir les corrélations après encodage"):
            corr_with_target = df_encoded.corr()['HeartDisease'].sort_values(ascending=False)
            st.write(corr_with_target)
        
        # Section 5: Comparaison des modèles
        st.markdown("---")
        st.subheader("5️⃣ Comparaison des Modèles")
        
        if st.button("🎓 Entraîner et Comparer les Modèles"):
            with st.spinner("Entraînement en cours..."):
                # Préparation des données
                X = df_encoded.drop('HeartDisease', axis=1)
                y = df_encoded['HeartDisease']
                
                # Standardisation
                col_standardise = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
                scaler = StandardScaler()
                X[col_standardise] = scaler.fit_transform(X[col_standardise])
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                # Entraînement des modèles
                models = {
                    "Régression Logistique": LogisticRegression(max_iter=1000, random_state=42),
                    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
                }
                
                results = []
                models_dict = {}  # Pour stocker les modèles entraînés
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                    
                    # Stocker le modèle et les prédictions
                    models_dict[name] = {
                        'model': model,
                        'y_pred': y_pred,
                        'y_proba': y_proba
                    }
                    
                    results.append({
                        "Modèle": name,
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred),
                        "Recall": recall_score(y_test, y_pred),
                        "F1-Score": f1_score(y_test, y_pred),
                        "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else 0
                    })
                
                # Affichage des résultats
                comparison_df = pd.DataFrame(results)
                st.success("✅ Entraînement terminé!")
                
                st.write("### 📊 Tableau Comparatif")
                st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']))
                
                # Visualisation
                st.write("### 📈 Comparaison Visuelle (F1-Score)")
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(comparison_df['Modèle'], comparison_df['F1-Score'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ax.set_ylabel('F1-Score')
                ax.set_title('Comparaison des Modèles - F1-Score')
                ax.set_ylim(0.8, 0.92)
                
                # Ajouter les valeurs sur les barres
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Courbe ROC-AUC
                st.write("### 📈 Courbes ROC (Receiver Operating Characteristic)")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Couleurs pour chaque modèle
                colors = {'Régression Logistique': '#FF6B6B', 
                         'Random Forest': '#4ECDC4', 
                         'SVM': '#45B7D1'}
                
                # Tracer la courbe ROC pour chaque modèle
                for name, data in models_dict.items():
                    if data['y_proba'] is not None:
                        fpr, tpr, _ = roc_curve(y_test, data['y_proba'])
                        roc_auc = roc_auc_score(y_test, data['y_proba'])
                        ax.plot(fpr, tpr, color=colors[name], lw=2, 
                               label=f'{name} (AUC = {roc_auc:.3f})')
                
                # Ligne de référence (classificateur aléatoire)
                ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Aléatoire (AUC = 0.500)')
                
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
                ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
                ax.set_title('Courbes ROC - Comparaison des Modèles', fontsize=14, fontweight='bold')
                ax.legend(loc="lower right", fontsize=10)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                st.info("""
                💡 **Interprétation de la courbe ROC:**
                - Plus la courbe est proche du coin supérieur gauche, meilleur est le modèle
                - L'AUC (Area Under Curve) varie de 0.5 (aléatoire) à 1.0 (parfait)
                - Un AUC > 0.9 indique une excellente capacité discriminante
                - Nos modèles ont tous un AUC > 0.93, ce qui est excellent !
                """)
                
                # Conclusion
                st.write("### ✨ Conclusion")
                best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Modèle']
                st.success(f"""
                **Modèle choisi: {best_model}**
                
                La **Régression Logistique** a été sélectionnée pour les raisons suivantes:
                - ✅ Meilleur **Recall (92.2%)**: crucial pour détecter les vrais cas de maladie
                - ✅ Excellent **F1-Score (90.0%)**: bon équilibre précision/rappel
                - ✅ Très bon **ROC-AUC (93.4%)**: excellente capacité discriminante
                - ✅ **Interprétabilité**: permet de comprendre l'impact de chaque variable
                - ✅ **Simplicité**: moins de risque de surapprentissage
                
                En médecine, minimiser les **faux négatifs** est prioritaire, d'où l'importance du Recall élevé.
                """)
        
        # Section 6: Variables importantes
        st.markdown("---")
        st.subheader("6️⃣ Variables les Plus Importantes")
        
        st.write("### Top 5 des Indicateurs de Maladie Cardiovasculaire")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Facteurs de risque positifs:**")
            st.markdown("""
            1. 🔴 **ST_Slope_Flat** (0.554)
            2. 🔴 **ExerciseAngina_Y** (0.494)
            3. 🔴 **Oldpeak** (0.425)
            4. 🔴 **Sex_M** (0.305)
            5. 🔴 **Age** (0.282)
            """)
        
        with col2:
            st.write("**Facteurs protecteurs:**")
            st.markdown("""
            1. 🟢 **ST_Slope_Up** (-0.622)
            2. 🟢 **ChestPainType_ATA** (-0.426)
            3. 🟢 **MaxHR** (-0.403)
            4. 🟢 **ChestPainType_NAP** (-0.267)
            5. 🟢 **ChestPainType_TA** (-0.101)
            """)
        
        st.info("""
        💡 **Interprétation:**
        Les tests d'effort (pente ST, angine d'effort, Oldpeak) sont les prédicteurs les plus puissants 
        de maladie cardiovasculaire, surpassant les facteurs physiologiques de base.
        """)
    
    else:
        st.warning("Veuillez ajouter le fichier heart.csv dans le dossier data/")

