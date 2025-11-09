"""
Script d'entraînement du modèle de prédiction cardiovasculaire

Ce script est basé sur le notebook "Projet ML - Cardiovasculaire.ipynb"
Il utilise le dataset heart.csv et entraîne un modèle de Régression Logistique.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import pickle
from pathlib import Path

def preprocess_data(df):
    """
    Prétraite les données comme dans le notebook
    - Remplace les valeurs aberrantes par NaN
    - Remplit les valeurs manquantes avec la médiane
    """
    print("🔧 Prétraitement des données...")
    
    # Isoler les valeurs aberrantes
    df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df.loc[df['Oldpeak'] < 0, 'Oldpeak'] = np.nan
    
    # Remplacer par la médiane
    df['RestingBP'].fillna(df['RestingBP'].median(), inplace=True)
    df['Cholesterol'].fillna(df['Cholesterol'].median(), inplace=True)
    df['Oldpeak'].fillna(df['Oldpeak'].median(), inplace=True)
    
    print(f"  ✅ Valeurs aberrantes traitées")
    
    return df

def train_model(data_path='data/heart.csv'):
    """
    Entraîne le modèle de prédiction cardiovasculaire
    Suit exactement le processus du notebook
    """
    print("\n" + "="*50)
    print("🫀 ENTRAÎNEMENT DU MODÈLE CARDIOVASCULAIRE")
    print("="*50 + "\n")
    
    # 1. Chargement des données
    print("📊 Chargement des données...")
    try:
        df = pd.read_csv(data_path)
        print(f"  ✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    except FileNotFoundError:
        print(f"  ❌ Fichier {data_path} non trouvé.")
        return None, None
    
    # 2. Prétraitement
    df = preprocess_data(df)
    
    # 3. Encodage des variables catégorielles
    print("\n🔄 Encodage des variables catégorielles...")
    df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)
    print(f"  ✅ Encodage terminé: {df_encoded.shape[1]} colonnes")
    
    # 4. Séparation Features / Target
    print("\n🎯 Séparation des données...")
    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']
    
    # 5. Standardisation
    print("\n📏 Standardisation des données numériques...")
    col_standardise = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    scaler = StandardScaler()
    X[col_standardise] = scaler.fit_transform(X[col_standardise])
    print(f"  ✅ Standardisation appliquée sur {len(col_standardise)} colonnes")
    
    # 6. Split Train/Test
    print("\n✂️ Division des données...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"  ✅ Train: {X_train.shape[0]} échantillons")
    print(f"  ✅ Test:  {X_test.shape[0]} échantillons")
    
    # 7. Entraînement du modèle
    print("\n🎓 Entraînement du modèle Régression Logistique...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("  ✅ Entraînement terminé")
    
    # 8. Évaluation
    print("\n📈 Évaluation du modèle...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\n" + "="*50)
    print("✨ RÉSULTATS DU MODÈLE")
    print("="*50)
    print(f"  📊 Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  🎯 Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"  🔍 Recall:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"  ⚖️  F1-Score:  {f1:.3f} ({f1*100:.1f}%)")
    print(f"  📈 ROC-AUC:   {roc_auc:.3f} ({roc_auc*100:.1f}%)")
    print("="*50 + "\n")
    
    # 9. Sauvegarde du modèle et du scaler
    print("💾 Sauvegarde du modèle et du scaler...")
    
    # Créer le dossier models s'il n'existe pas
    Path("models").mkdir(exist_ok=True)
    
    # Sauvegarder le modèle
    with open('models/heart_disease_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("  ✅ Modèle sauvegardé: models/heart_disease_model.pkl")
    
    # Sauvegarder le scaler
    with open('models/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    print("  ✅ Scaler sauvegardé: models/scaler.pkl")
    
    print("\n" + "="*50)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*50 + "\n")
    print("Vous pouvez maintenant lancer l'application Streamlit:")
    print("  👉 streamlit run app.py\n")
    
    return model, scaler

if __name__ == "__main__":
    train_model()
