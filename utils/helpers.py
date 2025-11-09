"""
Utilitaires pour le projet de prédiction cardiovasculaire
Basé sur le dataset heart.csv et le notebook "Projet ML - Cardiovasculaire.ipynb"
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_heart_data(df):
    """
    Prétraite les données du dataset heart.csv
    - Remplace les valeurs aberrantes (0 pour RestingBP et Cholesterol, valeurs négatives pour Oldpeak)
    - Remplit avec la médiane
    
    Args:
        df: DataFrame avec les données brutes
        
    Returns:
        DataFrame prétraité
    """
    df_clean = df.copy()
    
    # Isoler les valeurs aberrantes
    df_clean['RestingBP'] = df_clean['RestingBP'].replace(0, np.nan)
    df_clean['Cholesterol'] = df_clean['Cholesterol'].replace(0, np.nan)
    df_clean.loc[df_clean['Oldpeak'] < 0, 'Oldpeak'] = np.nan
    
    # Remplacer par la médiane
    df_clean['RestingBP'].fillna(df_clean['RestingBP'].median(), inplace=True)
    df_clean['Cholesterol'].fillna(df_clean['Cholesterol'].median(), inplace=True)
    df_clean['Oldpeak'].fillna(df_clean['Oldpeak'].median(), inplace=True)
    
    return df_clean

def encode_categorical_features(df):
    """
    Encode les variables catégorielles avec One-Hot Encoding
    
    Args:
        df: DataFrame avec les données
        
    Returns:
        DataFrame encodé
    """
    df_encoded = pd.get_dummies(
        df, 
        columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], 
        drop_first=True
    )
    return df_encoded

def save_model(model, filepath='models/heart_disease_model.pkl'):
    """
    Sauvegarde le modèle entraîné
    
    Args:
        model: Modèle entraîné
        filepath: Chemin du fichier de sauvegarde
    """
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f"✅ Modèle sauvegardé dans {filepath}")

def load_model(filepath='models/heart_disease_model.pkl'):
    """
    Charge un modèle sauvegardé
    
    Args:
        filepath: Chemin du fichier du modèle
        
    Returns:
        Modèle chargé
    """
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

def save_scaler(scaler, filepath='models/scaler.pkl'):
    """
    Sauvegarde le scaler
    
    Args:
        scaler: StandardScaler entraîné
        filepath: Chemin du fichier de sauvegarde
    """
    with open(filepath, 'wb') as file:
        pickle.dump(scaler, file)
    print(f"✅ Scaler sauvegardé dans {filepath}")

def load_scaler(filepath='models/scaler.pkl'):
    """
    Charge un scaler sauvegardé
    
    Args:
        filepath: Chemin du fichier du scaler
        
    Returns:
        Scaler chargé
    """
    with open(filepath, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle
    
    Args:
        model: Modèle entraîné
        X_test: Données de test
        y_test: Labels de test
        
    Returns:
        Dictionnaire avec les métriques
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    return metrics

def get_feature_importance(model, feature_names):
    """
    Récupère l'importance des features pour le modèle
    
    Args:
        model: Modèle entraîné
        feature_names: Liste des noms de features
        
    Returns:
        DataFrame avec les features et leur importance
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        return importance_df
    elif hasattr(model, 'coef_'):
        # Pour la régression logistique
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', ascending=False, key=abs)
        return importance_df
    else:
        return None
