# 🫀 Système de Prédiction des Maladies Cardiovasculaires

Application Streamlit pour prédire le risque de maladies cardiovasculaires à l'aide d'un modèle de **Régression Logistique**.

## 📋 Description

Cette application web permet de prédire le risque de maladies cardiovasculaires d'un patient basé sur différents paramètres médicaux, démographiques et résultats de tests d'effort. Elle utilise un modèle de **Régression Logistique** entraîné sur le dataset `heart.csv`.

### 🎯 Performance du Modèle

- **Accuracy**: 88.6%
- **Precision**: Élevée
- **Recall**: 92.2% (excellent pour détecter les cas positifs)
- **F1-Score**: 90.0%
- **ROC-AUC**: 93.4%

## ✨ Fonctionnalités

- 📊 Interface utilisateur intuitive avec Streamlit
- 🎯 Prédiction en temps réel du risque cardiovasculaire
- 📈 Affichage des probabilités et visualisations
- 🔍 Identification des facteurs de risque individuels
- 💡 Recommandations personnalisées selon le résultat
- 🔧 Structure modulaire basée sur le notebook d'analyse

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de packages Python)

### Étapes d'installation

1. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

## 📂 Structure du Projet

```
streamlit-Cardiovasculaire-MLproject/
│
├── app.py                                    # Application Streamlit principale
├── train_model.py                            # Script d'entraînement du modèle
├── requirements.txt                          # Dépendances Python
├── README.md                                 # Documentation
├── Projet ML - Cardiovasculaire.ipynb       # Notebook d'analyse original
│
├── models/                                   # Modèles entraînés
│   ├── heart_disease_model.pkl              # Modèle de Régression Logistique
│   └── scaler.pkl                           # StandardScaler pour normalisation
│
├── data/                                     # Données
│   └── heart.csv                            # Dataset cardiovasculaire
│
├── utils/                                    # Fonctions utilitaires
│   └── helpers.py                           # Fonctions d'aide et prétraitement
│
└── .streamlit/                              # Configuration Streamlit
    └── config.toml
```

## 🎓 Utilisation

### 1. Le Dataset heart.csv

Le fichier `heart.csv` contient **918 patients** avec les colonnes suivantes :

**Données démographiques:**
- `Age` : Âge du patient (années)
- `Sex` : Sexe (M=Homme, F=Femme)

**Mesures médicales:**
- `RestingBP` : Pression artérielle au repos (mmHg)
- `Cholesterol` : Cholestérol sérique (mg/dL)
- `FastingBS` : Glycémie à jeun > 120 mg/dL (1=Oui, 0=Non)
- `MaxHR` : Fréquence cardiaque maximale atteinte

**Résultats de tests:**
- `ChestPainType` : Type de douleur thoracique (TA, ATA, NAP, ASY)
- `RestingECG` : Résultats ECG au repos (Normal, ST, LVH)
- `ExerciseAngina` : Angine induite par l'exercice (Y=Oui, N=Non)
- `Oldpeak` : Dépression du segment ST
- `ST_Slope` : Pente du segment ST (Up, Flat, Down)

**Variable cible:**
- `HeartDisease` : Maladie cardiovasculaire (1=Oui, 0=Non)

### 2. Entraîner le modèle

Le dataset `heart.csv` est déjà inclus dans le dossier `data/`. Pour entraîner le modèle :

```bash
python train_model.py
```

Ce script va :
- Charger les données de `data/heart.csv`
- Nettoyer les valeurs aberrantes (RestingBP=0, Cholesterol=0, Oldpeak<0)
- Encoder les variables catégorielles (One-Hot Encoding)
- Standardiser les variables numériques
- Entraîner un modèle de Régression Logistique
- Évaluer ses performances
- Sauvegarder le modèle et le scaler dans `models/`

### 3. Lancer l'application Streamlit

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse `http://localhost:8501`

### 4. Utiliser l'interface

1. **Remplissez les informations du patient** dans les trois colonnes :
   - Données démographiques (âge, sexe)
   - Mesures médicales (pression artérielle, cholestérol, glycémie, fréquence cardiaque)
   - Résultats de tests (type de douleur thoracique, ECG, angine, Oldpeak, pente ST)

2. **Cliquez sur "🔮 Prédire le Risque Cardiovasculaire"**

3. **Consultez les résultats** :
   - Prédiction (Risque Élevé ou Risque Faible)
   - Probabilités détaillées
   - Visualisation en graphique
   - Facteurs de risque identifiés
   - Recommandations personnalisées

## 🔬 Analyse des Données

Le notebook `Projet ML - Cardiovasculaire.ipynb` contient l'analyse complète :

### Prétraitement
- Détection et remplacement des valeurs aberrantes
- Analyse de corrélation
- Visualisations exploratoires

### Encodage
- One-Hot Encoding pour les variables catégorielles
- Standardisation (StandardScaler) pour les variables numériques

### Modélisation
Trois modèles ont été testés :
- ✅ **Régression Logistique** (CHOISI - meilleures performances)
- Random Forest
- SVM

### Pourquoi la Régression Logistique ?

La régression logistique a été choisie car elle présente :
- Le meilleur **Recall (92.2%)** : crucial pour minimiser les faux négatifs en médecine
- Excellente **Accuracy (88.6%)**
- Très bon **F1-Score (90.0%)**
- **Interprétabilité** : on peut comprendre l'impact de chaque variable
- **Rapidité** : prédictions instantanées

## 📊 Variables les Plus Importantes

Selon l'analyse de corrélation :

**Indicateurs positifs de maladie :**
1. 🔴 **ST_Slope_Flat** (0.554) - Pente ST plate
2. 🔴 **ExerciseAngina_Y** (0.494) - Angine à l'effort
3. 🔴 **Oldpeak** (0.425) - Dépression ST élevée

**Indicateurs protecteurs :**
1. 🟢 **ST_Slope_Up** (-0.622) - Pente ST montante
2. 🟢 **MaxHR** (-0.403) - Fréquence cardiaque max élevée
3. 🟢 **ChestPainType_ATA** (-0.426) - Douleur atypique

## 🛠️ Personnalisation

### Modifier les paramètres du modèle

Dans `train_model.py`, vous pouvez ajuster :

```python
# Changer les hyperparamètres de la régression logistique
model = LogisticRegression(
    max_iter=1000,      # Nombre d'itérations
    random_state=42,    # Reproductibilité
    C=1.0,             # Paramètre de régularisation
    solver='lbfgs'     # Algorithme d'optimisation
)
```

### Tester d'autres modèles

Vous pouvez facilement tester Random Forest ou SVM :

```python
# Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=42)

# SVM
from sklearn.svm import SVC
model = SVC(kernel='rbf', probability=True, random_state=42)
```

### Ajuster l'interface

Modifiez `app.py` pour personnaliser :
- Les couleurs et le design (fichier `.streamlit/config.toml`)
- Les seuils de détection des facteurs de risque
- Les messages et recommandations
- Les visualisations

## 📊 Métriques du Modèle

Le modèle de **Régression Logistique** affiche :

| Métrique | Score | Description |
|----------|-------|-------------|
| **Accuracy** | 88.6% | Précision globale |
| **Precision** | Élevée | Fiabilité des prédictions positives |
| **Recall** | 92.2% | Détection des vrais cas positifs (crucial en médecine) |
| **F1-Score** | 90.0% | Équilibre précision/rappel |
| **ROC-AUC** | 93.4% | Excellente capacité discriminante |

## ⚠️ Avertissement

Cette application est à **titre éducatif et informatif uniquement**. Elle ne remplace en aucun cas un diagnostic médical professionnel. 

**Consultez toujours un professionnel de santé qualifié pour :**
- Un diagnostic médical
- Des conseils de traitement
- L'interprétation de résultats médicaux

## 🤝 Contribution

Améliorations possibles :
- Ajouter d'autres algorithmes de ML
- Implémenter la validation croisée
- Créer des visualisations interactives supplémentaires
- Ajouter l'export des résultats en PDF
- Intégrer un système de gestion des patients

## 📝 Technologies Utilisées

- **Streamlit** : Interface web interactive
- **Scikit-learn** : Modèles de Machine Learning
- **Pandas** : Manipulation des données
- **NumPy** : Calculs numériques
- **Matplotlib/Seaborn** : Visualisations

## 📧 Informations du Projet

Basé sur le dataset `heart.csv` et le notebook d'analyse `Projet ML - Cardiovasculaire.ipynb`.

**Modèle** : Régression Logistique  
**Dataset** : 918 patients  
**Features** : 11 variables (6 numériques + 5 catégorielles)  
**Performance** : 88.6% accuracy, 92.2% recall

---

**Développé avec ❤️ en utilisant Streamlit et Scikit-learn**
