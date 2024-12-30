import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb  # Importer XGBoost
import scaler as sc

# Charger le jeu de données de référence (data_ref)
data_ref = pd.read_csv(r"C:\Users\joeto\projetfina_ML\data\ref_data.csv")  # Remplace par le bon chemin de ton fichier
data_ref["label"] = data_ref["label"].apply(lambda x: 0 if x == 'cat' else 1)

# Séparer les variables indépendantes (X) et la variable cible (y)
X = data_ref.drop(columns=['label'])  # Remplace 'label' par le nom correct de la colonne cible si nécessaire
y = data_ref['label']

# Si nécessaire, normaliser les données
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Diviser les données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train, X_test ,scaler= sc.normalize_data(X_train, X_test)


# Initialiser le modèle XGBoost
model = xgb.XGBClassifier(
    max_depth=6,
    n_estimators=300,
    learning_rate=0.1,
    objective='binary:logistic',
    subsample=0.8,
    colsample_bytree=0.8
)

# Entraîner le modèle avec le jeu de données d'entraînement
model.fit(X_train, y_train)

# Sauvegarder le modèle et le scaler
with open(r"C:\Users\joeto\projetfina_ML\artifacts\model.pickle", "wb") as f:
    pickle.dump(model, f)

with open(r"C:\Users\joeto\projetfina_ML\artifacts\scaler.pickle", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Le modèle XGBoost a été entraîné et sauvegardé avec succès !")
