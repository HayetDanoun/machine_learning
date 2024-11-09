import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def normalize_data(X_train, X_test):
    """Normalise les données d'entraînement et de test et sauvegarde le scaler dans un fichier pickle."""
    scaler = StandardScaler()
    
    # Vérifiez que X_train et X_test ne sont pas vides
    if X_train is None or X_test is None or len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Les données d'entrée ne peuvent pas être vides.")
    
    # Ajuster et transformer les données d'entraînement
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    # Définir le chemin de sauvegarde
    artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)  # Crée le dossier si nécessaire
    scaler_path = os.path.join(artifacts_dir, 'scaler.pickle')
    
    # Sauvegarder le scaler dans un fichier pickle
    try:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print("Scaler sauvegardé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du scaler : {e}")
    
    return X_train_normalized, X_test_normalized, scaler

# Exemple de données pour tester la fonction
X_train = np.array([[1, 2], [3, 4], [5, 6]])  # Exemple de données d'entraînement
X_test = np.array([[7, 8], [9, 10]])          # Exemple de données de test

# Appel de la fonction avec les données
X_train_normalized, X_test_normalized, scaler = normalize_data(X_train, X_test)
