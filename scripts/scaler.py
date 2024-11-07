import pickle
from sklearn.preprocessing import StandardScaler

def normalize_data(X_train, X_test):
    """Normalise les données d'entraînement et de test et sauvegarde le scaler dans un fichier pickle."""
    scaler = StandardScaler()
    
    # Vérifiez que X_train et X_test ne sont pas vides
    if X_train is None or X_test is None:
        raise ValueError("Les données d'entrée ne peuvent pas être vides.")
    
    # Ajuster et transformer les données d'entraînement
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    # Sauvegarder le scaler dans un fichier pickle
    try:
        with open('/app/artifacts/scaler.pickle', 'wb') as f:
            pickle.dump(scaler, f)
        print("Scaler sauvegardé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du scaler : {e}")
    
    return X_train_normalized, X_test_normalized, scaler
