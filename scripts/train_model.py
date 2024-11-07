import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pickle  # Utilisation de pickle pour l'enregistrement des objets

def load_data(filepath):
    """Charge le jeu de données à partir d'un fichier CSV."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Sépare les caractéristiques et les étiquettes, et encode les étiquettes."""
    X = df.drop(columns=['label'])  # Caractéristiques (vecteurs d'images)
    y = df['label']  # Étiquettes
    # Encoder les étiquettes
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)  # Encode les étiquettes en entiers
    return X, y_encoded, encoder  # Retourne aussi l'encodeur

def normalize_data(X_train, X_test):
    """Normalise les données d'entraînement et de test."""
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)  # Ajuster et transformer les données d'entraînement
    X_test_normalized = scaler.transform(X_test)        # Transformer les données de test
    return X_train_normalized, X_test_normalized, scaler

def train_model(X_train, y_train):
    """Entraîne un modèle de Random Forest."""
    model = RandomForestClassifier(random_state=42)  # Fixer le random_state pour la reproductibilité
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle en calculant l'AUC-ROC et le F1 Score."""
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    y_pred = (y_pred_prob >= 0.5).astype(int)  # Seuil de 0.5 pour les prédictions binaires
    f1 = f1_score(y_test, y_pred)
    return auc_roc, f1, y_pred_prob, y_pred

def plot_roc_curve(y_test, y_pred_prob, auc_roc):
    """Trace la courbe ROC."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Courbe ROC (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Ligne diagonale
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    """Affiche la matrice de confusion."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matrice de Confusion')
    plt.show()

def save_model(model):
    """Enregistre le modèle, le scaler et l'encodeur sous forme de fichiers pickle."""
    with open('/app/artifacts/model.pickle', 'wb') as f:
        pickle.dump(model, f)
    
if __name__ == "__main__":
    # Chargement des données
    data_folder = '/data'
    refdata_file = os.path.join(data_folder, 'ref_data.csv')  # ajuster le nom du fichier si nécessaire

    df = load_data(refdata_file)

    # Prétraitement des données
    X, y, encoder = preprocess_data(df)  # Ne retourne plus l'encodeur
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation des données
    X_train_normalized, X_test_normalized, scaler = normalize_data(X_train, X_test)

    # Entraînement du modèle
    model = train_model(X_train_normalized, y_train)

    # Évaluation du modèle
    auc_roc, f1, y_pred_prob, y_pred = evaluate_model(model, X_test_normalized, y_test)
    print(f"ROC: {auc_roc:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Tracer la courbe ROC
    plot_roc_curve(y_test, y_pred_prob, auc_roc)
    plot_confusion_matrix(y_test, y_pred)

    # Sauvegarder les objets
    save_model(model)
