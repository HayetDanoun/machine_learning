import pickle
import numpy as np
from PIL import Image

# Fonction pour transformer l'image en vecteur
def image_to_vector(image_path):
    try:
        # Charger et redimensionner l'image
        image = Image.open(image_path).resize((32, 32)).convert("L")  # Redimensionner à 32x32 et convertir en niveaux de gris
        image_array = np.array(image)  # Convertir l'image en tableau numpy

        # Aplatir l'image en un vecteur
        vector = image_array.flatten()  # Transformer l'image en vecteur
        return vector
    except Exception as e:
        print(f"Erreur de chargement de l'image : {image_path}, Erreur: {e}")
        return None

# Fonction pour sauvegarder la fonction image_to_vector dans un fichier pickle
def save_image_model(model_function, model_path="pca.pickle"):
    """Sauvegarde la fonction de transformation d'image dans un fichier pickle."""
    with open(model_path, 'wb') as f:
        pickle.dump(model_function, f)
    print(f"Le modèle de transformation d'image a été sauvegardé à {model_path}")

# Sauvegarder la fonction dans un fichier pickle
save_image_model(image_to_vector)

# Exemple d'utilisation : chemin de ton image
image_path = r"C:\Users\joeto\OneDrive\Desktop\m2\machine_learning\PetImages\Cat\0.jpg"

# Charger la fonction depuis le pickle
def load_image_model(model_path="pca.pickle"):
    """Charge la fonction de transformation d'image depuis un fichier pickle."""
    try:
        with open(model_path, 'rb') as f:
            model_function = pickle.load(f)
        return model_function
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None

# Charger la fonction depuis le fichier pickle
image_to_vector = load_image_model("pca.pickle")

# Exemple d'utilisation de la fonction chargée avec ton chemin d'image
if image_to_vector:
    image_vector = image_to_vector(image_path)  # Passer le chemin de l'image ici
    print(image_vector)  # Afficher le vecteur de l'image
