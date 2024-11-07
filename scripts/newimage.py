import pickle
import numpy as np
from PIL import Image

# Classe pour encapsuler la fonction de transformation d'image
class ImageTransformer:
    def __init__(self):
        pass

    def image_to_vector(self, image_path):
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

# Fonction pour sauvegarder la classe avec la méthode dans un fichier pickle
def save_image_model(image_transformer, model_path='/app/artifacts/pca.pickle'):
    """Sauvegarde l'objet contenant la fonction de transformation d'image dans un fichier pickle."""
    with open(model_path, 'wb') as f:
        pickle.dump(image_transformer, f)
    print(f"Le modèle de transformation d'image a été sauvegardé à {model_path}")

# Exemple d'utilisation : Enregistrer la classe dans un fichier pickle
transformer = ImageTransformer()
save_image_model(transformer)

# Exemple de chargement de l'objet depuis le pickle dans votre API FastAPI
with open('/app/artifacts/pca.pickle', 'rb') as f:
    image_transformer_from_pickle = pickle.load(f)

