import pickle
import numpy as np
from PIL import Image

# Fonction de transformation d'image
def image_to_vector(image_path):
    try:
        # Charger et redimensionner l'image
        image = Image.open(image_path).resize((32, 32)).convert("L")  # Redimensionner à 32x32 et convertir en niveaux de gris
        
        # Convertir l'image en tableau numpy
        image_array = np.array(image)
        
        # Aplatir l'image en un vecteur
        vector = image_array.flatten()
        
        return vector
    except Exception as e:
        print(f"Erreur de chargement de l'image : {image_path}, Erreur: {e}")
        return None


with open(r"C:\Users\joeto\projetfina_ML\artifacts\pca.pickle", 'wb') as f:
    pickle.dump(image_to_vector, f)
print(f"La fonction de transformation d'image a été sauvegardée à")

