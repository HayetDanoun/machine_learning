import os
import numpy as np
import pandas as pd
from PIL import Image

# Définir les chemins pour les images de chats et de chiens
DATA_FOLDERS = {
    'cat': "PetImages/Cat",
    'dog': "PetImages/Dog"
}

image_data = []
labels = []

# Compteur d'images chargées
loaded_count = 0

for label, folder in DATA_FOLDERS.items():
    for image_file in os.listdir(folder):
        if image_file.endswith((".jpg", ".png")):
            image_path = os.path.join(folder, image_file)
            try:
                # Charger et redimensionner l'image
                image = Image.open(image_path).resize((32, 32)).convert("L")  # Redimensionner à 32x32 et convertir en niveaux de gris
                image_array = np.array(image)  # Convertir l'image en tableau numpy

                # Aplatir l'image en un vecteur
                vector = image_array.flatten()  # Transformer l'image en vecteur
                image_data.append(vector)  # Ajouter le vecteur à la liste
                labels.append(label)  # Ajouter l'étiquette (cat ou dog)
                loaded_count += 1
            except Exception as e:
                print(f"Erreur de chargement de l'image : {image_path}, Erreur: {e}")

print(f"Total des images chargées : {loaded_count}")

# Créer un DataFrame à partir des données d'image et des labels
df = pd.DataFrame(image_data)
df['label'] = labels  # Ajouter la colonne des étiquettes (cat ou dog)

# Enregistrer le DataFrame dans un fichier CSV
output_csv = "ref_data.csv"
df.to_csv(output_csv, index=False)
print(f"Données transformées enregistrées dans le fichier : {output_csv}")
