import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import os
from PIL import Image


def create_reference_data(
    data_folder, output_folder, n_components=50, max_images_per_class=1000
):
    """
    Transforme un ensemble d'images en vecteurs via PCA et sauvegarde le résultat

    Parameters:
    data_folder: dossier contenant les images
    output_folder: dossier où sauvegarder les fichiers
    n_components: nombre de composantes PCA
    max_images_per_class: nombre maximum d'images par classe à utiliser
    """

    # Liste pour stocker les données
    image_data = []
    labels = []

    # Charger et prétraiter les images
    for class_folder in os.listdir(data_folder):
        class_path = os.path.join(data_folder, class_folder)
        if os.path.isdir(class_path):
            # Limiter le nombre d'images par classe
            images = [
                img for img in os.listdir(class_path) if img.endswith((".jpg", ".png"))
            ]
            selected_images = images[:max_images_per_class]

            for image_file in selected_images:
                # Charger l'image
                image_path = os.path.join(class_path, image_file)
                image = Image.open(image_path)

                # Redimensionner en 64x64 et convertir en niveaux de gris
                image = image.resize((64, 64)).convert("L")

                # Convertir en array numpy et aplatir
                image_array = np.array(image).flatten()

                image_data.append(image_array)
                labels.append(class_folder)

    # Convertir en array numpy
    X = np.array(image_data)
    y = np.array(labels)

    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Application de la PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Créer le DataFrame de référence
    columns = [f"component_{i}" for i in range(n_components)]
    ref_data = pd.DataFrame(X_pca, columns=columns)
    ref_data["target"] = y
    print("test sortie :", ref_data)
    # Sauvegarder le DataFrame
    os.makedirs(output_folder, exist_ok=True)
    ref_data.to_csv(os.path.join(output_folder, "ref_data.csv"), index=False)

    # Sauvegarder les modèles de prétraitement
    artifacts_folder = os.path.join(output_folder, "artifacts")
    os.makedirs(artifacts_folder, exist_ok=True)

    with open(os.path.join(artifacts_folder, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(artifacts_folder, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    return ref_data, scaler, pca


# Exemple d'utilisation
if __name__ == "__main__":
    # Définir les chemins
    DATA_FOLDER = "PetImages"  # Dossier contenant vos images classées par sous-dossiers
    OUTPUT_FOLDER = "data"  # Dossier où sauvegarder ref_data.csv et les artifacts

    # Créer les données de référence
    ref_data, scaler, pca = create_reference_data(
        DATA_FOLDER, OUTPUT_FOLDER, max_images_per_class=1000
    )

    # Afficher les informations sur les données transformées
    print("Forme des données après PCA:", ref_data.shape)
    print("Variance expliquée cumulée:", pca.explained_variance_ratio_.cumsum()[-1])
