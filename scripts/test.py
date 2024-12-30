import pandas as pd

# Charger le fichier CSV situé dans /app/data/data.csv
data = pd.read_csv("./app/data/ref_data.csv")

# Afficher les premières lignes pour vérifier
print(data.head())
