# Utiliser une image de base officielle Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application dans le conteneur
COPY . .

# Exposer le port 8081 pour l'application Streamlit
EXPOSE 8081

# Démarrer l'application Streamlit
CMD ["streamlit", "run", "api.py", "--server.port=8081"]
