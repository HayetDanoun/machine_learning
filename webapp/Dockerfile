# Utiliser une image de base de Python
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier de dépendances et l'installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Exposer le port 8081
EXPOSE 8081

# Lancer l'application Streamlit
CMD ["streamlit", "run", "api.py", "--server.port=8081", "--server.address=0.0.0.0"]
