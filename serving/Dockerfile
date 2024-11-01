FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le code de l'API et le fichier requirements
COPY api.py .
COPY requirements.txt .

# Mettre à jour pip et installer les packages
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Commande pour exécuter l'API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
