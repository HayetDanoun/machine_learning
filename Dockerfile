# Utiliser une image de base Jupyter
FROM jupyter/scipy-notebook:latest

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le contenu du dossier app dans le conteneur
COPY ./app /app

# Exposer le port sur lequel Jupyter s'exécute
EXPOSE 8888

# Commande pour démarrer Jupyter Notebook
CMD ["start-notebook.sh", "--NotebookApp.token=''"]
