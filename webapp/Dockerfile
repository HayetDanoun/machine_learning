# Utiliser une image de base Jupyter
FROM jupyter/scipy-notebook:latest

# Définir le répertoire de travail dans le conteneur comme 'scripts'
WORKDIR /scripts

# Copier le contenu du dossier courant dans le conteneur
COPY . /scripts

# Exposer le port sur lequel Jupyter s'exécute
EXPOSE 8888

# Commande pour démarrer Jupyter Notebook
CMD ["start-notebook.sh", "--NotebookApp.token=''"]
