# Classifieur Chat-Chien

## Introduction
Ce projet a pour objectif de développer un classifieur capable de différencier les images de chats et de chiens en utilisant des algorithmes de machine learning supervisés. Plusieurs modèles ont été testés et comparés afin de sélectionner celui offrant les meilleures performances pour le modèle final.

## Git
Ce projet peut être retrouvée en suivant ce lien https://github.com/toubia-j/machine_learning.

Le code final du projet est disponible sur la branche `main`.

## Auteurs
Ce projet a été réalisé par :

- Toubia Joe - P2312945
- Rezaoui Yanis - P1810590
- Danoun Hayet - P1925003
- Koudia Selma - P2408052
- Bonhotal Jules - P2003042

## Exécution

Les adresses des pages seront affichées après l’exécution des commandes suivantes. Entrez ces adresses dans un navigateur pour accéder aux pages correspondantes.

### Construire et Démarrer le Service

Dans le dossier `serving`, exécutez la commande suivante :

```bash
docker-compose up --build
```

### Démarrer l'Application Streamlit

Dans le dossier `webapp`, exécutez la commande suivante :

```bash
docker-compose up --build
```

Dans le dossier `reporting`, exécutez la commande suivante :

```bash
docker-compose up --build
```

**Remarque :** Le calcul du rapport peut prendre plusieurs minutes. Une fois le processus terminé, le rapport sera disponible à l'adresse suivante : [http://0.0.0.0:8501/](http://0.0.0.0:8501/)

