# Analyse des avis employés – Projet de fin de formation
Ce projet a été réalisé dans le cadre de ma formation en Data Science.
L’objectif était de développer un outil d’analyse des avis d’employés pour prédire leur niveau de satisfaction, à partir de commentaires textuels collectés sur Glassdoor.

## Objectifs principaux
Prétraiter et analyser les données textuelles (pros, cons, headline)

Entraîner et comparer plusieurs modèles de classification du sentiment (TF-IDF, Word2Vec, Transformer)

Développer un tableau de bord interactif pour explorer les résultats

## Contenu du dépôt
Notebooks/ → Exploration des données, entraînement des modèles, évaluation

Streamlit_dashboard/ → Fichiers Streamlit pour le déploiement de l’application

Mlflow/ → Fichiers pour le déploiement du server MLflow et le suivi de l'entraînement du modèle

Diaporama de présentation du projet

## Application déployée
Le tableau de bord est accessible sur Hugging Face Spaces :
👉 Voir l’application en ligne : https://huggingface.co/spaces/jedha0padavan/sentiment_analysis_FP

## Exemple de fonctionnalités
Visualisation des avis textuels sous forme de WordClouds et n-grammes

Prédiction automatique du sentiment des employés

Onglet dédié à l’analyse d’une entreprise externe non vue par le modèle


