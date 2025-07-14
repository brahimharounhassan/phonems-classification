# Travail d'Etude et de Recherche (TER)

![Python 3.13.1](https://img.shields.io/badge/Python-3.13.1-yellow?style=plastic)

## Nom

Projet TER.

## Description

Dans ce projet on cherche à comprendre comment le système auditif du nourrisson parvient-il à faire abstraction de certaines variations acoustiques pour reconnaitre un mot prononcé dans différents contextes ?
On vise à mieux comprendre les mécanismes cérébraux impliqués dans l’analyse de la parole chez le nourrisson et caractériser les différentes étapes de traitement des sons de sa langue maternelle.

## Aperçu du problem

Etant donné des enregistrements Electroencéphalographiques (EEG), nous allons analyser des données EEG acquis chez des nourrissons de 3 mois auxquels étaient présentées différentes syllabes de leur langue maternelle; en particulier les phoménes /ba/ et /da/.

## Objectif

- Mettre en place un algorithme de prétraitement des données EEG, de façon à en retirer les segments corrompus (filtrage des données et définition de seuils de réjection d'artefacts) et à formatter les données pour les analyses suivantes.
- Appliquer des algorithmes d’apprentissage afin de tester si différentes propriétés (acoustique et linguistique) des sons présentés peuvent être décodées à partir des enregistrements EEG : en utilisant un modèle de régression logistique.
- Evaluer les effets des hyperparamètres du modèle, et de la taille du jeu d’entrainement sur ses performances de classification.

## Structure du projet

- [x] `cluster/` : Ce répertoire contient les fichiers de configuration pour l'éxecution dans le cluster de calcul.
- [x] `configs/` : Pour les fichiers de configurations.
- [x] `data/` : Contient tous les données EEG.
  - [x] `clean/` : Contient les données prétraitées.
  - [x] `raw/` : Contient les données Brûtes.
- [x] `logs/` : Répertoire contenant l'historique et traces d'exécutions.
- [x] `models/` : Ce dossier contient le modèle entrainé.
- [x] `notebooks/` : Dossier contenant les fichiers de types notebooks avec usage sur jupyter ou Colab.
- [x] `output/` : Ce dossier contient tous les résultats d'exécution.
- [x] `src/` : Contient les fichiers sources type scripts python.
  - `utils.py` : Contient les implementations de classes et méthodes utiles.
  - `train_func.py` : Contient les implémentation nécessaire pour l'apprentissage de modèles.
  - `train.py` : Pour entrainer le modèle simple.
  - `train_micro_avg.py` : Pour entrainer le modèle avec la technique de micro-moyennage avec et sans rémise.
  - `evaluate.py` : Pour la générer les résultats sous forme de graphes.
  - `visualize.py` : Pour la visualisation des données.
- [x] `tests/` : Répertoire contenant les tests des méthodes implémentées.
- [x] `ter_report.pdf` : Le rapport du projet.
- [x] `README.md` :
- [x] `LICENSE` :

## Usage

Une fois les machines virtuelles en fonctionnement :
Téléchargez (zip) ou clonez le projet à l'aide d'un terminal dans un dossier spécifique.:

```
# - fichier train_micro_avg.py pour l'entrainement.
cd ter_project/
# sans remise
python src/train_micro_avg.py  --data_dir ./data/clean --output_dir ./output  --iter 50 --start_n 1 --end_n 11 --n_splits 5 --subject 33
# ou avec remise
python src/train_micro_avg.py  --data_dir ./data/clean --output_dir ./output  --reuse true --iter 50 --start_n 1 --end_n 11 --n_splits 5 --subject 33
# - fichier evaluate.py pour l'évaluation.
# Pour un sujet spécifique sans selection d'essais et sans remise
python src/evaluate.py  --data_dir ./output/train --output_dir ./output  --iter 100  --avg true --subject 3
# Pour un sujet spécifique sans selection d'essais et avec remise
python src/evaluate.py  --data_dir ./output/train --output_dir ./output  --iter 100  --avg true --reuse true --subject 3
# Pour un sujet spécifique avec selection d'essais
python src/evaluate.py  --data_dir ./output/train --output_dir ./output  --iter 100  --subject all --include 4,5 --drop_t t

# Ou pour tout les sujets
python src/evaluate.py  --data_dir ./output/train --output_dir ./output  --iter 100  --avg true --reuse true --subject all

# - fichier train.py.
python src/train.py  --data_dir ./data/clean --output_dir ./output  --iter 100  --subject 27
```

-->

## Auteur

- [Brahim Haroun Hassan]

## License

Ce projet est distribué sous licence CC BY-NC (Creative Commons - Non Commercial) version 4.0 — voir le fichier LICENSE pour les détails.

## Status du projet

En devéloppement ...
