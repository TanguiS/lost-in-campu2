# Projet Lost in Campus 2

Ce git contient les sources pour le projet 2A Lost in Campus 2, réalisé par :

- Steimetz Tangui
- Seng Thomas
- Combet Marceau
- Ducastel Matéo

## Sommaire

- [Projet Lost in Campus 2](#projet-lost-in-campus-2)
  - [Sommaire](#sommaire)
  - [Introduction](#introduction)
  - [Script de prétraitement](#script-de-prétraitement)
  - [Exemple d'évaluation](#exemple-dévaluation)
  - [Analyse des images sur une heatmap](#analyse-des-images-sur-une-heatmap)
  - [Analyse des prédictions](#analyse-des-prédictions)
  - [Autre](#autre)
  - [Ref](#ref)

## Introduction

L'objectif de ce projet était de développer un modèle d'intelligence artificielle permettant d'obtenir une estimation de coordonnées GPS à partir d'une photo.

Plusieurs outils ont été développés dans cet objectif :

- script de prétraitement des images
- script pour évaluer des images
- Visualisation des photos sur une heatmap
- Visualisation des estimations

## Script de prétraitement

Afin de pouvoir utiliser des images pour le modèle, il va d'abord falloir les traiter afin qu'elle ait le bon format (tel, nom de fichier, etc..).
Pour ce faire, nous avons développé plusieurs scripts pour faire ces traitements. Le fichier [script/main_preprocess.py](script/main_preprocess.py) effectue le traitement entierement.

Ce script de prétraitement permet donc de modifier les images normales dans des images utilisables par le modèle. Il est impératif d'effectuer ces traitements avant de pouvoir lancer un entrainement ou une évaluation.

Le fichier [Colab](colab/processed.ipynb) associé montre comment l'utiliser.

## Exemple d'évaluation

Le fichier [script/main_evaluation_example.py](script/main_evaluation_example.py) fournit un exemple d'évaluation du modèle. On peut grace à celui-ci fournir deux dossiers qui contiennent les images pour extraire les descripteurs et les images à évaluer, ainsi que plusieurs paramètres pour le modèle à charger. On aura en retour une évaluation de nos images par le modèle choisi.

Le fichier [Colab](colab/test_evaluation.ipynb) associé montre comment l'utiliser.

## Analyse des images sur une heatmap

Le fichier [script/main_analysis.py](script/main_analysis.py) fournit un script pour effectuer une Heatmap sur une zone GPS donnée ainsi que les images de l'ensemble de données. Ce script permet de vérifier la couverture des données.

## Analyse des prédictions

Le fichier [script/main_evaluation_sections](script/main_evaluation_sections.py) fournit un script pour évaluer les performances d'un modèle donné en argument. Ce script permet d'afficher un graphique pour confronter la position réelle de l'image par rapport à sa prediction. De plus, il fournit un graphique permettant d'évaluer chaque section à l'aide de 'pie charts' représentant les proportions de bonnes et mauvaises prédictions (selon le critère "Inférieur à 10 m"), ainsi qu'un label indiquant le temps moyen de prédiction d'un photo de la section.
Selon le choix de l'utilisateur, il peut effectuer l'évaluation sur une liste définié de sections, une liste de sections choisies aléatoirement parmi l'ensemble des sections disponibles, ou sur la totalité des sections.

*Note concernant les lignes de distance* :
Les lignes possèdent un gradient de couleurs.
Dans la zone verte, la distance est entre 0 m et 5 m.
Dans la zone jaune, la distance est entre 5 m et 10 m.
Dans la zone orange, la distance est entre 10 m et 15 m.
Dans la zone rouge, la distance est au-dessus de 15 m. 

Le fichier [Colab](colab/test_section_evaluator.ipynb) associé montre comment l'utiliser.

## Autre

Il existe également un fichier [Colab](colab/train.ipynb) pour l'entrainement de notre model

## Ref

@InProceedings {Berton_CVPR_2022_CosPlace,
    author    = {Berton, Gabriele and Masone, Carlo and Caputo, Barbara},
    title     = {Rethinking Visual Geo-Localization for Large-Scale Applications},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4878-4888}
} 

