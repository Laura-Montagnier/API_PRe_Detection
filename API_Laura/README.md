# API pour la création de features de fichiers EXE

## Description

Cette API permet de créer différentes représentations de fichiers exécutables (EXE). Elle génère plusieurs types de features et les enregistre dans le dossier `Résultats`.

## Instructions

### 1. Placer les fichiers exécutables

Placez les fichiers exécutables dans le répertoire `Fichiers_Exécutables`.

### 2. Adapter ember 

pip install -r requirements.txt

python3 setup.py install

### 3. Installer LEAF

Installer la version 14 de Leaf.

### 3. Rendre le script exécutable

Ouvrez un terminal et rendez le script `représentations.sh` exécutable en entrant la commande suivante :

chmod +x représentations.sh

### 4. Exécuter le script

./représentations.sh

## Résultats

Les différents types de features sont présents dans le dossier "Résultats".

## Références

Pour le Pack_EMBER : https://github.com/elastic/ember
Pour PE_Feats : Charles-Henry Bertrand van Ouytsel et al.
