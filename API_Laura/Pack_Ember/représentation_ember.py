#!/usr/bin/env python

import os
import argparse
import ember
import numpy as np

def main(folder_path):
    # Vérifier si le dossier existe
    if not os.path.exists(folder_path):
        print(f"Le dossier {folder_path} n'existe pas.")
        return

    # Ouvrir le fichier CSV pour écriture
    output_file = "../Résultats/Features_Ember.csv"
    processed_files_list = "../Résultats/Processed_Files.txt"

    with open(output_file, "a") as f, open(processed_files_list, "a") as p:
        for binary in os.listdir(folder_path):
            file_path = os.path.join(folder_path, binary)
            #Ecrire le nom du fichier dans la liste
            p.write(binary + "\n")


            # Vérifier si le chemin est un fichier
            if os.path.isfile(file_path):
                try:
                    # Lire le fichier binaire
                    with open(file_path, "rb") as file_data:
                        data = file_data.read()
                    
                    # Créer les caractéristiques
                    feature = ember.create_features(data)
                    
                    # Vérifier les caractéristiques
                    if feature is not None and not np.isnan(feature).all():
                        # Écrire les caractéristiques dans le fichier CSV
                        f.write(binary)
                        for fea in feature:
                            f.write(",")
                            f.write(str(fea))
                        f.write("\n")
                    else:
                        print(f"Caractéristiques invalides pour le fichier {binary}")
                        
                except Exception as e:
                    print(f"Erreur lors du traitement du fichier {binary}: {e}")

if __name__ == "__main__":
    # Définition de l'argument du chemin du dossier
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('folder_path', type=str, help='Chemin vers le dossier contenant les fichiers exécutables')

    # Analyser les arguments de la ligne de commande
    args = parser.parse_args()

    # Appeler la fonction principale avec le chemin du dossier passé en argument
    main(args.folder_path)
