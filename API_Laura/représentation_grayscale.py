import numpy as np
import csv
from PIL import Image
import os
import argparse

# Première étape, transformer le binaire en csv : on a une fonction spécifique
def convert_to_csv(input_file, output_file):
    try:
        with open(input_file, 'rb') as f_in:
            bytes_list = list(f_in.read())

        with open(output_file, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            # Parcourir la liste d'octets et écrire chaque octet dans une ligne du fichier CSV
            for byte_value in bytes_list:
                writer.writerow([byte_value])
    except Exception as e:
        print(f"Erreur lors de la conversion en CSV du fichier {input_file}: {str(e)}")

# Deuxième étape, transformer le csv en png : on a une fonction spécifique
def csv_to_image(input_file, output_file):
    try:
        # Charger les données CSV en tant que tableau NumPy
        data_array = np.loadtxt(input_file, delimiter=',', dtype=np.uint8)

        # Inverser les couleurs pour correspondre à la représentation souhaitée
        data_array = 255 - data_array

        # Redimensionner les données pour correspondre à la taille de l'image
        largeur = 250
        hauteur = data_array.shape[0] // largeur
        data_array = data_array[:hauteur * largeur].reshape(hauteur, largeur)

        # Créer une image à partir du tableau de données
        img = Image.fromarray(data_array, 'L')  # 'L' pour l'image en niveaux de gris

        # Enregistrer l'image
        img.save(output_file)
    except Exception as e:
        print(f"Erreur lors de la conversion en image du fichier {input_file}: {str(e)}")

# Fonction principale
def main(exe_dir):
    # Définir les répertoires intermédiaires et de sortie
    interm_dir = os.path.expanduser('~/API_Laura/Images_intermediaires')
    output_dir = os.path.expanduser('~/API_Laura/Résultats/Images_Grayscale')

    # Vérifier si le répertoire existe
    if not os.path.exists(exe_dir):
        print(f"Le répertoire {exe_dir} n'existe pas.")
        return

    # Créer le répertoire intermédiaire s'il n'existe pas
    if not os.path.exists(interm_dir):
        os.makedirs(interm_dir)

    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parcourir chaque fichier dans le répertoire exe_dir
    for file_name in os.listdir(exe_dir):
        file_path = os.path.join(exe_dir, file_name)
        
        # Vérifier si le chemin est un fichier
        if os.path.isfile(file_path):
            # Définir les chemins de sortie pour les fichiers CSV et PNG
            csv_output_path = os.path.join(interm_dir, f"{file_name}_pour_image.csv")
            png_output_path = os.path.join(interm_dir, f"{file_name}.png")
            
            # Convertir le fichier binaire en fichier CSV
            convert_to_csv(file_path, csv_output_path)
            
            # Convertir le fichier CSV en image PNG
            csv_to_image(csv_output_path, png_output_path)

            # Supprimer le fichier CSV intermédiaire
            os.remove(csv_output_path)

    # Parcourir chaque fichier dans le répertoire interm_dir
    for filename in os.listdir(interm_dir):
        # Chemin complet du fichier d'entrée
        input_path = os.path.join(interm_dir, filename)
        
        # Vérifier si le chemin est un fichier
        if os.path.isfile(input_path):
            try:
                # Ouvrir l'image
                with Image.open(input_path) as img:
                    # Redimensionner l'image en 250x250 pixels
                    resized_img = img.resize((250, 250))
                    
                    # Chemin complet du fichier de sortie
                    output_path = os.path.join(output_dir, filename)
                    
                    # Enregistrer l'image redimensionnée
                    resized_img.save(output_path)
                    print(f"Image {filename} en Grayscale créée avec succès.")
            except Exception as e:
                print(f"Erreur lors du redimensionnement de l'image {filename}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process executable files to images.')
    parser.add_argument('exe_dir', type=str, help='Directory containing the executable files.')

    args = parser.parse_args()

    # Convert executable files to images
    main(args.exe_dir)

