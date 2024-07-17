#!/bin/bash

# Définir le chemin vers le répertoire des exécutables
dir_path="/root/API_Laura/Fichiers_exécutables/"

# Exécuter le script Python pour les Grayscales avec le chemin en argument
python3 représentation_grayscale.py "$dir_path"

# Charger le modèle à partir du fichier .pkl
python3 << END
import pickle
import numpy as np
import os
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import sys
sys.path.append('/root/Modèles')
from convnet import Convnet


# Chemin complet vers le modèle .pkl
model_path = '/root/Modèles/cnn_model_grayscale.pkl'

# Charger le modèle
with open(model_path, 'rb') as file:
	cnn_model_grayscale = pickle.load(file)

# Chemin complet vers le répertoire contenant les fichiers Grayscales
grayscales_dir = "/root/API_Laura/Résultats/Images_Grayscale"

# Parcourir les fichiers dans le répertoire Grayscales
for filename in os.listdir(grayscales_dir):
	file_path = os.path.join(grayscales_dir, filename)
        
	# Charger les données à partir du fichier
	image = Image.open(file_path).convert('L')  # Convertir en niveaux de gris
	data = np.array(image)
        
	# Reshape the data to match the input shape of the model
	data = data.reshape(1, 1, data.shape[0], data.shape[1])
	data = torch.tensor(data, dtype=torch.float32)

	# Faire des prédictions avec le modèle
	cnn_model_grayscale.eval()  # Mettre le modèle en mode évaluation

	with torch.no_grad():
		# Obtenir les sorties brutes du modèle
		outputs = cnn_model_grayscale(data)
		# Appliquer la fonction softmax pour obtenir les probabilités
		probabilities = F.softmax(outputs, dim=1)
		# Obtenir l'indice de la classe prédite
		idx = probabilities.argmax(axis=1).cpu().numpy()[0]
		# Afficher les probabilités et la prédiction
		print(f"Probabilities: {probabilities.cpu().numpy()}")
		if probabilities.cpu().numpy()[0][0]==1:
			print("Le Grayscale suffit !")
		print(f"Predicted class index for {filename}: {idx}")


# Supposons que 'ma_variable' contienne la valeur que vous souhaitez récupérer
		ma_variable = str(probabilities.cpu().numpy()[0][0])

# Écriture de la valeur dans un fichier
		with open('mon_fichier.txt', 'w') as f:
			f.write(ma_variable)
END

valeur_recuperee=$(< mon_fichier.txt)

#if [ "$valeur_recuperee" == "1.0" ]; then
	#print("Fin du script")
	#exit 0

rm mon_fichier.txt

# Exécuter le script Python pour les Graphes d'entropie avec le chemin en argument
python3 représentation_entropy_graph.py "$dir_path"

# Charger le modèle à partir du fichier .pkl
python3 << END
import pickle
import numpy as np
import os
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import sys
sys.path.append('/root/Modèles')
from convnet import Convnet


# Chemin complet vers le modèle .pkl
model_path = '/root/Modèles/cnn_model_entropie.pkl'

# Charger le modèle

with open(model_path, 'rb') as file:
         cnn_model_entropy = pickle.load(file)

# Chemin complet vers le répertoire contenant les fichiers Grayscales
entropy_dir = "/root/API_Laura/Résultats/Graphes_entropie"

#On va parcourir les fichiers dans le répertoire Grayscales
for filename in os.listdir(entropy_dir):
	file_path = os.path.join(entropy_dir, filename)

	# Charger les données à partir du fichier
	image = Image.open(file_path).convert('L')  # Convertir en niveaux de gris
	image = image.resize((250,250))
	data = np.array(image)
	
	data = data.reshape(1, 1, data.shape[0], data.shape[1])
	data = torch.tensor(data, dtype=torch.float32)
	
	# Faire des prédictions avec le modèle
	cnn_model_entropy.eval()  # Mettre le modèle en mode évaluation

	with torch.no_grad():
                # Obtenir les sorties brutes du modèle
		outputs = cnn_model_entropy(data)
		# Appliquer la fonction softmax pour obtenir les probabilités
		probabilities = F.softmax(outputs, dim=1)
		# Obtenir l'indice de la classe prédite
		idx = probabilities.argmax(axis=1).cpu().numpy()[0]
		# Afficher les probabilités et la prédiction
		print(f"Probabilities: {probabilities.cpu().numpy()}")
		if probabilities.cpu().numpy()[0][0]==1:
			print("Le Graphe d'entropie suffit !")
		print(f"Predicted class index for {filename}: {idx}")
		
		ma_variable = str(probabilities.cpu().numpy()[0][0])
		
		with open('mon_fichier.txt', 'w') as f:
			f.write(ma_variable)

END

valeur_recuperee=$(< mon_fichier.txt)

#if [ "$valeur_recuperee" == "1.0"]; then
	#print("Fin du script")
	#exit 0

rm mon_fichier.txt

#Les features PE_feats
cd PE_feats
touch ../Résultats/pe_feats.csv
for filename in "$dir_path"/*
	do
	./pefeats "$filename" >> ../Résultats/pe_feats.csv
	./pefeats "$filename" >> ../Résultats/pe_feats.csv
	done
cd ..

python3 << END
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Chemin complet vers le csv contenant les features PE_feats
pe_feats_file = "/root/API_Laura/Résultats/pe_feats.csv"

# Chemin complet vers le modèle .pkl
model_path = '/root/Modèles/GB_model_pe_feats.pkl'

# Charger le modèle

with open(model_path, 'rb') as file:
	GB_model_pe_feats = pickle.load(file)

#Ouvrir le fichier csv et le lire
features_pe_feats = pd.read_csv(pe_feats_file)


#Récupérer les noms de colonnes
num_features = features_pe_feats.shape[1]
features_pe_feats = features_pe_feats.iloc[:, 1:]
new_column_names = [f'F{i+1}' for i in range(num_features-1)]
features_pe_feats.columns = new_column_names

print("Selon PE_feats, la classe prédite est :")
# Faire des prédictions avec le modèle chargé
predictions = GB_model_pe_feats.predict(features_pe_feats)

# Afficher les prédictions
print(predictions)

# Afficher un message avant de faire les prédictions
print("Selon PE_feats, les probabilités de prédiction sont :")

# Faire des prédictions avec les probabilités
predictions_proba = GB_model_pe_feats.predict_proba(features_pe_feats)

u = max(predictions_proba[0][1],predictions_proba[0][0])
print(u)

ma_variable=str(u)
with open('mon_fichier.txt', 'w') as f:
	f.write(ma_variable)

END

valeur_recuperee=$(< mon_fichier.txt)

#if [ "$valeur_recuperee" == "1.0"]; then
	#print("Fin du script")
	#exit 0

rm mon_fichier.txt

#Les features Ember
cd Pack_Ember
python3 représentation_ember.py "$dir_path"
python3 représentation_ember.py "$dir_path"
cd ..

python3 << END

import pickle
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Chemin complet vers le csv contenant les features Ember
ember_file = "/root/API_Laura/Résultats/Features_Ember.csv"

# Chemin complet vers le modèle .pkl
model_path = '/root/Modèles/GB_model_ember.pkl'

# Charger le modèle

with open(model_path, 'rb') as file:
	GB_model_ember = pickle.load(file)

#Ouvrir le fichier csv et le lire
features_ember = pd.read_csv(ember_file)

#Récupérer les noms de colonnes
num_features = features_ember.shape[1]
features_ember = features_ember.iloc[:, 1:]
new_column_names = [f'F{i+1}' for i in range(num_features-1)]
features_ember.columns = new_column_names

print("Selon Ember, la classe prédite est :")

# Faire des prédictions avec le modèle chargé
predictions = GB_model_ember.predict(features_ember)

# Afficher les prédictions
print(predictions)

END
