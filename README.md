Cette API sert à détecter si un Portable Executable est un Malware (un logiciel malveillant). 
Pour ce faire, il faut aller dans API_Laura et faire toutes les installations pré-requises dans le README.md. 
Ensuite, il faut placer le Portable Executable dans le dossier Fichiers_Exécutables, supprimer le dossier Résultats afin de ne pas avoir des résultats obtenus précédemment.
On peut alors lancer le script script_principal.
Il va déterminer en transformant le PE en image Grayscale s'il s'agit d'un logiciel malveillant.
S'il n'est pas sûr de lui à 100%, il va utiliser le graphe d'entropie du logiciel.
S'il n'est toujours pas sûr de lui à 100%, il va utiliser la représentation PE_feats du logiciel.
S'il n'est (encore) pas sûr à 100%, il va utiliser la représentation Ember du logiciel.
