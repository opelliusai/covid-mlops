'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Fichier de nettoyage des données après téléchargement
Stratégie choisie: Utiliser uniquement les images et non les masques
Génération d'un fichier metadata.json qui pourrait être réutilisée par MLFlow
Génération d'un fichier csv contenant des informations sur les images, Classe, source, date
'''
## 
# Imports
from datetime import datetime
import os
import shutil
import json
import csv
import math
import os
import shutil
import random


# import information logger et fichier de configuration
from src.config.run_config import init_paths,dataset_info
from src.config.log_config import logger

##
def clean_data_multiclasse_toutes_images(path_to_data,dataset_output,version):
    logger.debug(f"---------------clean_data_multiclasse(path_to_data={path_to_data}, dataset_output={dataset_output},version={version}---------------")
    
    """
    Nettoyage des données
    1. Création du répertoire destination dans processed
    2a. Création des répertoire source dans destination après avoir supprimé l'espace
    2b. Copie du contenu de sous-répertoires/images dans le nouveau sous-répertoire
    2c. Mise à jour du fichier csv avec les informations sur les images: Répertoire initial, répertoire cible, nom de l'image
    3. Après parcours de tous les sous-répertoires, mise à jour du fichier metadata.json
    """
    
    # 1. Création du répertoire destination dans processed
    logger.info("Création du répertoire destination dans processed")
    os.makedirs(dataset_output, exist_ok=True)
    # Initialisation de l'objet data qui contient toutes les lignes
    data=[]
    num_classes=0
    nb_data_per_class={}
    for subfolder in os.listdir(path_to_data):
        # Vérifier s'il s'agit d'un sous-répertoire
        if os.path.isdir(os.path.join(path_to_data, subfolder)):
            num_classes+=1
            logger.debug(f"Traitement du sous-répertoire {subfolder}")
            logger.debug("Utilisation des images du sous-répertoire images")
            full_sub_rep=os.path.join(path_to_data, subfolder,"images")
            logger.debug(f"Chemin complet du sous-répertoire {full_sub_rep}")
            logger.debug(f"Parcours du Sous-répertoire {subfolder}")
            # renommer le sous-fichier avant de le créer dans le répertoire de destination
            new_folder_name=util_remove_space_from_foldername(subfolder)
            logger.debug(f"Nouveau nom de sous-répertoire {new_folder_name}")
            full_target_subrep=os.path.join(dataset_output, new_folder_name)
            logger.debug(f"Chemin complet du nouveau sous-répertoire {full_target_subrep}")
            # 3a. Création des répertoire source dans destination après avoir supprimé l'espace
            logger.debug(f"Création du sous-répertoire {full_target_subrep}")
            os.makedirs(full_target_subrep, exist_ok=True)
            # 3b. Copie du contenu de sous-répertoires/images dans le nouveau sous-répertoire
            nb_data=0
            for filename in os.listdir(full_sub_rep):
                full_filename_path=os.path.join(full_sub_rep, filename)
                nb_data+=1
                if os.path.isfile(full_filename_path) and filename!=".DS_Store":
                    logger.debug(f"Copie du fichier {filename} dans le sous-répertoire {full_target_subrep}")
                    shutil.copy(os.path.join(full_sub_rep, filename), full_target_subrep)
                    # 3c. création d'un dictionnaire des informations sur le fichier
                    file_info={}
                    file_info['Sous-répertoire SOURCE']=subfolder
                    file_info['Classe']=subfolder
                    file_info['Sous-répertoire CIBLE']=new_folder_name
                    file_info['Nom de fichier']=filename
                    file_info['Date d\'ajout']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logger.debug(f"Informations sur le fichier {file_info}")
                    # 3c. Mise à jour de l'objet data qui sera ajouté dans le fichier csv
                    data.append(file_info)
                    logger.debug("Informations du fichier ajouté dans le batch data pour le fichier csv")
            # ajout des informations du nombre d'image pour chaque nouvelle classe
            logger.debug(f"Nombre d'images trouvés dans le sous-répertoire source: {nb_data}")
            nb_data_per_class[subfolder]= nb_data    
            logger.debug(f"Sous-Répertoire source {subfolder}: {nb_data_per_class[subfolder]} fichiers ajoutés")    
        else:
            logger.debug(f"Fichier {subfolder} ignoré car pas un répertoire")    
    #4. Mise à jour du fichier csv avec les informations sur les images: Répertoire initial, répertoire cible, nom de l'image
    # Fichier csv, ajout du header
    csv_output=os.path.join(dataset_output,"metadata.csv")
    logger.debug(f"Chemin du fichier csv {csv_output}")
    with open(csv_output, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        # Entête du fichier csv: Sous-répertoire source, Classe (Sous-répertoire source), Sous-répertoire cible, Nom du fichier, Date d'ajout
        writer.writeheader()
         # Write the data rows
        for row in data:
            logger.debug(f"")
            writer.writerow(row)

    #5. Après parcours de tous les sous-répertoires, mise à jour du fichier metadata.json
    description=f"This dataset contains data that targets {num_classes} classes"
    version=version
    current_date = datetime.now()
    # Format the date as YYYY-MM-DD
    creation_date = current_date.strftime("%Y-%m-%d")
    size=util_get_size(dataset_output)
    num_classes=num_classes
    logger.debug(f"Nombre d'images par classe: {nb_data_per_class}")
    nb_sample_per_class=str(nb_data_per_class)# num fichier dans subfolder
    class_types=list(nb_data_per_class.keys()) # nom des subfolders
    last_modification=creation_date

    json_dict={}
    json_dict["dataset_name"]=dataset_info["dataset_name"]+" MC"
    json_dict["description"]=description
    json_dict["version"]=version
    json_dict["creation_date"]=creation_date
    json_dict["size"]=size
    json_dict["num_classes"]=num_classes
    json_dict["nb_samples_per_class"]=nb_sample_per_class
    json_dict["classes_types"]=class_types
    json_dict["last_modification"]=last_modification
    print(f"json dict {json_dict}")
    
    # Ecrire les informations dans un metadata.json qui sera au même niveau que le répertoire principal du dataset nettoyé
    metadata_filename=os.path.join(dataset_output, "metadata.json")
    with open(metadata_filename, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)
    
    return

def clean_data_multiclasse_select_image(path_to_data,dataset_output,version,num_images_per_class):
    logger.debug(f"---------------clean_data_multiclasse(path_to_data={path_to_data}, dataset_output={dataset_output},version={version}---------------")
    
    """
    Nettoyage des données
    1. Création du répertoire destination dans processed
    2a. Création des répertoire source dans destination après avoir supprimé l'espace
    2b. Copie du contenu de sous-répertoires/images dans le nouveau sous-répertoire
    2c. Mise à jour du fichier csv avec les informations sur les images: Répertoire initial, répertoire cible, nom de l'image
    3. Après parcours de tous les sous-répertoires, mise à jour du fichier metadata.json
    """
    
    # 1. Création du répertoire destination dans processed
    logger.info("Création du répertoire destination dans processed")
    os.makedirs(dataset_output, exist_ok=True)
    # Initialisation de l'objet data qui contient toutes les lignes
    data=[]
    num_classes=0
    nb_data_per_class={}
    for subfolder in os.listdir(path_to_data):
        # Vérifier s'il s'agit d'un sous-répertoire
        if os.path.isdir(os.path.join(path_to_data, subfolder)):
            num_classes+=1
            logger.debug(f"Traitement du sous-répertoire {subfolder}")
            logger.debug("Utilisation des images du sous-répertoire images")
            full_sub_rep=os.path.join(path_to_data, subfolder,"images")
            logger.debug(f"Chemin complet du sous-répertoire {full_sub_rep}")
            logger.debug(f"Parcours du Sous-répertoire {subfolder}")
            # renommer le sous-fichier avant de le créer dans le répertoire de destination
            new_folder_name=util_remove_space_from_foldername(subfolder)
            logger.debug(f"Nouveau nom de sous-répertoire {new_folder_name}")
            full_target_subrep=os.path.join(dataset_output, new_folder_name)
            logger.debug(f"Chemin complet du nouveau sous-répertoire {full_target_subrep}")
            # 3a. Création des répertoire source dans destination après avoir supprimé l'espace
            logger.debug(f"Création du sous-répertoire {full_target_subrep}")
            os.makedirs(full_target_subrep, exist_ok=True)
            # 3b. Copie du contenu de sous-répertoires/images dans le nouveau sous-répertoire
            nb_data=0
            all_files = [filename for filename in os.listdir(full_sub_rep) if os.path.isfile(os.path.join(full_sub_rep, filename)) and filename != ".DS_Store"]
            selected_files = random.sample(all_files, min(num_images_per_class, len(all_files)))

            for filename in selected_files:
                full_filename_path = os.path.join(full_sub_rep, filename)
                nb_data += 1
                logger.debug(f"Copie du fichier {filename} dans le sous-répertoire {full_target_subrep}")
                shutil.copy(full_filename_path, full_target_subrep)

                # 3c. création d'un dictionnaire des informations sur le fichier
                file_info = {}
                file_info['Sous-répertoire SOURCE'] = subfolder
                file_info['Classe'] = subfolder
                file_info['Sous-répertoire CIBLE'] = new_folder_name
                file_info['Nom de fichier'] = filename
                file_info['Date d\'ajout'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.debug(f"Informations sur le fichier {file_info}")

                # 3c. Mise à jour de l'objet data qui sera ajouté dans le fichier csv
                data.append(file_info)
                logger.debug("Informations du fichier ajouté dans le batch data pour le fichier csv")
            # ajout des informations du nombre d'image pour chaque nouvelle classe
            logger.debug(f"Nombre d'images trouvés dans le sous-répertoire source: {nb_data}")
            nb_data_per_class[subfolder]= nb_data    
            logger.debug(f"Sous-Répertoire source {subfolder}: {nb_data_per_class[subfolder]} fichiers ajoutés")    
        else:
            logger.debug(f"Fichier {subfolder} ignoré car pas un répertoire")    
    #4. Mise à jour du fichier csv avec les informations sur les images: Répertoire initial, répertoire cible, nom de l'image
    # Fichier csv, ajout du header
    csv_output=os.path.join(dataset_output,"metadata.csv")
    logger.debug(f"Chemin du fichier csv {csv_output}")
    with open(csv_output, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        # Entête du fichier csv: Sous-répertoire source, Classe (Sous-répertoire source), Sous-répertoire cible, Nom du fichier, Date d'ajout
        writer.writeheader()
         # Write the data rows
        for row in data:
            logger.debug(f"")
            writer.writerow(row)

    #5. Après parcours de tous les sous-répertoires, mise à jour du fichier metadata.json
    description=f"This dataset contains data that targets {num_classes} classes"
    version=version
    current_date = datetime.now()
    # Format the date as YYYY-MM-DD
    creation_date = current_date.strftime("%Y-%m-%d")
    size=util_get_size(dataset_output)
    num_classes=num_classes
    logger.debug(f"Nombre d'images par classe: {nb_data_per_class}")
    nb_sample_per_class=str(nb_data_per_class)# num fichier dans subfolder
    class_types=list(nb_data_per_class.keys()) # nom des subfolders
    last_modification=creation_date

    json_dict={}
    json_dict["dataset_name"]=dataset_info["dataset_name"]+" MC"
    json_dict["description"]=description
    json_dict["version"]=version
    json_dict["creation_date"]=creation_date
    json_dict["size"]=size
    json_dict["num_classes"]=num_classes
    json_dict["nb_samples_per_class"]=nb_sample_per_class
    json_dict["classes_types"]=class_types
    json_dict["last_modification"]=last_modification
    print(f"json dict {json_dict}")
    
    # Ecrire les informations dans un metadata.json qui sera au même niveau que le répertoire principal du dataset nettoyé
    metadata_filename=os.path.join(dataset_output, "metadata.json")
    with open(metadata_filename, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)
    
    return

def clean_data_3classes_toutes_images(path_to_data,dataset_output,version):
    logger.debug(f"---------------clean_data_multiclasse(path_to_data={path_to_data}, dataset_output={dataset_output},version={version}---------------")
    
    """
    Nettoyage des données pour :
        - Utiliser uniquement les images sans les masques
        - Exclure Lung_Opacity
        - Renommer les noms des classes sans espace (remplacés par _)
    1. Création du répertoire destination dans processed
    2a. Création des répertoire source dans destination après avoir supprimé l'espace
    2b. Copie du contenu de sous-répertoires/images dans le nouveau sous-répertoire
    2c. Mise à jour du fichier csv avec les informations sur les images: Répertoire initial, répertoire cible, nom de l'image
    3. Après parcours de tous les sous-répertoires, mise à jour du fichier metadata.json
    """
    
    # 1. Création du répertoire destination dans processed
    logger.info("Création du répertoire destination dans processed")
    os.makedirs(dataset_output, exist_ok=True)
    # Initialisation de l'objet data qui contient toutes les lignes
    data=[]
    num_classes=0
    nb_data_per_class={}
    for subfolder in os.listdir(path_to_data):
        # Vérifier s'il s'agit d'un sous-répertoire
        if os.path.isdir(os.path.join(path_to_data, subfolder)) and 'lung'  not in subfolder.lower():
            num_classes+=1
            logger.debug(f"Traitement du sous-répertoire {subfolder}")
            logger.debug("Utilisation des images du sous-répertoire images")
            full_sub_rep=os.path.join(path_to_data, subfolder,"images")
            logger.debug(f"Chemin complet du sous-répertoire {full_sub_rep}")
            logger.debug(f"Parcours du Sous-répertoire {subfolder}")
            # renommer le sous-fichier avant de le créer dans le répertoire de destination
            new_folder_name=util_remove_space_from_foldername(subfolder)
            logger.debug(f"Nouveau nom de sous-répertoire {new_folder_name}")
            full_target_subrep=os.path.join(dataset_output, new_folder_name)
            logger.debug(f"Chemin complet du nouveau sous-répertoire {full_target_subrep}")
            # 3a. Création des répertoire source dans destination après avoir supprimé l'espace
            logger.debug(f"Création du sous-répertoire {full_target_subrep}")
            os.makedirs(full_target_subrep, exist_ok=True)
            # 3b. Copie du contenu de sous-répertoires/images dans le nouveau sous-répertoire
            nb_data=0
            for filename in os.listdir(full_sub_rep):
                full_filename_path=os.path.join(full_sub_rep, filename)
                nb_data+=1
                if os.path.isfile(full_filename_path) and filename!=".DS_Store":
                    logger.debug(f"Copie du fichier {filename} dans le sous-répertoire {full_target_subrep}")
                    shutil.copy(os.path.join(full_sub_rep, filename), full_target_subrep)
                    # 3c. création d'un dictionnaire des informations sur le fichier
                    file_info={}
                    file_info['Sous-répertoire SOURCE']=subfolder
                    file_info['Classe']=subfolder
                    file_info['Sous-répertoire CIBLE']=new_folder_name
                    file_info['Nom de fichier']=filename
                    file_info['Date d\'ajout']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logger.debug(f"Informations sur le fichier {file_info}")
                    # 3c. Mise à jour de l'objet data qui sera ajouté dans le fichier csv
                    data.append(file_info)
                    logger.debug("Informations du fichier ajouté dans le batch data pour le fichier csv")
            # ajout des informations du nombre d'image pour chaque nouvelle classe
            logger.debug(f"Nombre d'images trouvés dans le sous-répertoire source: {nb_data}")
            nb_data_per_class[subfolder]= nb_data    
            logger.debug(f"Sous-Répertoire source {subfolder}: {nb_data_per_class[subfolder]} fichiers ajoutés")    
        else:
            logger.debug(f"Fichier {subfolder} ignoré car pas un répertoire")    
    #4. Mise à jour du fichier csv avec les informations sur les images: Répertoire initial, répertoire cible, nom de l'image
    # Fichier csv, ajout du header
    csv_output=os.path.join(dataset_output,"metadata.csv")
    logger.debug(f"Chemin du fichier csv {csv_output}")
    with open(csv_output, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        # Entête du fichier csv: Sous-répertoire source, Classe (Sous-répertoire source), Sous-répertoire cible, Nom du fichier, Date d'ajout
        writer.writeheader()
         # Write the data rows
        for row in data:
            logger.debug(f"")
            writer.writerow(row)

    #5. Après parcours de tous les sous-répertoires, mise à jour du fichier metadata.json
    description=f"This dataset contains data that targets {num_classes} classes"
    version=version
    current_date = datetime.now()
    # Format the date as YYYY-MM-DD
    creation_date = current_date.strftime("%Y-%m-%d")
    size=util_get_size(dataset_output)
    num_classes=num_classes
    logger.debug(f"Nombre d'images par classe: {nb_data_per_class}")
    nb_sample_per_class=str(nb_data_per_class)# num fichier dans subfolder
    class_types=list(nb_data_per_class.keys()) # nom des subfolders
    last_modification=creation_date

    json_dict={}
    json_dict["dataset_name"]=dataset_info["dataset_name"]+" 3C"
    json_dict["description"]=description
    json_dict["version"]=version
    json_dict["creation_date"]=creation_date
    json_dict["size"]=size
    json_dict["num_classes"]=num_classes
    json_dict["nb_samples_per_class"]=nb_sample_per_class
    json_dict["classes_types"]=class_types
    json_dict["last_modification"]=last_modification
    print(f"json dict {json_dict}")
    
    # Ecrire les informations dans un metadata.json qui sera au même niveau que le répertoire principal du dataset nettoyé
    metadata_filename=os.path.join(dataset_output, "metadata.json")
    with open(metadata_filename, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)
    
    return

def clean_data_3classes_select_image(path_to_data,dataset_output,version,num_images_per_class):
    logger.debug(f"---------------clean_data_multiclasse(path_to_data={path_to_data}, dataset_output={dataset_output},version={version}---------------")
    
    """
     Nettoyage des données pour :
        - Utiliser uniquement les images sans les masques
        - Exclure Lung_Opacity
        - Renommer les noms des classes sans espace (remplacés par _)
    1. Création du répertoire destination dans processed
    2a. Création des répertoire source dans destination après avoir supprimé l'espace
    2b. Copie du contenu de sous-répertoires/images dans le nouveau sous-répertoire
    2c. Mise à jour du fichier csv avec les informations sur les images: Répertoire initial, répertoire cible, nom de l'image
    3. Après parcours de tous les sous-répertoires, mise à jour du fichier metadata.json
    """
    
    # 1. Création du répertoire destination dans processed
    logger.info("Création du répertoire destination dans processed")
    os.makedirs(dataset_output, exist_ok=True)
    # Initialisation de l'objet data qui contient toutes les lignes
    data=[]
    num_classes=0
    nb_data_per_class={}
    for subfolder in os.listdir(path_to_data):
        # Vérifier s'il s'agit d'un sous-répertoire
        if os.path.isdir(os.path.join(path_to_data, subfolder)) and 'lung'  not in subfolder.lower():
            num_classes+=1
            logger.debug(f"Traitement du sous-répertoire {subfolder}")
            logger.debug("Utilisation des images du sous-répertoire images")
            full_sub_rep=os.path.join(path_to_data, subfolder,"images")
            logger.debug(f"Chemin complet du sous-répertoire {full_sub_rep}")
            logger.debug(f"Parcours du Sous-répertoire {subfolder}")
            # renommer le sous-fichier avant de le créer dans le répertoire de destination
            new_folder_name=util_remove_space_from_foldername(subfolder)
            logger.debug(f"Nouveau nom de sous-répertoire {new_folder_name}")
            full_target_subrep=os.path.join(dataset_output, new_folder_name)
            logger.debug(f"Chemin complet du nouveau sous-répertoire {full_target_subrep}")
            # 3a. Création des répertoire source dans destination après avoir supprimé l'espace
            logger.debug(f"Création du sous-répertoire {full_target_subrep}")
            os.makedirs(full_target_subrep, exist_ok=True)
            # 3b. Copie du contenu de sous-répertoires/images dans le nouveau sous-répertoire
            nb_data=0
            all_files = [filename for filename in os.listdir(full_sub_rep) if os.path.isfile(os.path.join(full_sub_rep, filename)) and filename != ".DS_Store"]
            selected_files = random.sample(all_files, min(num_images_per_class, len(all_files)))

            for filename in selected_files:
                full_filename_path = os.path.join(full_sub_rep, filename)
                nb_data += 1
                logger.debug(f"Copie du fichier {filename} dans le sous-répertoire {full_target_subrep}")
                shutil.copy(full_filename_path, full_target_subrep)

                # 3c. création d'un dictionnaire des informations sur le fichier
                file_info = {}
                file_info['Sous-répertoire SOURCE'] = subfolder
                file_info['Classe'] = subfolder
                file_info['Sous-répertoire CIBLE'] = new_folder_name
                file_info['Nom de fichier'] = filename
                file_info['Date d\'ajout'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.debug(f"Informations sur le fichier {file_info}")

                # 3c. Mise à jour de l'objet data qui sera ajouté dans le fichier csv
                data.append(file_info)
                logger.debug("Informations du fichier ajouté dans le batch data pour le fichier csv")
            # ajout des informations du nombre d'image pour chaque nouvelle classe
            logger.debug(f"Nombre d'images trouvés dans le sous-répertoire source: {nb_data}")
            nb_data_per_class[subfolder]= nb_data    
            logger.debug(f"Sous-Répertoire source {subfolder}: {nb_data_per_class[subfolder]} fichiers ajoutés")    
        else:
            logger.debug(f"Fichier {subfolder} ignoré car pas un répertoire")    
    #4. Mise à jour du fichier csv avec les informations sur les images: Répertoire initial, répertoire cible, nom de l'image
    # Fichier csv, ajout du header
    csv_output=os.path.join(dataset_output,"metadata.csv")
    logger.debug(f"Chemin du fichier csv {csv_output}")
    with open(csv_output, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        # Entête du fichier csv: Sous-répertoire source, Classe (Sous-répertoire source), Sous-répertoire cible, Nom du fichier, Date d'ajout
        writer.writeheader()
         # Write the data rows
        for row in data:
            logger.debug(f"")
            writer.writerow(row)

    #5. Après parcours de tous les sous-répertoires, mise à jour du fichier metadata.json
    description=f"This dataset contains data that targets {num_classes} classes"
    version=version
    current_date = datetime.now()
    # Format the date as YYYY-MM-DD
    creation_date = current_date.strftime("%Y-%m-%d")
    size=util_get_size(dataset_output)
    num_classes=num_classes
    logger.debug(f"Nombre d'images par classe: {nb_data_per_class}")
    nb_sample_per_class=str(nb_data_per_class)# num fichier dans subfolder
    class_types=list(nb_data_per_class.keys()) # nom des subfolders
    last_modification=creation_date

    json_dict={}
    json_dict["dataset_name"]=dataset_info["dataset_name"]+" 3C"
    json_dict["description"]=description
    json_dict["version"]=version
    json_dict["creation_date"]=creation_date
    json_dict["size"]=size
    json_dict["num_classes"]=num_classes
    json_dict["nb_samples_per_class"]=nb_sample_per_class
    json_dict["classes_types"]=class_types
    json_dict["last_modification"]=last_modification
    print(f"json dict {json_dict}")
    
    # Ecrire les informations dans un metadata.json qui sera au même niveau que le répertoire principal du dataset nettoyé
    metadata_filename=os.path.join(dataset_output, "metadata.json")
    with open(metadata_filename, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)
    
    return

def util_remove_space_from_foldername(folder_name):
    """
    Fonction qui harmonise les noms des répertoires en supprimant les espaces remplacés par _
    A terme, on peut aussi mettre en minuscule les noms de répertoires
    Le nouveau nom de répertoire servira aussi de nom de classe
    """
    return folder_name.replace(" ", "_")
    
def util_get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return util_convert_size(total_size)

def util_convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

if __name__ == "__main__":
    version="1.0"
    path_to_data=os.path.join(init_paths["main_path"],init_paths["raw_datasets_folder"],"covid19-db","COVID-19_Radiography_Dataset")
    dataset_output=os.path.join(init_paths["main_path"],init_paths["processed_datasets_folder"],f"COVID-19_MC_{version}")
    clean_data_multiclasse_toutes_images(path_to_data,dataset_output,version)
    version="1.1"
    path_to_data=os.path.join(init_paths["main_path"],init_paths["raw_datasets_folder"],"covid19-db","COVID-19_Radiography_Dataset")
    dataset_output=os.path.join(init_paths["main_path"],init_paths["processed_datasets_folder"],f"COVID-19_MC_{version}")
    clean_data_multiclasse_select_image(path_to_data,dataset_output,version,20)

    version="1.2"
    path_to_data=os.path.join(init_paths["main_path"],init_paths["raw_datasets_folder"],"covid19-db","COVID-19_Radiography_Dataset")
    dataset_output=os.path.join(init_paths["main_path"],init_paths["processed_datasets_folder"],f"COVID-19_MC_{version}")
    clean_data_3classes_toutes_images(path_to_data,dataset_output,version)
    version="1.3"
    path_to_data=os.path.join(init_paths["main_path"],init_paths["raw_datasets_folder"],"covid19-db","COVID-19_Radiography_Dataset")
    dataset_output=os.path.join(init_paths["main_path"],init_paths["processed_datasets_folder"],f"COVID-19_MC_{version}")
    clean_data_3classes_select_image(path_to_data,dataset_output,version,20)