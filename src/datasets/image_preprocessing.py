'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Image Preprocessing

'''

#####
# Imports 
#####
# Modules de preprocessing pour le modèle EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as pp_effnet
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# useful modules
import numpy as np
import os
import csv

### Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths,infolog
from src.config.log_config import logger
### Functions

def preprocess_data(dataset_path,size,dim):
    logger.debug(f"--------------preprocess_data(dataset_path={dataset_path},size={size},dim={dim})")
    """
    Fonction qui prétraite les données d'image pour les modèles de réseau de neurones.
    Le fichier metadata.csv présent dans le dataset permet d'identifier le label qui peut être différent du nom du sous-répertoire.
    :param img: L'objet image à preprocesser.
    :param size: Entier, la taille cible à laquelle chaque image est redimensionnée. Pour efficientNetB0, ce sera 224 
    :param dim: Entier, le nombre de dimensions de couleur requis (1 pour les niveaux de gris, 3 pour RGB). Pour efficientNetB0, le dim sera 1
    
    Applique la fonction de preprocessing de l'architecture EfficientNetB0.
    On isole cette partie dans une fonction séparé pour le rendre scalable en cas de changement de choix d'architecture (resnet etc.)
    :return les objets images et labels associés
    :raises Exception: Si :
        - FileNotFoundError Le fichier metadata.csv est erroné
        - Le fichier cherché n'est pas présent
        - ou toute autre erreur
    """

    # Initialisation des listes pour stocker les images et les étiquettes
    data = []

    # Parcourt du dataset et lecture du fichier metadata.csv
    try:
        with open(os.path.join(dataset_path, 'metadata.csv'), 'r') as f:
            # Ignore la première ligne (en-tête)
            reader = csv.DictReader(f)
            #next(f)
            # Récupérer le label de la colonne "Classe" et le chemin de l'image dans "Sous-répertoire CIBLE" et "Nom de fichier"
            for row in reader:
                logger.debug(f"row {row}") 
                label = row["Classe"]
                rep_src = row["Sous-répertoire CIBLE"]
                img_name = row["Nom de fichier"]
                img_path= os.path.join(dataset_path,rep_src,img_name)
                logger.debug(f"label={label}")
                logger.debug(f"rep_src={rep_src}")
                logger.debug(f"img_name={img_name}")
                logger.debug(f"img_path={img_path}")
                try:
                    img = load_img(img_path, target_size=(size, size), color_mode='grayscale' if dim == 1 else 'rgb')
                    img_array = img_to_array(img)
                    img_array = preprocess_input(img_array)
                    data.append((img_array, label))
                except Exception as e:
                    logger.error(f"Erreur de processing sur l'image {img_path}: {str(e)}")
                    raise Exception(f"Erreur de processing sur l'image {img_path}: {str(e)}")
            logger.info("Images preprocessés et retournés avec leur labels correspondant")
            logger.debug(f"Taille data {len(data)}")
            return data
    except FileNotFoundError:
        logger.error(f"Le fichier metadata.csv n'a pas été trouvé dans le répertoire {dataset_path}")
        raise FileNotFoundError()
    






def preprocess_data_init(image_paths, labels, size, dim):
    """
    Fonction de prétraitement des données d'image pour les modèles de réseaux de neurones.

    :param image_paths: Liste des chemins de fichiers vers les images.
    :param labels: Liste des classes correspondant à chaque image. Utilisé lorsqu'on souhaite labelliser autrement un sous-répertoire 
    Par exemple pour une classification binaire sain/malade ou normal est labellisé sain, et covid/viral_pneumonia/Lung_opacity sont labellisés malade
    :param size: Entier, la taille cible à laquelle chaque image est redimensionnée.
    :param dim: Entier, le nombre de dimensions de couleur requis (1 pour les niveaux de gris, 3 pour RGB).
    
    :return: Une liste de tuples, chacun contenant une représentation en tableau d'une image et son étiquette correspondante.

    :raises ValueError: Si 'dim' n'est pas 1 ou 3.
    :raises FileNotFoundError: Si un chemin d'image dans 'image_paths' ne pointe pas vers un fichier existant.
    :raises Exception: Pour tout autre problème lors du chargement ou du traitement des images, comme des types de fichiers incompatibles ou des erreurs dans les fonctions de prétraitement.

    """

    logger.debug("---------------preprocess_data------------")
    if dim not in [1, 3]:
        raise ValueError("Dimension 'dim' doit être 1 (grayscale) ou 3 (RGB).")
    data = []
    for img_path, label in zip(image_paths, labels):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image path {img_path} n'existe pas.")
        try:
            img = load_img(img_path, target_size=(size, size), color_mode='grayscale' if dim == 1 else 'rgb')
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            data.append((img_array, label))
        except Exception as e:
            logger.error(f"Erreur de processing sur l'image {img_path}: {str(e)}")
            raise Exception(f"Erreur de processing sur l'image {img_path}: {str(e)}")

    return data


# Preprocessing redirection to the model's function
def preprocess_input(img):
    """
    Fonction qui prétraite les données d'image pour les modèles de réseau neuronal.

    
    :param img: L'objet image à preprocesser.
    Applique la fonction de preprocessing de l'architecture EfficientNetB0.
    On isole cette partie dans une fonction séparé pour le rendre scalable en cas de changement de choix d'architecture (resnet etc.)
    :raises ValueError: 
    """
    try:
        logger.debug(f"Preprocessing EfficientNetB0")
        img = pp_effnet(img)
        return img
    
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise


## Fonction principale
def main():
    dataset_path = os.path.join(init_paths["main_path"],init_paths["processed_datasets_folder"],"COVID-19_MC_1.3") # ./data/raw/datasets
    logger.debug("main - dataset_path{dataset_path}")
    data=preprocess_data(dataset_path,224,3)
    i=0
    for (img,label) in data:
        i+=1
        logger.debug(f" {i} - Label {label}")
if __name__ == "__main__":
    main()