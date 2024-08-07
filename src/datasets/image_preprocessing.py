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
                    img_array= preprocess_one_image(img_path,size,dim)
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
 
def preprocess_one_image(img_path,size=224,dim=3):
    logger.debug(f"--------------preprocess_one_image(img_path={img_path},size={size},dim={dim})-----------")
    try:
        img = load_img(img_path, target_size=(size, size), color_mode='grayscale' if dim == 1 else 'rgb')
        img_array = img_to_array(img)
        img_array = pp_effnet(img_array)
        return img_array
    except Exception as e:
        logger.error(f"Erreur de processing sur l'image {img_path}: {str(e)}")
        raise Exception(f"Erreur de processing sur l'image {img_path}: {str(e)}")    

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