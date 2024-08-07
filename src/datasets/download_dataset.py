'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Fichier de téléchargement de la base de données
Appel de l'API Kaggle

'''

### Import des modules
import os

# import information logger et fichier de configuration
from src.config.run_config import init_paths,dataset_info,eval_dataset_info
from src.config.log_config import logger

# import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

### Fin des imports

def get_dataset_kaggle_api(url,destination):
    """
    Télécharge et extrait un dataset depuis Kaggle vers un dossier de destination.

    Params:
        url (str): L'URL du dataset Kaggle à télécharger.
        destination (str): Le chemin du dossier où extraire le contenu du dataset.

    Raises:
        Exception: En cas d'erreur non gérée.
    """
    
    logger.debug(f"---------------get_dataset_kaggle_api(url={url}, destination={destination}---------------")
    
    try:
        # Initialiser l'API Kaggle
        api = KaggleApi()
        api.authenticate()
        
        temporaire = os.path.join(init_paths["main_path"],init_paths["temp_datasets_folder"]) # ./data/raw/datasets_temp
        logger.debug(f"Création du Répertoire temporaire  {temporaire}")
        # On crée le dossier temporaire pour télécharger le fichier
        if not os.path.exists(temporaire):
            os.makedirs(temporaire)
    
        # On crée le dossier du dataset s'il n'existe pas
        if not os.path.exists(destination):
            os.makedirs(destination)
        
        logger.debug(f"Téléchargement du dataset {url}")
        # Télécharger le dataset
        api.dataset_download_files(url, path=destination, unzip=True)
        
        logger.info(f"Dataset copié dans le répertoire {destination}")

        return
        
    except Exception as e:
        logger.error(f"Une erreur s'est produite : {e}")
        return
    
## Fonction principale
def main():
    # URL avec appel à kaggleAPI
    url = eval_dataset_info["eval_dataset_url"] 
    destination = os.path.join(init_paths["main_path"],init_paths["raw_datasets_folder"]) # ./data/raw/datasets
    
    get_dataset_kaggle_api(url, destination)
    
if __name__ == "__main__":
    main()