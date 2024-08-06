'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Initialisation des répertoires 
Pour s'assurer qu'aucun répertoire principal ne manque pour la bonne exécution du code

'''
import os
from src.config.run_config import init_paths
from src.config.log_config import logger

def create_directories(paths):
    logger.debug(f"-------------create_directories(paths={paths})")
    for key, path in paths.items():
        full_path = os.path.join(paths["main_path"], path)
        os.makedirs(full_path, exist_ok=True)
        logger.debug(f"Created directory: {full_path}")

if __name__ == "__main__":
    create_directories(init_paths)