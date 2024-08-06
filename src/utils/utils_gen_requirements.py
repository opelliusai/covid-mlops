'''
Créé le 05/08/2024
@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Generation de requirements.txt pour chaque dossier de source python
A DEBUGGER
'''
import os
import subprocess

### Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths,infolog
from src.config.log_config import logger
### Functions


def exec_pipreqs_folders():
    for dir in os.path.join(init_paths["main_path"],"src"):
        if os.path.isdir(dir):
            subprocess.run(["pipreqs", "--force", "--use-local", os.path.join(init_paths["main_path"], "src", dir)])


if __name__ == "__main__":
    exec_pipreqs_folders()