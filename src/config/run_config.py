'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Fichier de configuration des logs
Actuellement toutes les logs seront stockées dans le même fichier projet
A terme sera divisé par périmètre en déclarant le logger au niveau du fichier

'''

# set the absolute path
import os
import sys
sys.path.insert(0,os.path.abspath("."))

######
## PROJECT and RUN Paths and file creation information (prefix)
######

init_paths = {
    "main_path": ".",
    # tous les chemins sont basés sur main_path
    "data_folder": "data",
    "archive_folder":"data/archives",
    "models_path": "data/models",
    "temp_datasets_folder":"data/raw/datasets_temp",
    "prediction_images_folder":"data/raw/prediction_images",
    "run_folder":"data/processed/mflow_runs",
    "keras_tuner_folder":"data/processed/keras_tuner",
    "raw_datasets_folder": "data/raw/datasets",
    "processed_datasets_folder": "data/processed/datasets", 
    "prediction_logging_folder": "data/processed/predictions",
    "test_images":"data/raw/test_images", 
    "logs_folder": "logs",
    "docker_folder":"dockerfiles",
    "airflow_dags_pipelines":"dags",
    "docs":"docs",
    "references":"references",
    "notebooks":"notebooks",
    }

# Base de données utilisée, URL Kaggle qui sera chargée via l'API
# Attention: Il faut récupérer un token d'accès à stocker dans un fichier .kaggle 
dataset_info = {
    "dataset_url":"tawsifurrahman/covid19-radiography-database",
    "dataset_name":"covid19-db",
}

eval_dataset_info = {
    "eval_dataset_url":"ahmadalmahsiri/covid19-radiography-database",
    "eval_dataset_name":"eval-covid19-db",
}

model_info = {
    "selected_model_name":"COVID19_Effnetb0_Model_1.1.h5",
    "model_name_prefix":"covid_model",
    "prediction_logging_filename":"prediction_logging.csv",
    }

user_api_info = {
    "USER_API_URL":"http://127.0.0.1:8000",
    "PREDICT_URL":"/predict"
}

admin_api_info = {
    "ADMIN_API_URL":"http://127.0.0.1:8080",
}

api_info = {
    "users_path":".secrets/authorized_users.json"
}
######
## LOG management
######
infolog = {
    "project_name":"Covid19_MLOps",
    "logfile_name": "covid19-mlops.log",
}