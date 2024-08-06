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
    "raw_datasets_folder": "data/raw/datasets",
    "processed_datasets_folder": "data/processed/datasets", 
    "test_images":"data/raw/test_images", 
    "logs_folder": "logs",
    "docker_folder":"dockerfiles",
    "airflow_dags_pipelines":"dags",
    "docs":"docs",
    "references":"references"
    }

# Base de données utilisée, URL Kaggle qui sera chargée via l'API
# Attention: Il faut récupérer un token d'accès à stocker dans un fichier .kaggle 
dataset_info = {
    "dataset_url":"tawsifurrahman/covid19-radiography-database",
    "dataset_name":"covid19-db",
}

model_info = {
    "selected_model_name":"covid19_test_arbo.h5",
    "model_name_prefix":"covid_model",
    # Users 
    
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