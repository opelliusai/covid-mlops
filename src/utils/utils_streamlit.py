'''
Créé le 08/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Fonctions utiles pour streamlit:
-- 

'''

### IMPORTS
import requests

### Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths,infolog,user_api_info
from src.config.log_config import logger
import os

### Construction des URL
API_URL = user_api_info["USER_API_URL"]
## URL prédiction
API_URL_PREDICT=user_api_info["PREDICT_URL"]
API_FULL_URL_PREDICT = API_URL + API_URL_PREDICT
logger.debug(f"API_FULL_URL_PREDICT : {API_FULL_URL_PREDICT}")

def lancer_une_prediction(file,filename):
    logger.debug("----------------lancer_une_prediction(file)------------")
    #logger.debug(f"Lancement de la prédiction pour le fichier {file.filename}")
    # Essayer de se connecter pour obtenir le token
    prediction_url = API_FULL_URL_PREDICT
    logger.debug(f"prediction_url = {prediction_url}")
    files = {"image": (filename, file, "image/jpeg")}
                
#    file = {"file": (file.name, file, "application/octet-stream")}
    
    response = requests.post(prediction_url, files=files)
    
    if response.status_code == 200:
        prediction = response.json().get('prediction')
        confiance = response.json().get('confiance')
        temps_prediction=response.json().get('temps_prediction')
        image_upload_path=response.json().get('image_upload_path')
        return prediction,confiance,temps_prediction,image_upload_path
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None
 
def ajout_image_dataset(image_path,label):
    logger.debug(f"----------------ajout_image_dataset(image_path={image_path},label={label})------------")
    