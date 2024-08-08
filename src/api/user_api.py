'''
Créé le 07/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: API pour la mise à disposition de services aux utilisateurs
-- Gestion du compte
    -- Inscription (à venir) 
    -- Authentification (Avec Profil simple ou admin)
    -- Modification du mot de passe (à venir)
    -- Suppression du compte (à venir)
-- Prédiction
    -- Historique des prédictions de l'utilisateur (à venir)
    -- Exécution d'une prédiction et visualisation du résultat avec indice de confiance
    -- Action: Valider/Invalider/Modifier la prédiction
'''

## Imports

## FastAPI
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse,JSONResponse

## Utiles
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
from dotenv import load_dotenv
import jwt
import os
import json

### Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths,model_info,infolog
from src.config.log_config import logger
from src.utils import utils_models
### Import de modules internes: ici la fonction de prediction et log
from src.models.predict_model import predict_and_log
from src.datasets import update_dataset
# Import temporaire pour l'exécution de l'API depuis le fichier python
import uvicorn

## 1 - Initialisation de l'API FastAPI avec des informations de bases
app = FastAPI(
    title="Détection d'anomalies Pulmonaires",
    description="API pour la détection d'anomalies Pulmonaires COVID et Pneumonia Virale.",
    version="0.1"
)

## 2 - Créer les chemins d'accès aux fichiers et dossiers
# Nom du modèle à utiliser
model_path=os.path.join(init_paths["main_path"],init_paths["models_path"],model_info["selected_model_name"])
img_storage_path=os.path.join(init_paths["main_path"],init_paths["prediction_images_folder"])
prediction_logging_filepath=os.path.join(init_paths["main_path"],init_paths["prediction_logging_folder"],model_info["prediction_logging_filename"])

## 3 - Chargement du modèle une fois
model_name=model_info["selected_model_name"]
model=utils_models.load_model(model_path)

## 4 - Charger les variables d'environnement
load_dotenv()

## 5 - Initier les principas URL qui seront utilisées dans l'API basées sur les SPECS
@app.get("/", summary="Health Check", description="Vérification de l'état de l'application")
async def health_check():
    """
    Point d'accès utilisé pour vérifier l'état de l'API (démarré, bon fonctionnement)
    :return 200 si l'API est opérationnelle
    """
    return JSONResponse(status_code=200, content={"status": "OK"})

@app.post("/predict",summary="Prédiction sur une image", description="Evaluation de l'état pulmonaire basé sur une image")
async def predict(image:UploadFile = File(...)):
    # Temporairement sans authentification
    logger.debug("---------------user_api: /predict---------------") 
    image_original_name = image.filename
    # construction du nom du fichier basé sur la date courante
    logger.debug(f"Image reçue: {image_original_name}")
    logger.debug("Construction du nom du fichier basé sur la date courante ")
    current_time = datetime.now()
    # Formater la date et l'heure pour les inclure dans le nom du fichier
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    # Construire le chemin avec le nouveau nom de fichier
    image_path=os.path.join(img_storage_path,f"upload_{formatted_time}_{image_original_name}")
    logger.debug(f"Chemin complet du fichier en local {image_path}")
    # Enregistrer l'image dans le dossier des Upload
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())
        logger.debug(f"Image sauvegardée à: {image_path}")
    # Exécuter la prédiction
    prediction,confiance,temps_prediction = predict_and_log(model,model_name,image_path,prediction_logging_filepath)
    
    #return JSONResponse(status_code=200, content={"prediction": prediction,"confiance":confiance,"temps_prediction":temps_prediction})
    return {"prediction": prediction,"confiance":confiance,"temps_prediction":temps_prediction,"image_upload_path":image_path}     

@app.post("/add_image",summary="",description="")
async def add_image(image_path:str,
                    image_label:str):
    old_dataset_version,new_dataset_version=update_dataset.add_one_or_muliple_images(image_path,image_label)
    return {"old_dataset_version":old_dataset_version,"new_dataset_version":new_dataset_version}

#@app.get("/get_classes")

if __name__ == "__main__":
    # Initialisation des chemins d'accès aux fichiers et dossiers
    uvicorn.run("user_api:app", host="0.0.0.0", port=8000, reload=True)
