'''
Créé le 07/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: API d'administration 
-- Etend l'API utilisateur avec des fonctionnalités d'administration

-- 
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
from src.config.run_config import init_paths,infolog
from src.config.log_config import logger
from src.utils import utils_data
### Import de modules internes: ici la fonction de prediction et log
from src.models.predict_model import predict_and_log

# Import temporaire pour l'exécution de l'API depuis le fichier python
import uvicorn

## 1 - Initialisation de l'API FastAPI avec des informations de bases
app = FastAPI(
    title="Détection d'anomalies Pulmonaires - API Admin",
    description="API d'administration.",
    version="0.1"
)

## 2 - Créer les chemins d'accès aux fichiers et dossiers

## 3 - Charger les variables d'environnement
load_dotenv()

## 4 - Initier les principas URL qui seront utilisées dans l'API basées sur les SPECS

### Health Check
@app.get("/", summary="Health Check", description="Vérification de l'état de l'application")
async def health_check():
    """
    Point d'accès utilisé pour vérifier l'état de l'API (démarré, bon fonctionnement)
    :return 200 si l'API est opérationnelle
    """
    return JSONResponse(status_code=200, content={"status": "OK"})

### 

### 
#@app.post("/add_image")

### 
#@app.get("/get_classes")

if __name__ == "__main__":
    # Initialisation des chemins d'accès aux fichiers et dossiers
    uvicorn.run("admin_api:app", host="0.0.0.0", port=8080, reload=True)
