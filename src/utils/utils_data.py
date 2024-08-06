'''
Créé le 06/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Fonctions utiles pour les données:
-- Correspondance des labels avec des valeurs numériques 
-- chargement d'un modèle
-- sauvegarde/chargement des résultats d'un modèle (historique, training plots)

'''
from src.config.log_config import logger

def label_to_numeric(labels, correspondance):
    '''
    Convertit les labels en valeurs numériques en utilisant une correspondance donnée.
    '''
    logger.debug(f"---------label_to_numeric(labels={labels}, correspondance={correspondance})")
    res=[correspondance[label] for label in labels]
    logger.debug(f"res={res}")
    return res

def numeric_to_label(numbers, correspondance):
    '''
    Convertit les valeurs numériques en labels en utilisant une correspondance donnée.
    '''
    logger.debug(f"---------numeric_to_label(numbers={numbers}, correspondance={correspondance})")
    res=[correspondance[number] for number in numbers]
    logger.debug(f"res={res}")
    return res

def generate_numeric_correspondance(labels):
    '''
    Génère une correspondance entre les labels et des valeurs numériques uniques.
    '''
    logger.debug(f"---------generate_numeric_correspondance(labels={labels})")
    unique_labels = list(set(labels))
    logger.debug(f"Unique labels {unique_labels}")
    correspondance = {label: i for i, label in enumerate(unique_labels)}
    logger.debug(f"Correspondance {correspondance}")
    return correspondance
