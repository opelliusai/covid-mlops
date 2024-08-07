'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Evaluation et exécution de prédictions

'''

#####
# Imports 
#####
## Useful modules import
import pandas as pd

## Internal imports
from src.utils import utils_models
from src.utils import utils_data
from src.datasets import image_preprocessing
from datetime import datetime
import os
### Configuration file import
'''
src.config file contains information of the repository
paths[]: Folders and subfolders to find or generate information ex : paths['main'] is the project path
infolog[]: Logging information : Folder, logname ex : utils_models.log
'''
from src.config.log_config import logger
from src.config.run_config import init_paths,model_info
from keras.optimizers import Adam
from keras.models import load_model
import time
import numpy as np
# Model evaluation function
def evaluate_model(model, X_eval, y_eval, num_classes, class_mapping):
    """
    Evaluates the model and returns metrics, confusion matrix, and classification report.
    
    :param model: Model to be evaluated.
    :param X_eval: Evaluation images.
    :param y_eval: Evaluation targets.
    :param num_classes: Number of classes, defining binary or multiclass classification.
    :param class_mapping: Dictionary for class mapping.
    
    :return: A tuple containing metrics dictionary, confusion matrix dataframe, and classification report dataframe.
             Confusion matrix and Classification report will be saved as files by MLFlow.
             Metrics will be stored in the RUN.

    :raises Exception: If an error occurs during the model evaluation process.
    """

    try:
        # 1- Retrieving additional metrics. 
        # Specific functions have been created for binary and multiple classification because of the high divergence of the metrics generation and management
        metrics_dict = utils_data.bin_get_prediction_metrics(model, X_eval, y_eval) if num_classes == 1 else utils_data.multi_get_prediction_metrics(model, X_eval, y_eval)
        logger.debug(f"metrics_dict {metrics_dict}")
        # Building the metrics dictionary
        final_metrics = {
            "Accuracy": metrics_dict["Accuracy"],
            "Recall": metrics_dict["Recall"],
            "F1-score": metrics_dict["F1-score"]
        }
        '''
        for i, (sensitivity, specificity) in enumerate(zip(metrics_dict["Sensitivity - Recall"], metrics_dict["Specificity"])):
            final_metrics[f'Recall sensitivity_class_{i}'] = sensitivity
            final_metrics[f'Recall specificity_class_{i}'] = specificity
        '''
        logger.debug(f"Metrics: {final_metrics}")
        logger.debug(f"Confusion Matrix: {metrics_dict['Confusion Matrix']}")
        logger.debug(f"Classification Report: {metrics_dict['Classification report']}")

        # Confusion matrix as dataframe
        conf_matrix_df = pd.DataFrame(metrics_dict["Confusion Matrix"], index=[i for i in class_mapping], columns=[i for i in class_mapping])
        # Classification report as dataframe
        class_report_df = pd.DataFrame(metrics_dict["Classification report"]).transpose()
        
        return final_metrics, conf_matrix_df, class_report_df

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise Exception(f"Error during model evaluation: {e}")

def predict_and_log(model,model_name,image_path,prediction_logging_filepath):
    logger.debug(f"------------predict_and_log(model,model_name={model_name},image_path={image_path},prediction_logging_filepath={prediction_logging_filepath})----------")
    '''
    Exécution de la prédiction sur une image et enregistrement de la prédiction dans un fichier pour l'évaluation des performances
    '''
    image_obj=image_preprocessing.preprocess_one_image(image_path)
    logger.debug("Dimensions de l'image chargée :", image_obj.shape)
    logger.debug("Redimensionnement de l'image")
    image_obj = np.expand_dims(image_obj, axis=0)
    logger.debug("Lancement de la prédiction")
    prediction,confiance,temps_prediction=predict_one_image(model,image_obj)
    logger.debug(f"prediction {prediction}")
    logger.debug(f"confiance {confiance}")
    logger.debug(f"temps_prediction {temps_prediction}")
    # Enregistrement de la prédiction dans un fichier
    date_prediction=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    utils_models.save_prediction(model_name,image_path,prediction,confiance,temps_prediction,date_prediction,prediction_logging_filepath)
    return prediction,confiance,temps_prediction

def predict_one_image(model,image_obj):
    '''
    Exécution de la prédiction et retour de la prediction (numérique)
    '''
    logger.debug("----------predict(model,image_obj)---------")
    ## Début du compteur
    start_time=time.time()
    
    # Exécution de la prédiction
    predictions_list= model.predict(image_obj)[0]
    logger.debug(f"predictions_list {predictions_list}")
    correspondance={'Viral Pneumonia': 0, 'COVID': 1, 'Normal': 2}
    correspondance_num=utils_data.invert_dict(correspondance)
    ## Fin du compteur
    end_time=time.time()
    # Calcul du temps de prédiction
    temps_prediction=round(end_time-start_time,2)
    logger.debug(f"Prediction: {predictions_list}")
    indice_max = np.argmax(predictions_list)
    print("Indice de la valeur maximale :", indice_max) 
    confiance_prediction=f"{round(predictions_list[indice_max]*100,2)}"" %" ## A implementer
    logger.debug(f"confiance_prediction: {confiance_prediction}")
    prediction_label=correspondance_num[indice_max]
    logger.debug(f"prediction_label {prediction_label}")
    logger.debug(f"temps_prediction {temps_prediction} secondes")
    return prediction_label,confiance_prediction,temps_prediction

def main():
    #chemin des modeles
    model_path=os.path.join(init_paths["main_path"],init_paths["models_path"],model_info["selected_model_name"])
    prediction_logging_filepath=os.path.join(init_paths["main_path"],init_paths["prediction_logging_folder"],model_info["prediction_logging_filename"])
    logger.debug("-------predict model : paths")
    logger.debug(f"model_path {model_path}")
    logger.debug(f"prediction_logging_filepath {prediction_logging_filepath}")
    logger.debug(f"Chargement du modèle {model_info['selected_model_name']}")
    model=utils_models.load_model(model_path)
    #custom_objects = {'Adam': Adam}
    #model = load_model(model_path, custom_objects=custom_objects)
    logger.debug(f"Model chargé - Lancement de la prédiction")
    image_path=os.path.join(init_paths["main_path"],init_paths["test_images"],"COVID-1.png")
    logger.debug(f"image_path {image_path}")
    prediction,confiance,temps_prediction=predict_and_log(model,model_info["selected_model_name"],image_path,prediction_logging_filepath)
    logger.debug(f"Predictions pour l'image {image_path}")
    logger.debug(f"prediction {prediction}")
    logger.debug(f"confiance {confiance}")
    logger.debug(f"temps_prediction {temps_prediction}")    
    

    logger.debug(f"Model chargé - Lancement de la prédiction")
    image_path=os.path.join(init_paths["main_path"],init_paths["test_images"],"COVID-2.png")
    logger.debug(f"image_path {image_path}")
    prediction,confiance,temps_prediction=predict_and_log(model,model_info["selected_model_name"],image_path,prediction_logging_filepath)
    logger.debug(f"Predictions pour l'image {image_path}")
    logger.debug(f"prediction {prediction}")
    logger.debug(f"confiance {confiance}")
    logger.debug(f"temps_prediction {temps_prediction}")    
    

    logger.debug(f"Model chargé - Lancement de la prédiction")
    image_path=os.path.join(init_paths["main_path"],init_paths["test_images"],"COVID-3.png")
    logger.debug(f"image_path {image_path}")
    prediction,confiance,temps_prediction=predict_and_log(model,model_info["selected_model_name"],image_path,prediction_logging_filepath)
    logger.debug(f"Predictions pour l'image {image_path}")
    logger.debug(f"prediction {prediction}")
    logger.debug(f"confiance {confiance}")
    logger.debug(f"temps_prediction {temps_prediction}")    
    

    logger.debug(f"Model chargé - Lancement de la prédiction")
    image_path=os.path.join(init_paths["main_path"],init_paths["test_images"],"Normal-4.png")
    logger.debug(f"image_path {image_path}")
    prediction,confiance,temps_prediction=predict_and_log(model,model_info["selected_model_name"],image_path,prediction_logging_filepath)
    logger.debug(f"Predictions pour l'image {image_path}")
    logger.debug(f"prediction {prediction}")
    logger.debug(f"confiance {confiance}")
    logger.debug(f"temps_prediction {temps_prediction}")    
    

    logger.debug(f"Model chargé - Lancement de la prédiction")
    image_path=os.path.join(init_paths["main_path"],init_paths["test_images"],"Normal-5.png")
    logger.debug(f"image_path {image_path}")
    prediction,confiance,temps_prediction=predict_and_log(model,model_info["selected_model_name"],image_path,prediction_logging_filepath)
    logger.debug(f"Predictions pour l'image {image_path}")
    logger.debug(f"prediction {prediction}")
    logger.debug(f"confiance {confiance}")
    logger.debug(f"temps_prediction {temps_prediction}")    
    

    logger.debug(f"Model chargé - Lancement de la prédiction")
    image_path=os.path.join(init_paths["main_path"],init_paths["test_images"],"Normal-6.png")
    logger.debug(f"image_path {image_path}")
    prediction,confiance,temps_prediction=predict_and_log(model,model_info["selected_model_name"],image_path,prediction_logging_filepath)
    logger.debug(f"Predictions pour l'image {image_path}")
    logger.debug(f"prediction {prediction}")
    logger.debug(f"confiance {confiance}")
    logger.debug(f"temps_prediction {temps_prediction}")    
    
if __name__ == "__main__":
    main()