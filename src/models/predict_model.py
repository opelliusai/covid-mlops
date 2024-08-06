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
from src.utils import utils_models as um
### Configuration file import
'''
src.config file contains information of the repository
paths[]: Folders and subfolders to find or generate information ex : paths['main'] is the project path
infolog[]: Logging information : Folder, logname ex : utils_models.log
'''
from src.config.log_config import logger

# Model evaluation function
def evaluate_model(model, X_eval, y_eval, num_classes, classes):
    """
    Evaluates the model and returns metrics, confusion matrix, and classification report.
    
    :param model: Model to be evaluated.
    :param X_eval: Evaluation images.
    :param y_eval: Evaluation targets.
    :param num_classes: Number of classes, defining binary or multiclass classification.
    :param classes: Dictionary for class mapping.
    
    :return: A tuple containing metrics dictionary, confusion matrix dataframe, and classification report dataframe.
             Confusion matrix and Classification report will be saved as files by MLFlow.
             Metrics will be stored in the RUN.

    :raises Exception: If an error occurs during the model evaluation process.
    """

    try:
        # 1- Retrieving additional metrics. 
        # Specific functions have been created for binary and multiple classification because of the high divergence of the metrics generation and management
        metrics_dict = um.bin_get_prediction_metrics(model, X_eval, y_eval) if num_classes == 1 else um.multi_get_prediction_metrics(model, X_eval, y_eval)
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
        conf_matrix_df = pd.DataFrame(metrics_dict["Confusion Matrix"], index=[i for i in classes], columns=[i for i in classes])
        # Classification report as dataframe
        class_report_df = pd.DataFrame(metrics_dict["Classification report"]).transpose()
        
        return final_metrics, conf_matrix_df, class_report_df

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise Exception(f"Error during model evaluation: {e}")
