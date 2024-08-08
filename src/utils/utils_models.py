'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Fonctions utiles pour les modèles:
-- sauvegarde d'un modèle
-- chargement d'un modèle
-- sauvegarde/chargement des résultats d'un modèle (historique, training plots)

'''

### IMPORTS
import os
from tensorflow.keras.models import load_model
import pickle
import json
# Usual functions for calculation, dataframes and plots
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
import csv
# metrics
from tensorflow.keras.losses import CategoricalCrossentropy,SparseCategoricalCrossentropy,BinaryCrossentropy
from tensorflow.keras.layers import Conv2D
from keras.models import Model
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
import keras

### Internal imports
from src.datasets import image_preprocessing
### Configuration file import

### Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths,infolog
from src.config.log_config import logger

## Fonctions

# Sauvegarder les predictions
def save_prediction(model_name,image_path,prediction,confiance,temps_prediction,date_prediction,prediction_logging_filepath):
    ''' Mise à jour du fichier de prediction avec les informations recueillies'''
    logger.debug("-------save_prediction(model_name={model_name},image_path={img_path},prediction={prediction},confiance={confiance},temps_prediction={temps_prediction},date_prediction={date_prediction},prediction_logging_filepath={prediction_logging_filepath})----")
    # Créer un dictionnaire avec les informations
    data = {
        'Nom du modèle': model_name,
        'Chemin de l\'image': image_path,
        'Prédiction': prediction,
        'Indice de confiance': confiance,
        'Temps de prédiction':temps_prediction,
        'Date de prédiction':date_prediction
    }
    file_exists = os.path.isfile(prediction_logging_filepath)
    with open(prediction_logging_filepath, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        # Si le fichier n'existe pas, on ajoute l'entête
        if not file_exists:
            logger.debug(f"Fichier de logging n'existe pas")
            logger.debug(f"Ecriture de l'entête {data.keys()}")
            writer.writerow(data.keys())
        else:
            logger.debug("Le fichier de logging existe, stockage des résultats en fin de fichier")
            logger.debug(f"Ecriture des résultats {data.values()}")
        # Write the new data to the file
        writer.writerow(data.values())
         # Write the data rows
        '''
        for row in data:
            logger.debug(f"")
            writer.writerow(row)
        '''

### FUNCTIONS
## Keras Models save and load
def save_model(model, save_path):
    """
    Saves a Keras model to the specified path.
    
    :param model: The model to save.
    :param save_path: The file FULL path where to save the model.
    :raises PermissionError: If there has been a write permision issue
    :raises IOError: if there has been an I/O error
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------save_model--------------------")
    try: # Saving the model
        # Extract the directory from the complete file path
        directory = os.path.dirname(save_path)
        
        # Check if the directory already exists
        if not os.path.exists(directory):
            # Create the directory if it does not exist
            os.makedirs(directory)
        
        # Save the model to the specified path
        model.save(save_path)
        
    except PermissionError:
        # Exception if there is no permission to write in the directory
        logger.error("Error: No permission to write to the specified directory.")
        raise
    except IOError as e:
        # Handle general I/O errors
        logger.error(f"An I/O error occurred: {e}")
        raise
    except Exception as e:
        # Handle other possible exceptions
        logger.error(f"An unexpected error occurred: {e}")
        raise

    return save_path

def save_weights(model, save_path):
    """
    Saves a Keras model to the specified path.
    
    :param model: The model to save.
    :param save_path: The file FULL path where to save the model.
    :raises PermissionError: If there has been a write permision issue
    :raises IOError: if there has been an I/O error
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------save_model--------------------")
    try: # Saving the model
        # Extract the directory from the complete file path
        directory = os.path.dirname(save_path)
        
        # Check if the directory already exists
        if not os.path.exists(directory):
            # Create the directory if it does not exist
            os.makedirs(directory)
        
        # Save the model to the specified path
        model.save_weights(save_path)
        
    except PermissionError:
        # Exception if there is no permission to write in the directory
        logger.error("Error: No permission to write to the specified directory.")
        raise
    except IOError as e:
        # Handle general I/O errors
        logger.error(f"An I/O error occurred: {e}")
        raise
    except Exception as e:
        # Handle other possible exceptions
        logger.error(f"An unexpected error occurred: {e}")
        raise

    return save_path


## Models loading. Names load_models to distinguish the function from the keras 'load_model' function
def load_models(load_path):
    """
    Loads a Keras model from a file.
    uses tensorflow.keras.models.load_model
    :param load_path: The FULL file path containing the model to load.
    :return: The loaded model.
    :raises IOError: if the file to load is not accessible
    :raises ValueError: If there has been an issue with the model interpretation (not a model or corrupted file)
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------load_models--------------------")
    try:
        # Load the model from the specified file path
        model = load_model(load_path)
        return model
    except IOError as e:
        # Handle errors related to file access issues
        logger.error(f"Error: Could not access file to load model. {e}")
        raise
    except ValueError as e:
        # Handle errors related to the model file being invalid or corrupted
        logger.error(f"Error: The file might not be a Keras model file or it is corrupted. {e}")
        raise
    except Exception as e:
        # Handle other possible exceptions
        logger.error(f"An unexpected error occurred while loading the model: {e}")
        raise

## Models training history save and load (pickle or json format)
def save_history(history, save_path):
    """
    Saves a training history to a file, supporting both Pickle, JSON and CSV formats.
    
    uses pickle.dump or json.dump depending on the specified file path extension
    :param history: The training history to save.
    :param save_path: The FULL file path where the training history should be saved.
    
    :raises If there has been an issue with the model interpretation (not a model or corrupted file)
    :raises PermissionError: If the write permission is not granted to the specified path
    :raises IOError: if there is an I/O issue
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------save_history--------------------")
    
    try:
        # Ensure the directory exists
        logger.debug(f"Saving file in {save_path}")
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Determine the file format from the extension and save accordingly
        _, ext = os.path.splitext(save_path)
        if ext == '.pkl':
            with open(save_path, 'wb') as f:
                pickle.dump(history, f)
        elif ext == '.json':
            with open(save_path, 'w') as f:
                json.dump(history, f)
        elif ext =='.csv':
            # We need to ensure that the history object is a dataframe, if not, saving might be corrupted and therefore the save request is rejected
            if isinstance(history, pd.DataFrame):
                history.to_csv(save_path, index=False)
            else:
                raise ValueError("Unsupported object format. Object must be a dataframe to be stored in a csv format")
        else:
            raise ValueError("Unsupported file format. Please use .pkl or .json.")
    except PermissionError:
        logger.error(f"Error: No permission to write to {save_path}.")
        raise
    except IOError as e:
        logger.error(f"An I/O error occurred while saving to {save_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    return save_path

## Models training history load (pickle, json or csv format)
def load_history(load_path):
    """
    Loads a training history from a file, supporting both pickle and JSON formats.
    uses pickle.load or json.load depending on the specified file path extension
    :param load_path: The file path of the training history to load.
    :return: The loaded training history.
    :raises ValueError: If the file format is not suppoeted
    :raises IOError: if there is an I/O issue
    :raises pickle.PickleError: if the pickle file might be corrupted
    :raises json.JSONDecodeError: if the JSON file might be corrupted
    :raises pd.errors.ParserError: if the CSV file to be stored in a dataframe might be corrupted
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------load_history--------------------")
    try:
        # Determine the file format from the extension
        _, file_extension = os.path.splitext(load_path)
        with open(load_path, 'rb' if file_extension.lower() == '.pkl' else 'r') as f:
            if file_extension.lower() == '.pkl':
                history = pickle.load(f)
            elif file_extension.lower() == '.json':
                history = json.load(f)
            elif file_extension.lower() == '.csv':
                history = pd.read_csv(f)
            else:
                raise ValueError("Unsupported file format. Please use .pkl, .json or .csv .")
        return history
    except IOError as e:
        logger.error(f"Error: Could not access file at {load_path}. {e}")
        raise
    except (pickle.PickleError, json.JSONDecodeError, pd.errors.ParserError) as e:
        logger.error(f"Error: The file might be corrupted or in an incorrect format. {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

## Saving Training visual reports (history) Pickle, JSON or CSV file
def generate_training_plots(history_file, output_filepath, run_info):
    """
    Generates and saves training and validation loss and accuracy plots from a history file.
    The history file can be in PKL, CSV, or JSON format.

    :param history_file: The FULL filepath of the training history file.
    :param output_folder: The output directory to save the plots.
    :param run_info: (currently run id) Information or identifier for the training run to be included in the plot titles.
    the file name format is "{run_info}_training_validation_plots.png"
    :raises Exception: If an error occurs
    """
    logger.debug("--------------------generate_training_plots--------------------")
    # Determine file extension and load history accordingly
    logger.debug(f"history_file {history_file}")
    logger.debug(f"output_folder {output_filepath}")
    logger.debug(f"run_id {run_info}")
    _, file_extension = os.path.splitext(history_file)
    
    # first step is to load the history file
    try:
        if file_extension.lower() == '.pkl':
            # Load history from PKL file
            logger.debug("Managing Pickle file")
            history = pd.read_pickle(history_file)
        elif file_extension.lower() == '.csv':
            # Load history from CSV file
            logger.debug("Managing CSV file")
            history = pd.read_csv(history_file)
        elif file_extension.lower() == '.json':
            # Load history from JSON file
            logger.debug("Managing JSON file")
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please use .pkl, .csv, or .json")

        # Create plot
        plt.figure(figsize=(12, 4))

        # Loss subplot
        plt.subplot(121)
        plt.plot(history['loss'], label='train')
        plt.plot(history['val_loss'], label='test')
        plt.title(f'Run {run_info} - Model loss by epoch')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')

        # Accuracy subplot
        plt.subplot(122)
        plt.plot(history['accuracy'], label='train')
        plt.plot(history['val_accuracy'], label='test')
        plt.title(f'Run {run_info} - Model accuracy by epoch')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')

        # Save the plot
        plt.savefig(output_filepath)
        plt.close()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

## Save confusion matrix and classification report from dataframes
def save_dataframe_plot(df, output_filepath, plot_type, labels_dict=None):
    """
    Generates and saves a plot based on the specified type, appending the plot type as a suffix to the run ID for the filename.

    :param df: DataFrame to be plotted, can be a confusion matrix or a classification report.
    :param output_folder: The output directory to save the plot.
    :param run_id: The identifier for the run or experiment. Used as the base name for the output file.
    :param plot_type: Type of the plot to generate - 'confusion_matrix' or 'classification_report'. Also used as a suffix for the filename.
    :param labels_dict: Optional; needed if plot_type is 'confusion_matrix'. It maps class numbers to class names.
    """
    logger.debug("--------------------save_dataframe_plot--------------------")
    # Ensure plot_type is valid and prepare the plot accordingly
    if plot_type == 'confusion_matrix' and labels_dict is not None:
        # Generate labels for confusion matrix
        axis_labels = [labels_dict[i] for i in labels_dict]

        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=axis_labels, yticklabels=axis_labels)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        
    elif plot_type == 'classification_report':
        plt.figure(figsize=(10, len(df) * 0.5))
        sns.heatmap(data=df, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.title('Classification Report')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
    else:
        raise ValueError("Invalid plot type specified. Use 'confusion_matrix' or 'classification_report'.")

    
    # Save the plot
    plt.savefig(output_filepath)
    plt.close()

    logger.debug(f"Plot saved to {output_filepath}")

