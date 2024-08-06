'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Entrainement (ou réentrainement) du modèle sélectionné

'''

#####
# Imports 
#####
import os
from sklearn.model_selection import train_test_split
import time
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping,CSVLogger,LearningRateScheduler # Callbacks

from src.config.log_config import logger
from src.config.run_config import init_paths,infolog

### FUNCTIONS

def train_model(model, ml_hp, X, y, full_run_folder, model_training_history_csv):
    """
    Trains the model using provided data, MLFlow hyperparameters, and run ID for file naming (CSVLogger).
    
    :param model: The model to be trained.
    :param ml_hp: MLFlow hyperparameters.
    :param X: Training data features.
    :param y: Training data labels.
    :param run_id: Used for naming callback files (CSVLogger).
    
    :return: A tuple containing the trained model, basic metrics (other metrics to be calculated in MLFlow), training history, and execution time.
    
    :raises KeyError: If 'max_epochs' is missing from ml_hp.
    :raises ValueError: If data splitting results in empty training or validation sets.
    """
    logger.debug("--------- train_model ---------")
    try:
        # 1 - Retrieving MLFlow hyperparameters for model training
        max_epochs = ml_hp.get('max_epochs')
        if max_epochs is None:
            raise KeyError("'max_epochs' must be specified in ml_hp.")

        # 2 - Splitting data for training
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1234)
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Data splitting resulted in empty training or validation sets.")

        logger.debug("Data Split Info")
        logger.debug(f"Size X {len(X)}")
        logger.debug(f"Size y {len(y)}")
        logger.debug(f"Train X size {len(X_train)}")
        logger.debug(f"Train y size {len(y_train)}")
        logger.debug(f"Validation X size {len(X_val)}")
        logger.debug(f"Validation y size {len(y_val)}")
        
        # Callbacks: Early Stopping, CSV Logger, LearningRateScheduler with patience of 10 epochs
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        
        # CSV Logger
        #csv_logger_filename=f"model_training_logs_{run_id}_{ml_hp['archi']}.csv"
        logger.debug(f" Model Training CSV logger file name  {model_training_history_csv}")
        full_csv_logger_path=os.path.join(full_run_folder,model_training_history_csv)
        csv_logger = CSVLogger(full_csv_logger_path, append=True, separator=';')
        logger.debug(f" Model Training history file name stored in  (/!\ may be duplicate with the json file) {full_csv_logger_path}")
        
        
        def scheduler(epoch, lr):
            return lr * np.exp(-0.1) if epoch >= 10 else lr
        lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

        # Timing model training
        start_time = time.time()
        history = model.fit(X_train, y_train, epochs=max_epochs, validation_data=(X_val, y_val), callbacks=[early_stopping, lr_scheduler, csv_logger])
        end_time = time.time()
        
        execution_time = round(end_time - start_time, 2) / 60
        logger.debug(f"Model training time {execution_time} min")

        # Gathering basic metrics
        metrics = {
            'accuracy': max(history.history['accuracy']),
            'val_accuracy': max(history.history['val_accuracy']),
            'loss': max(history.history['loss']),
            'val_loss': max(history.history['val_loss']),
        }
        
        logger.debug(f"Execution time {execution_time}")
        logger.debug(f"history {history.history}")
        return model, metrics, history.history

    except KeyError as e:
        logger.error(f"KeyError: {e}")
        raise
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise