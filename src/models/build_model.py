'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Construction du modèle et utilisation de kerastuner pour optimiser la sélection du modèle

'''
## Imports
# EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB0

## Imports related to the Neural Networks construction
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization,Flatten,Dropout,MaxPooling2D,Conv2D # Layers etc.
from tensorflow.keras.models import Model,Sequential # Models objects
from tensorflow.keras.optimizers import Adam,SGD # Optimizers
from tensorflow.keras.losses import BinaryCrossentropy # Loss function
from tensorflow.keras.regularizers import l2 # Regularizers 
from tensorflow.keras.callbacks import EarlyStopping,CSVLogger,LearningRateScheduler# Callbacks
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time

### Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths,infolog
from src.config.log_config import logger

######
## Model : EfficientNetB0
######
def build_model_efficientnetb0(hp, ml_hp,num_classes):
    """
    Constructs an EfficientNetB0 architecture model for image classification, compatible with binary and multi-class scenarios.
    Utilizes both Keras Tuner optimization hyperparameters and general configuration parameters from a configuration object.
    This configuration includes:
    - 'archi': defining the architecture to be built,
    - 'img_size': defining the image size for model input,
    - 'img_dim': color dimensions (RGB 3 or Grayscale 1),
    - etc.

    :param hp: HyperParameters object from Keras Tuner for model hyperparameter optimization.
    :param ml_hp: MLFlow HyperParameters object containing configuration details.
    :param actual_num_classes: Number of classes is dynamic therefore it is provided after the subfolders processing
    
    :return: A compiled Keras model optimized with hyperparameters.

    :raises KeyError: If essential keys are missing from the ml_hp dictionary.
    :raises ValueError: If configuration values in ml_hp are invalid or unsupported, such as an unsupported number of classes.
    """
    logger.debug("---------build_model_efficientnetb0------------")

    try:
        # 1a - Retrieving MLFlow hyperparameters
        img_size = ml_hp["img_size"]
        img_dim = ml_hp["img_dim"]
        #num_classes = ml_hp["num_classes"]

        # 1b - Optional parameters will have a default value
        hidden_layers_activation = ml_hp.get("hl_activation", "relu")

        # 1c - Initialization based on mlflow_archive hyperparameters
        shape = (img_size, img_size, img_dim)

        # 2 - Classification specifics: loss function
        if num_classes == 1:
            logger.debug("--- BINARY CLASSIFICATION ------")
            loss_function = 'binary_crossentropy'
            output_activation = "sigmoid"
        else: #0 refers to dynamic multiple classification , therefore any num_classes other than 1 will be non binary
            logger.debug("--- MULTICLASS CLASSIFICATION ------")
            loss_function = 'sparse_categorical_crossentropy'
            output_activation = "softmax"
        
        logger.debug("--- Hyperparameters ---")
    
        
        # 4 - Defining hyperparameters to be optimized with Keras Tuner
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])
        num_dropout_layers = hp.Int('num_dropout_layers', min_value=1, max_value=5)
        dropout_connect_rate = hp.Float('dropout_connect_rate', min_value=0.2, max_value=0.4, step=0.1)

        logger.debug("--- Architecture-specific details ---")
        logger.debug(f"dropout_connect_rate = {dropout_connect_rate}")

        # 5 - Loading the base model
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=shape)

        # 6 - Layer adjustments for fine-tuning
        # Fine-tuning strategy: Unfreezing convolutional layers while keeping batch normalization layers frozen
        for layer in base_model.layers:
            layer.trainable = isinstance(layer, Conv2D)

        # 7 - Adding Fully Connected Layers
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(units, activation=hidden_layers_activation, kernel_regularizer=l2(l2_lambda))(x)
        x = BatchNormalization()(x)

        # 8 - Adding Dropout Layers
        for _ in range(num_dropout_layers):
            x = Dropout(dropout_rate)(x)

        # 9 - Output Layer
        output = Dense(num_classes, activation=output_activation)(x)

        # 10 - Finalizing model construction
        model = Model(inputs=base_model.input, outputs=output)

        # 11 - Compiling the model with hyperparameters and classification-specific loss functions
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=['accuracy'])
        return model
    except KeyError as e:
        logger.error(f"KeyError: Missing key in ml_hp: {e}")
        raise KeyError(f"Missing key in ml_hp: {e}") from None
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise ValueError(e) from None