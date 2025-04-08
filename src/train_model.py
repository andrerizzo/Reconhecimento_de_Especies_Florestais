# Módulo: Treinamento do Modelo

import os

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def train_model(model, 
                train_images, 
                val_images, 
                output_dir,
                model_file_name='model.keras', 
                epochs=20, 
                patience=5, 
                factorROP = 0.1, 
                patienceROP=5,
                min_lr_ROP=0.0001):

    """
    Parâmetros:
    ------------
    - model (tf.keras.Model): Modelo já compilado.
    - train_images (tf.data.Dataset): Dataset de treino.
    - val_images (tf.data.Dataset): Dataset de validação.
    - output_dir (str): Caminho para salvar o modelo treinado. Default: "outputs/modelos_salvos"
    - epochs (int): Número máximo de épocas.
    - patience (int): Número de épocas sem melhora para acionar o EarlyStopping.
    - factorROP (float): Fator de redução de learning rate no ReduceLROnPlateau.
    - patienceROP (int): Número de épocas sem melhora no val_loss antes de reduzir learning rate.
    - min_lr_ROP (float): Valor mínimo da learning rate ao usar ReduceLROnPlateau.

    Retorno:
    ---------
    - history (keras.callbacks.History): Histórico de métricas de treino e validação por época.

    Exemplo de uso:
    ----------------
    from train_model import train_model

    # Suponha que model, train_ds e val_ds já estejam definidos
    history = train_model(
        model=model,
        train_images=train_ds,
        val_images=val_ds,
        epochs=25,
        patience=5,
        factorROP=0.2,
        patienceROP=4,
        min_lr_ROP=1e-5
    )
    """

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_file_name)

    # Callbacks
    checkpoint = ModelCheckpoint(model_path, 
                                 monitor='val_accuracy', 
                                 save_best_only=True, 
                                 verbose=1)

    early_stop = EarlyStopping(monitor='val_loss', 
                               patience=patience, 
                               restore_best_weights=True, 
                               verbose=1)
    
    reduce_on_plateau =  ReduceLROnPlateau(monitor='val_loss', 
                                           factor=factorROP, 
                                           patience=patienceROP, 
                                           min_lr=min_lr_ROP)

 
    # Treinamento
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=epochs,
        callbacks=[checkpoint, 
                   early_stop,
                   reduce_on_plateau]
    )

    return history
