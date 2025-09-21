# Módulo: Construção do Modelo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras.applications import (MobileNetV2, 
                                MobileNetV3Large, 
                                MobileNetV3Small, 
                                EfficientNetB0,
                                EfficientNetB1,
                                EfficientNetB2,
                                EfficientNetB3,
                                EfficientNetB4,
                                EfficientNetV2B0,
                                EfficientNetV2B1,
                                EfficientNetV2B2
                                )
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy, CategoricalFocalCrossentropy



# Carrega o modelo MobileNetV2_v1 removendo as camadas densas
def build_model_mobilenetV2_v1(formato_imagem=(224,224,3), num_classes=41):
  '''
    Modelo base: MobileNetV2
    Camadas convolucionais: Todas congeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

  '''

  # Carrega o modelo MobileNetV2 removendo as camadas densas
  base_model = MobileNetV2(weights='imagenet',
                           include_top=False,
                           input_shape=formato_imagem)

  # Mantém todas as camadas convolucionais congeladas
  for layer in base_model.layers:
    layer.trainable = False
  
  # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(512, activation="relu")(x)
  x = Dropout(0.5)(x)
  x = Dense(256, activation="relu")(x)
  x = Dropout(0.4)(x)

  # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
  classifier = Dense(num_classes, activation="softmax")(x)

  model_mobilenetV2_v1 = Model(inputs=base_model.inputs, outputs=classifier)

  return model_mobilenetV2_v1


# Carrega o modelo MobileNetV2_v2 removendo as camadas densas
def build_model_mobilenetV2_v2(formato_imagem=(224,224,3), num_classes=41):
  '''
    Modelo base: MobileNetV2
    Camadas convolucionais: 20 últimas descongeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

  '''

  # Carrega o modelo MobileNetV2 removendo as camadas densas
  base_model = MobileNetV2(weights='imagenet',
                           include_top=False,
                           input_shape=formato_imagem)

  # Mantém todas as camadas convolucionais congeladas
  for layer in base_model.layers[-20:]:
    layer.trainable = True
  
  # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(512, activation="relu")(x)
  x = Dropout(0.5)(x)
  x = Dense(256, activation="relu")(x)
  x = Dropout(0.4)(x)

  # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
  classifier = Dense(num_classes, activation="softmax")(x)

  model_mobilenetV2_v2 = Model(inputs=base_model.inputs, outputs=classifier)

  return model_mobilenetV2_v2


def build_model_mobilenetv3_large_v1(formato_imagem=(224,224,3), num_classes=41):
  '''
    Modelo base: MobileNetV3 Large
    Camadas convolucionais: Todas congeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

  '''

  # Carrega o modelo MobileNetV3 Large removendo as camadas densas
  base_model = MobileNetV3Large(weights='imagenet',
                           include_top=False,
                           input_shape=formato_imagem)

  # Mantém todas as camadas convolucionais congeladas
  for layer in base_model.layers:
    layer.trainable = False
  
  # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(512, activation="relu")(x)
  x = Dropout(0.5)(x)
  x = Dense(256, activation="relu")(x)
  x = Dropout(0.4)(x)

  # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
  classifier = Dense(num_classes, activation="softmax")(x)

  model_mobilenetv3_large_v1 = Model(inputs=base_model.inputs, outputs=classifier)

  return model_mobilenetv3_large_v1



def build_model_mobilenetv3_large_v2(formato_imagem=(224,224,3), num_classes=41):
  '''
    Modelo base: MobileNetV3 Large
    Camadas convolucionais: 20 últimas descongeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

  '''

  # Carrega o modelo MobileNetV3 Large removendo as camadas densas
  base_model = MobileNetV3Large(weights='imagenet',
                           include_top=False,
                           input_shape=formato_imagem)

  # Mantém todas as camadas convolucionais congeladas
  for layer in base_model.layers[-20:]:
    layer.trainable = True
  
  # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(512, activation="relu")(x)
  x = Dropout(0.5)(x)
  x = Dense(256, activation="relu")(x)
  x = Dropout(0.4)(x)

  # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
  classifier = Dense(num_classes, activation="softmax")(x)

  model_mobilenetv3_large_v2 = Model(inputs=base_model.inputs, outputs=classifier)

  return model_mobilenetv3_large_v2



def build_model_mobilenetv3_small_v1(formato_imagem=(224,224,3), num_classes=41):
  '''
    Modelo base: MobileNetV3 Small
    Camadas convolucionais: Todas as camadas convolucionais congeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

  '''

  # Carrega o modelo MobileNetV3 Large removendo as camadas densas
  base_model = MobileNetV3Small(weights='imagenet',
                           include_top=False,
                           input_shape=formato_imagem)

  # Mantém todas as camadas convolucionais congeladas
  for layer in base_model.layers:
    layer.trainable = False
  
  # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(512, activation="relu")(x)
  x = Dropout(0.5)(x)
  x = Dense(256, activation="relu")(x)
  x = Dropout(0.4)(x)

  # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
  classifier = Dense(num_classes, activation="softmax")(x)

  model_mobilenetv3_small_v1 = Model(inputs=base_model.inputs, outputs=classifier)

  return model_mobilenetv3_small_v1



def build_model_mobilenetv3_small_v2(formato_imagem=(224,224,3), num_classes=41):
  '''
    Modelo base: MobileNetV3 Small
    Camadas convolucionais: 20 últimas descongeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

  '''

  # Carrega o modelo MobileNetV3 Large removendo as camadas densas
  base_model = MobileNetV3Small(weights='imagenet',
                           include_top=False,
                           input_shape=formato_imagem)

  # Descongela as 20 últimas camadas convolucionais
  for layer in base_model.layers[-20:]:
    layer.trainable = True
  
  # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(512, activation="relu")(x)
  x = Dropout(0.5)(x)
  x = Dense(256, activation="relu")(x)
  x = Dropout(0.4)(x)

  # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
  classifier = Dense(num_classes, activation="softmax")(x)

  model_mobilenetv3_small_v2 = Model(inputs=base_model.inputs, outputs=classifier)

  return model_mobilenetv3_small_v2



###  MODELO MOBILENETV3 SMALL COM PARÂMETRO PARA DESCOLGELAMENTO DE CAMADAS CONVOLUCIONAIS
def build_model_mobilenetv3_small_v3(formato_imagem=(224,224,3), num_classes=41, trainable_layers=20):
  '''
    Modelo base: MobileNetV3 Small
    Camadas convolucionais: Número variável de camadas descongeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

  '''
  # Carrega o modelo MobileNetV3 Large removendo as camadas densas
  base_model = MobileNetV3Small(weights='imagenet',
                           include_top=False,
                           input_shape=formato_imagem)

  # Descongela as últimas 'trainable_layers' camadas convolucionais
  for layer in base_model.layers[-trainable_layers:]:
    layer.trainable = True
  
  # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(512, activation="relu")(x)
  x = Dropout(0.5)(x)
  x = Dense(256, activation="relu")(x)
  x = Dropout(0.4)(x)

  # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
  classifier = Dense(num_classes, activation="softmax")(x)

  model_mobilenetv3_small_v3 = Model(inputs=base_model.inputs, outputs=classifier)

  return model_mobilenetv3_small_v3




# -----------------------------------------------------------------------------
#      Família EfficientNet
# -----------------------------------------------------------------------------
# EfficientNet B0
def build_model_efficientnet_b0(formato_imagem=(224,224,3), num_classes=41):
    '''
    Modelo base: EfficientNet B0
    Camadas convolucionais: Todas as camadas convolucionais congeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

    '''

    # Carrega o modelo EfficientNet B0 removendo as camadas densas
    base_model = EfficientNetB0(weights='imagenet',
                                 include_top=False,
                                 input_shape=formato_imagem)

    # Mantém todas as camadas convolucionais congeladas
    for layer in base_model.layers:
        layer.trainable = False

    # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)

    # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
    classifier = Dense(num_classes, activation="softmax")(x)

    model_efficientnet_b0 = Model(inputs=base_model.inputs, outputs=classifier)

    return model_efficientnet_b0

  
# EfficientNet B1
def build_model_efficientnet_b1(formato_imagem=(240,240,3), num_classes=41):
    '''
    Modelo base: EfficientNet B1
    Camadas convolucionais: Todas as camadas convolucionais congeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

    '''

    # Carrega o modelo EfficientNet B1 removendo as camadas densas
    base_model = EfficientNetB1(weights='imagenet',
                                 include_top=False,
                                 input_shape=formato_imagem)

    # Mantém todas as camadas convolucionais congeladas
    for layer in base_model.layers:
        layer.trainable = False

    # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)

    # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
    classifier = Dense(num_classes, activation="softmax")(x)

    model_efficientnet_b1 = Model(inputs=base_model.inputs, outputs=classifier)

    return model_efficientnet_b1
  


# EfficientNet B2
def build_model_efficientnet_b2(formato_imagem=(260,260,3), num_classes=41):
    '''
    Modelo base: EfficientNet B2
    Camadas convolucionais: Todas as camadas convolucionais congeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

    '''

    # Carrega o modelo EfficientNet B2 removendo as camadas densas
    base_model = EfficientNetB2(weights='imagenet',
                                 include_top=False,
                                 input_shape=formato_imagem)

    # Mantém todas as camadas convolucionais congeladas
    for layer in base_model.layers:
        layer.trainable = False

    # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)

    # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
    classifier = Dense(num_classes, activation="softmax")(x)

    model_efficientnet_b2 = Model(inputs=base_model.inputs, outputs=classifier)

    return model_efficientnet_b2
  
  
  # EfficientNetV2 B0
def build_model_efficientnetv2_b0(formato_imagem=(224,224,3), num_classes=41):
    '''
    Modelo base: EfficientNetV2 B0
    Camadas convolucionais: Todas as camadas convolucionais congeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

    '''

    # Carrega o modelo EfficientNetV2 B0 removendo as camadas densas
    base_model = EfficientNetV2B0(weights='imagenet',
                                 include_top=False,
                                 input_shape=formato_imagem)

    # Mantém todas as camadas convolucionais congeladas
    for layer in base_model.layers:
        layer.trainable = False

    # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)

    # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
    classifier = Dense(num_classes, activation="softmax")(x)

    model_efficientnetv2_b0 = Model(inputs=base_model.inputs, outputs=classifier)

    return model_efficientnetv2_b0
  
  
  # EfficientNetV2 B1
def build_model_efficientnetv2_b1(formato_imagem=(240,240,3), num_classes=41):
    '''
    Modelo base: EfficientNetV2 B1
    Camadas convolucionais: Todas as camadas convolucionais congeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

    '''
    # Carrega o modelo EfficientNetV2 B1 removendo as camadas densas
    base_model = EfficientNetV2B1(weights='imagenet',
                                 include_top=False,
                                 input_shape=formato_imagem)

    # Mantém todas as camadas convolucionais congeladas
    for layer in base_model.layers:
        layer.trainable = False

    # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)

    # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
    classifier = Dense(num_classes, activation="softmax")(x)

    model_efficientnetv2_b1 = Model(inputs=base_model.inputs, outputs=classifier)

    return model_efficientnetv2_b1
  
  
  # EfficientNetV2 B2
def build_model_efficientnetv2_b2(formato_imagem=(260,260,3), num_classes=41):
    '''
    Modelo base: EfficientNetV2 B2
    Camadas convolucionais: Todas as camadas convolucionais congeladas
    Classificador: 
              GlobalAveragePooling2D()
              Dense(512, activation="relu")
              Dropout(0.5)
              Dense(256, activation="relu")
              Dropout(0.4)
              Dense(41, activation="softmax")

    '''
    # Carrega o modelo EfficientNetV2 B2 removendo as camadas densas
    base_model = EfficientNetV2B2(weights='imagenet',
                                 include_top=False,
                                 input_shape=formato_imagem)

    # Mantém todas as camadas convolucionais congeladas
    for layer in base_model.layers:
        layer.trainable = False

    # Criação de duas camadas densas com 512 e 254 neurônios respectivamente
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)

    # Camada de saída com 41 neurônios, isto é, um para cada espécie de madeira
    classifier = Dense(num_classes, activation="softmax")(x)

    model_efficientnetv2_b2 = Model(inputs=base_model.inputs, outputs=classifier)

    return model_efficientnetv2_b2  
  
  
  
  
  
# -----------------------------------------------------------------------------
#      Compilação dos Modelos
# -----------------------------------------------------------------------------

def compile_model_efficientnet(modelo, lr_rate=0.0001):
    '''
    Compila o modelo com otimizador Adam e categorical crossentropy.

    Args:
        model (tf.keras.Model): Modelo a ser compilado.
        learning_rate (float): Taxa de aprendizado do otimizador.

    Returns:
        model (tf.keras.Model): Modelo compilado.
    '''
    optimizer = Adam(learning_rate=lr_rate)

    modelo.compile(
        optimizer=optimizer,
        loss=CategoricalFocalCrossentropy(),
        metrics=['accuracy']
    )
    return modelo


def compile_model_mobilenetv3(modelo, lr_rate=0.0001):
    '''
    Compila o modelo com otimizador Adam e categorical crossentropy.

    Args:
        model (tf.keras.Model): Modelo a ser compilado.
        learning_rate (float): Taxa de aprendizado do otimizador.

    Returns:
        model (tf.keras.Model): Modelo compilado.
    '''
    optimizer = Adam(learning_rate=lr_rate)

    modelo.compile(
        optimizer=optimizer,
        loss=CategoricalFocalCrossentropy(),
        metrics=['accuracy']
    )
    return modelo


def compile_model_mobilenetv2(modelo, lr_rate=0.0001):
    '''
    Compila o modelo com otimizador Adam e categorical crossentropy.

    Args:
        model (tf.keras.Model): Modelo a ser compilado.
        learning_rate (float): Taxa de aprendizado do otimizador.

    Returns:
        model (tf.keras.Model): Modelo compilado.
    '''
    optimizer = Adam(learning_rate=lr_rate)

    modelo.compile(
        optimizer=optimizer,
        loss=CategoricalFocalCrossentropy(),
        metrics=['accuracy']
    )
    return modelo

