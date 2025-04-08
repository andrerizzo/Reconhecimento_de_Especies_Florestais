# Módulo: Aquisição de Dados

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings

import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory
from keras.utils import to_categorical
from keras.applications import MobileNetV2, MobileNetV3Large, MobileNetV3Small
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy, CategoricalFocalCrossentropy


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from collections import Counter

#from google.colab import drive, output

from tqdm import tqdm

from IPython.display import clear_output
import os
import glob
import gdown
import shutil

import zipfile
import tarfile
import glob
import os
import time
import string
import sys

import cv2


#warnings.filterwarnings('ignore')
#drive.mount('/content/drive')

'''
Função 1: upload_patches()

    Ações:
        Faz upload dos patches do Google Drive para
        máquina local ou Google Colab



Função 2: uncompress_patches()
    
    Ações:
        Descompacta patches
        


Função 3: gera_lista()
    
    Ações:
        Cria uma lista de todos os patches existentes
        Atribui uma classe a cada um dos patches

        
Função 4: train_test_val()

    Ações:
        Gera listas de patches que serão usados para treinamento.
        teste e validação do modelo


Função 5: folder_creation()

    Ações:
        Cria diretórios: train / test / val
        Cria uma estrutura de pastas abaixo dos diretórios train / test /val
        
        train/
        |
        |--- class_01/
        |--- class_02/
        |  ...
        |--- class_41/

        test/
        |
        |--- class_01/
        |--- class_02/
        |  ...
        |--- class_41/

        val/
        |
        |--- class_01/
        |--- class_02/
        |  ...
        |--- class_41/


Função 6: create_symlink()

Função 7: symlink()

Função 8: download_patches_file(id_google_drive_file, path_destino, arquivo_destino)

Função 9: uncompress_patches(arquivo_zip, path_destino)


Ordem de execução:
    gera_lista
    train_test_val
    train_test_val
    folder_creation
    symlink 


'''



def gera_lista(image_path):
    '''
        Gera uma lista de arquivos e classes a partir
        das imagens existentes.

        Parâmetros:
            image_path: diretório dos patches (string)

        Atributos:
            temp_ds: lista de imagens e suas classes (dataframe)

    '''
    image_set = []

    for _, _, imagem in os.walk(image_path):
        for img in imagem:
            arquivo = img.split("_")[0]
            classe = arquivo[:2]
            image_set.append([classe, arquivo])

    temp_ds = pd.DataFrame(image_set, columns=['classe', 'imagem'])
    temp_ds = temp_ds.drop_duplicates(subset='imagem', keep='first')
    temp_ds = temp_ds.reset_index(drop=True)

    return temp_ds


def train_test_val(image_ds, tam_teste, estratifica):
    '''
        Gera listas de imagens que serão usadas para treinamento.
        teste e validação do modelo

        Parâmetros:
            image_ds: lista de todas as imagens (dataframe)

            tam_teste: percentual dos dados para teste / validação (float)

            estratifica: informar os labels a serem usados na estratificação (string)

        Retorna:
            X_train: imagens a serem usadas para treinamento (lista)

            X_test: imagens a serem usadas para teste / validação (lista)

    '''
    X_train, X_test = train_test_split(image_ds,
                                        test_size=tam_teste,
                                        random_state=44,
                                        stratify=estratifica,
                                        shuffle=True)
    return X_train, X_test




def folder_creation(root_patches_folder):
    '''
    Criação de diretórios para armazenamento dos links 
    para os patches

    Parâmetro:
        root_patches_folder (str): diretório base dos patches
    
    Retorna:
        train_path, test_path, validation_path: diretórios de
        treino, teste e validação onde os links para os patches
        serão armazenados

    '''

    # Define os paths dos datasets de treino, teste e validação
    train_path = os.path.join(root_patches_folder,'train')
    test_path = os.path.join(root_patches_folder,'test')
    validation_path = os.path.join(root_patches_folder,'val')

    # Cria a estrutura de diretório de treinamento, se não existir.
    # Se existir, apaga e cria nova
    if not os.path.exists(train_path):
        for classe in range(1,42):

            os.makedirs(os.path.join(train_path, f'class_{classe:02d}'), exist_ok=True)
        print('Diretórios de treinamento criados com sucesso ! \n')
    else:
        shutil.rmtree(train_path)
        for classe in range(1,42):
            os.makedirs(os.path.join(train_path, f'class_{classe:02d}'), exist_ok=True)
        print('Diretórios de treinamento criados com sucesso ! \n')


    # Cria a estrutura de diretório de validação, se não existir
    if not os.path.exists(validation_path):
        for classe in range(1,42):
            os.makedirs(os.path.join(validation_path, f'class_{classe:02d}'), exist_ok=True)
        print('Diretórios de validação criados com sucesso ! \n')
    else:
        shutil.rmtree(validation_path)
        for classe in range(1,42):
            os.makedirs(os.path.join(validation_path, f'class_{classe:02d}'), exist_ok=True)
        print('Diretórios de validação criados com sucesso ! \n')

    # Cria a estrutura de diretório de teste, se não existir
    if not os.path.exists(test_path):
        for classe in range(1,42):
            os.makedirs(os.path.join(test_path, f'class_{classe:02d}'), exist_ok=True)
        print('Diretórios de teste criados com sucesso ! \n')
    else:
        shutil.rmtree(test_path)
        for classe in range(1,42):
            os.makedirs(os.path.join(test_path, f'class_{classe:02d}'), exist_ok=True)
        print('Diretórios de teste criados com sucesso ! \n')

    return train_path, test_path, validation_path




# ------------------------------------------------------------------
#            Cria links para os patches
# ------------------------------------------------------------------

def create_symlink(src, dest):
    """
        Cria links para os patches
    """

    if os.path.exists(src):
        os.symlink(src, dest)
    else:
        print(f'Arquivo não encontrado: {src}')


def symlink(image_list, path_origem, path_destino):

    for img in tqdm(image_list, desc="Creating symlinks"):
        class_num = int(img[:2])  # Extract class number
        for patch in range(0, 140):
            src = os.path.join(path_origem, f'class_{class_num}', f'{img}_patch_{patch}.jpg')
            dest = os.path.join(path_destino, f'class_{class_num:02d}', f'{img}_patch_{patch}.jpg')
            create_symlink(src, dest)



def download_patches_file(id_google_drive_file, path_destino, arquivo_destino):
  '''
  Realiza download dos arquivos .zip com os patches
  '''
  target_file = os.path.join(path_destino, arquivo_destino)
  if not os.path.exists(target_file):
    gdown.download(id=id_google_drive_file, 
                   output=target_file, 
                   quiet=False)
  return target_file


def uncompress_patches(arquivo_zip, path_destino):
  '''
  Descompacta arquivo com os patches
  '''
  if os.path.exists(arquivo_zip):
    print(f'\n Descompactando o arquivo {arquivo_zip} ...')
    with zipfile.ZipFile(arquivo_zip) as myzip:
        myzip.extractall(path_destino)
        os.remove(arquivo_zip)
  
    print(f'\n Descompactação finalizada \n')





