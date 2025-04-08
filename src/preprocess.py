# Módulo: Pré-processamento de Dados


import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory


#train_path = '/content/train'
#val_path = '/content/val'
#test_path = '/content/test'

#IMG_SIZE = (224, 224)


def gera_dataset_train_test_val(path_imagem, classes, tipo_classes, tam_batch=32, 
                                tam_imagem=(256,256), aleatorio=True, percent_val=None, 
                                subconjunto='validation', symlink=False, ):
    '''
    
        Args:
            path_imagem (str): caminho para local onde as imagens / symlinks estão armazenadas(os).
            classes ('inferred' ou 'None'): os nomes das classes são extraídas do nome dos diretórios das imagens.
            tipo_classes ('int', 'categorical', 'binary', 'None') 
            tam_batch (int): dimensão do batch
            tam_imagem (altura,largura): dimensões das imagens em pixels
            aleatorio (bool): True se deve realizar amostragem aleatória
            percent_val (float[0,1]): percentual de dados para validação
            subconjunto ('training', 'validation' ou both')
            symlink (bool): True se existirem ponteiros (symlinks) para as imagens reais
            

        Retorna:
            train_images OU teste_images OU val_images 
    '''
   
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path_imagem,
        labels = classes,
        label_mode=tipo_classes,
        batch_size=tam_batch,
        image_size=tam_imagem,
        shuffle=aleatorio,
        seed=42,
        validation_split=percent_val,  
        subset=subconjunto,
        follow_links=symlink
        
    )
    return dataset


def preprocess_config(image, label, preprocessing_fn):
    '''
    Aplica a função de pré-processamento fornecida
    '''
    image = preprocessing_fn(image)
    return image, label


def preprocess(dataset, preprocessing_fn):
    '''
    Realiza o pré-processamento usando a função passada como parâmetro

    Uso:
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess
        
        Supondo que você já tenha um dataset carregado
        preprocessed_ds = preprocess(dataset, preprocessing_fn=mobilenet_v3_preprocess)
    '''
    AUTOTUNE = tf.data.AUTOTUNE

    preprocessed_dataset = (
        dataset
        .map(lambda image, label: preprocess_config(image, label, preprocessing_fn), num_parallel_calls=AUTOTUNE)
        .prefetch(buffer_size=AUTOTUNE)
    )
    return preprocessed_dataset

