'''
Arquivo: evaluation.py
Autor: André Rizzo

Módulo para avaliação de modelos de classificação de imagens.
Inclui geração de gráficos de acurácia/perda e métricas de performance sobre o conjunto de teste.
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def plot_training_history(history):
    """
    Plota os gráficos de perda e acurácia para treino e validação.

    Args:
        history (History): objeto retornado pelo model.fit()
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    # Gráfico de acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Treino')
    plt.plot(epochs_range, val_acc, label='Validação')
    plt.title('Acurácia por Época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend(loc='lower right')

    # Gráfico de perda
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Treino')
    plt.plot(epochs_range, val_loss, label='Validação')
    plt.title('Perda por Época')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def performance_metrics(model, test_dataset, class_names):
    """
    Gera relatório de métricas e matriz de confusão no conjunto de teste.

    Args:
        model (tf.keras.Model): modelo treinado.
        test_dataset (tf.data.Dataset): dataset de teste.
        class_names (list): nomes das classes.
    """
    y_true = []
    y_pred = []

    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.show()