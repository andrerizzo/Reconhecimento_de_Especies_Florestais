'''
Arquivo: evaluation.py
Autor: André Rizzo

Módulo para avaliação de modelos de classificação de imagens.
Inclui geração de gráficos de acurácia/perda e métricas de performance sobre o conjunto de teste.
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score)
import mlflow
import dagshub
import json
import tensorflow as tf


def training_eval(model, history, TAGS, RUN_NAME):
    """
    Plota os gráficos de perda e acurácia para treino e validação.

    Args:
        history (History): objeto retornado pelo model.fit()
    """
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(train_acc))

    
    plt.figure(figsize=(14, 5))

    # Gráfico de acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Treino')
    plt.plot(epochs_range, val_acc, label='Validação')
    plt.title('Acurácia por Época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend(loc='lower right')

    # Gráfico de perda
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Treino')
    plt.plot(epochs_range, val_loss, label='Validação')
    plt.title('Perda por Época')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    plt.savefig("training_graphs.png", dpi=300, bbox_inches="tight")
    plt.close()
    

    with open("training_history.json", "w") as f:
        json.dump(history.history, f)

    # Configuração do MLflow
    dagshub.init(repo_owner='andrerizzo', repo_name='wood-species-recognition', mlflow=True)

    EXPERIMENT_NAME = "wood-species-experiments"
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print(f"Criando novo experimento: {EXPERIMENT_NAME}")
        mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        print(f"✅ Usando experimento existente: {EXPERIMENT_NAME}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Log dos gráficos no MLflow
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_artifact("training_graphs.png")
        mlflow.log_artifact("training_history.json")
        mlflow.log_param("epochs", len(epochs_range))
        mlflow.log_metric("final_train_accuracy", train_acc[-1])
        mlflow.log_metric("final_val_accuracy", val_acc[-1])
        mlflow.log_metric("final_train_loss", train_loss[-1])
        mlflow.log_metric("final_val_loss", val_loss[-1])
        mlflow.set_tags(tags=TAGS)
        mlflow.end_run()



def inference_performance_metrics(model_path, test_dataset, class_names):
    """
    Gera relatório de métricas e matriz de confusão no conjunto de teste.

    Args:
        model (tf.keras.Model): modelo treinado.
        test_dataset (tf.data.Dataset): dataset de teste.
        class_names (list): nomes das classes.
    """
    
    # Carregar modelo
    model = tf.keras.models.load_model(model_path)

    # Predições
    y_true, y_pred, y_score = [], [], []
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_score.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))

        if labels.shape[-1] == len(class_names):  
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        else:
            y_true.extend(labels.numpy())

    y_true, y_pred, y_score = np.array(y_true), np.array(y_pred), np.array(y_score)



    # Classification Report
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    
    # PNG (texto renderizado)
    plt.figure(figsize=(10, 0.5 * len(class_names) + 4))
    plt.axis("off")
    plt.text(0.01, 0.99, report_text, {'fontsize': 10}, fontproperties="monospace", va="top")
    plt.title("Classification Report", fontsize=14)
    plt.savefig("classification_report.png", dpi=300, bbox_inches="tight")
    plt.close()
  

    # Relatório de classificação
    print("\nRelatório de Classificação:")  
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.show()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ROC
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
        plt.plot(fpr, tpr, label=f"{class_name}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Curva ROC-AUC")
    plt.xlabel("Falso Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.legend(loc="lower right")
    plt.savefig("roc_auc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


    # PR
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true == i, y_score[:, i])
        plt.plot(recall, precision, label=f"{class_name}")
    plt.title("Curva Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.savefig("pr_auc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


    # Métricas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    pr_auc = average_precision_score(y_true, y_score, average="weighted")

    # Log no MLflow
    with mlflow.start_run(run_name="inference_evaluation"):
        mlflow.log_artifact("classification_report.png")
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("roc_auc_curve.png")
        mlflow.log_artifact("pr_auc_curve.png")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.end_run()

