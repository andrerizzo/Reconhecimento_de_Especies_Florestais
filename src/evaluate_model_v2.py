"""
Arquivo: evaluate_model.py
Autor: André Rizzo

Módulo de avaliação e logging no MLflow.
Deve ser chamado após o treinamento com train_model().
"""

import matplotlib.pyplot as plt
import numpy as np
import mlflow
import dagshub
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
import io
import contextlib
import json
import pandas as pd
import os


# =======================================================
# Inicialização DagsHub + MLflow
# =======================================================
dagshub.init(repo_owner='andrerizzo', repo_name='wood-species-recognition', mlflow=True)

EXPERIMENT_NAME = "wood-species-experiments"
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    print(f"🔧 Criando novo experimento: {EXPERIMENT_NAME}")
    mlflow.create_experiment(EXPERIMENT_NAME)
else:
    print(f"✅ Usando experimento existente: {EXPERIMENT_NAME}")
mlflow.set_experiment(EXPERIMENT_NAME)


# =======================================================
# Função auxiliar: extrair hiperparâmetros do modelo/treino
# =======================================================
def extract_model_params(model, history):
    params = {}

    # Informações do History (treino)
    if hasattr(history, "params"):
        params.update(history.params)  # inclui epochs, steps, batch_size etc.

    # Otimizador
    if hasattr(model, "optimizer") and hasattr(model.optimizer, "get_config"):
        opt_conf = model.optimizer.get_config()
        for k, v in opt_conf.items():
            params[f"optimizer_{k}"] = v
        params["optimizer_name"] = model.optimizer.__class__.__name__

    # Loss
    try:
        params["loss"] = model.loss if isinstance(model.loss, str) else model.loss.__class__.__name__
    except Exception:
        params["loss"] = str(model.loss)

    # Métricas
    if hasattr(model, "metrics_names"):
        params["metrics_tracked"] = ",".join(model.metrics_names)

    # Arquitetura do modelo
    try:
        params["input_shape"] = str(model.input_shape)
        params["output_shape"] = str(model.output_shape)
        params["num_layers"] = len(model.layers)
        params["num_params_total"] = model.count_params()
    except Exception:
        pass

    return params


# =======================================================
# Gráfico de histórico de treino
# =======================================================
def log_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    # Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Treino')
    plt.plot(epochs_range, val_acc, label='Validação')
    plt.title('Acurácia por Época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend(loc='lower right')

    # Perda
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Treino')
    plt.plot(epochs_range, val_loss, label='Validação')
    plt.title('Perda por Época')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact("training_history.png")

    # Também salvar como JSON e CSV (para MLflow)
    with open("training_history.json", "w") as f:
        json.dump(history.history, f)
    mlflow.log_artifact("training_history.json")

    pd.DataFrame(history.history).to_csv("training_history.csv", index=False)
    mlflow.log_artifact("training_history.csv")


# =======================================================
# Log do resumo do modelo
# =======================================================
def log_model_summary(model):
    # Captura o resumo como string
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        model.summary()
    summary_str = stream.getvalue()

    # Salva como TXT
    with open("model_summary.txt", "w") as f:
        f.write(summary_str)
    mlflow.log_artifact("model_summary.txt")

    # Salva também como PNG (texto renderizado)
    plt.figure(figsize=(12, 0.3 * len(summary_str.splitlines())))
    plt.axis("off")
    plt.text(0.01, 0.99, summary_str, {"fontsize": 8}, fontproperties="monospace", va="top")
    plt.title("Model Summary", fontsize=12)
    plt.savefig("model_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact("model_summary.png")


# =======================================================
# Avaliação completa com logging automático
# =======================================================
def evaluate_model(model, history, test_dataset, class_names, run_name="model_eval", tags: dict = None):
    """
    Avalia modelo treinado, gera gráficos e logs no MLflow.

    Args:
        model (tf.keras.Model): modelo treinado
        history (keras.callbacks.History): histórico do treinamento
        test_dataset (tf.data.Dataset): dataset de teste
        class_names (list): lista de classes
        run_name (str): nome do run no MLflow
        tags (dict, opcional): dicionário de tags a registrar no MLflow
    """
    with mlflow.start_run(run_name=run_name):
        # Se houver tags, registra
        if tags:
            mlflow.set_tags(tags)

        # Extração automática de parâmetros
        params = extract_model_params(model, history)
        mlflow.log_params(params)

        # Log do resumo do modelo
        log_model_summary(model)

        # 1. Log histórico (imagem + dados)
        log_training_history(history)

        # 2. Predições
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

        # 3. Classification Report (txt + png simples)
        report_text = classification_report(y_true, y_pred, target_names=class_names)

        # TXT
        with open("classification_report.txt", "w") as f:
            f.write(report_text)
        mlflow.log_artifact("classification_report.txt")

        # PNG (texto renderizado)
        plt.figure(figsize=(10, 0.5 * len(class_names) + 4))
        plt.axis("off")
        plt.text(0.01, 0.99, report_text, {'fontsize': 10}, fontproperties="monospace", va="top")
        plt.title("Classification Report", fontsize=14)
        plt.savefig("classification_report.png", dpi=300, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("classification_report.png")

        # 4. Matriz de Confusão
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de Confusão")

        cm_file = "confusion_matrix.png"
        plt.savefig(cm_file, dpi=300, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_file)

        # 5. ROC
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
        mlflow.log_artifact("roc_auc_curve.png")

        # 6. PR
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
        mlflow.log_artifact("pr_auc_curve.png")

        # 7. Métricas
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        pr_auc = average_precision_score(y_true, y_score, average="weighted")

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)
        mlflow.log_metric("test_pr_auc", pr_auc)

        # 8. Salvar modelo + history no Google Drive
        gdrive_dir = "/content/drive/MyDrive/modelos"
        os.makedirs(gdrive_dir, exist_ok=True)

        gdrive_model_path = f"{gdrive_dir}/MobileNetv3_small_v1.keras"
        model.save(gdrive_model_path)

        hist_json = f"{gdrive_dir}/MobileNetv3_small_v1_history.json"
        hist_csv = f"{gdrive_dir}/MobileNetv3_small_v1_history.csv"

        with open(hist_json, "w") as f:
            json.dump(history.history, f)
        pd.DataFrame(history.history).to_csv(hist_csv, index=False)

        # Registrar caminhos no MLflow
        mlflow.log_param("modelo_path_gdrive", gdrive_model_path)
        mlflow.log_param("history_json_gdrive", hist_json)
        mlflow.log_param("history_csv_gdrive", hist_csv)

        with open("model_path.txt", "w") as f:
            f.write(gdrive_model_path)
        mlflow.log_artifact("model_path.txt")

        print(f"Modelo salvo no Google Drive em: {gdrive_model_path}")
        print(f"History salvo no Google Drive em: {hist_json} e {hist_csv}")
        print("Caminhos registrados no MLflow (DagsHub)")
