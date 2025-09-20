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
import seaborn as sns


# =======================================================
# Inicialização DagsHub + MLflow
# =======================================================
dagshub.init(repo_owner='andrerizzo', repo_name='wood-species-recognition', mlflow=True)

EXPERIMENT_NAME = "wood-species-experiments"
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    print(f"Criando novo experimento: {EXPERIMENT_NAME}")
    mlflow.create_experiment(EXPERIMENT_NAME)
else:
    print(f"Usando experimento existente: {EXPERIMENT_NAME}")
mlflow.set_experiment(EXPERIMENT_NAME)


# =======================================================
# Função auxiliar: extrair hiperparâmetros do modelo/treino
# =======================================================
def extract_model_params(model, history):
    params = {}

    # Informações do History (treino)
    if hasattr(history, "params"):
        params.update(history.params)

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
    mlflow.log_artifact("training_history.png", artifact_path="graphs")

    with open("training_history.json", "w") as f:
        json.dump(history.history, f)
    mlflow.log_artifact("training_history.json", artifact_path="reports")

    pd.DataFrame(history.history).to_csv("training_history.csv", index=False)
    mlflow.log_artifact("training_history.csv", artifact_path="reports")


# =======================================================
# Funções auxiliares: análises de confusões
# =======================================================
def top_confusions(y_true, y_pred, classes, top_n=10):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    confusoes = []
    for i, real in enumerate(classes):
        for j, pred in enumerate(classes):
            if i != j and cm[i, j] > 0:
                confusoes.append((real, pred, cm[i, j]))
    confusoes_sorted = sorted(confusoes, key=lambda x: x[2], reverse=True)[:top_n]
    return pd.DataFrame(confusoes_sorted, columns=["Classe Real", "Classe Predita", "Quantidade"])


def log_top_confusions(y_true, y_pred, class_names, salvar_no_gdrive=False, gdrive_dir=None):
    # ---------- Tabela Top 10 ----------
    df_top10 = top_confusions(y_true, y_pred, class_names, top_n=10)
    df_top10.to_csv("top10_confusoes.csv", index=False)
    df_top10.to_string(open("top10_confusoes.txt", "w"))
    mlflow.log_artifact("top10_confusoes.csv", artifact_path="reports")
    mlflow.log_artifact("top10_confusoes.txt", artifact_path="reports")

    # ---------- Gráfico de Barras Top 10 ----------
    df_top10["Confusão"] = df_top10["Classe Real"] + " → " + df_top10["Classe Predita"]
    df_top10_sorted = df_top10.sort_values("Quantidade", ascending=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Quantidade", y="Confusão", data=df_top10_sorted, palette="Blues_d")
    plt.xlabel("Quantidade de Erros")
    plt.ylabel("Confusão (Real → Predita)")
    plt.title("Top 10 Confusões")
    plt.tight_layout()
    plt.savefig("top10_confusoes_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact("top10_confusoes_bar.png", artifact_path="graphs")

    # ---------- Heatmap Top 20 ----------
    df_top20 = top_confusions(y_true, y_pred, class_names, top_n=20)
    pivot = df_top20.pivot(index="Classe Real", columns="Classe Predita", values="Quantidade").fillna(0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds", cbar=True)
    plt.title("Mapa de Calor - Top 20 Erros de Classificação")
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    plt.savefig("top20_confusoes_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact("top20_confusoes_heatmap.png", artifact_path="graphs")




# =======================================================
# Avaliação completa com logging automático
# =======================================================
def evaluate_model(model, history, test_dataset, class_names, run_name="model_eval", tags: dict = None, salvar_no_gdrive=False):
    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)

        params = extract_model_params(model, history)
        mlflow.log_params(params)

        # Histórico de treino
        log_training_history(history)

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
        with open("classification_report.txt", "w") as f:
            f.write(report_text)
        mlflow.log_artifact("classification_report.txt", artifact_path="reports")

        
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
        mlflow.log_artifact("roc_auc_curve.png", artifact_path="graphs")

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
        mlflow.log_artifact("pr_auc_curve.png", artifact_path="graphs")

        # Métricas
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

        # Modelo + history no GDrive
        if salvar_no_gdrive:
            os.makedirs(gdrive_dir, exist_ok=True)
            gdrive_model_path = f"{gdrive_dir}/MobileNetv3_small_v1.keras"
            model.save(gdrive_model_path)
            hist_json = f"{gdrive_dir}/MobileNetv3_small_v1_history.json"
            hist_csv = f"{gdrive_dir}/MobileNetv3_small_v1_history.csv"
            with open(hist_json, "w") as f:
                json.dump(history.history, f)
            pd.DataFrame(history.history).to_csv(hist_csv, index=False)
            mlflow.log_param("modelo_path_gdrive", gdrive_model_path)
            mlflow.log_param("history_json_gdrive", hist_json)
            mlflow.log_param("history_csv_gdrive", hist_csv)
            with open("model_path.txt", "w") as f:
                f.write(gdrive_model_path)
            mlflow.log_artifact("model_path.txt")
            print(f"Modelo salvo no Google Drive em: {gdrive_model_path}")
            print(f"History salvo no Google Drive em: {hist_json} e {hist_csv}")
            print("Caminhos registrados no MLflow (DagsHub)")
        
        # Matriz de Confusão
        log_top_confusions(y_true, y_pred, class_names)

