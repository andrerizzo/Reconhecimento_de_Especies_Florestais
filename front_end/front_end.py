# Front-end


import streamlit as st
import numpy as np
import time
import pandas as pd
from PIL import Image
from patchify import patchify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import collections

# Lista com os nomes das espécies (classe 1 está no índice 0)
class_names = [
    "Arcoarcarpus", "Amapá", "Andiroba", "Angelim Pedra", "Araucária",
    "Assacu", "Bracatinga", "Cabriúva Vermelha", "Castanheira", "Cedrinho",
    "Cedro", "Cedrorana", "Cinamomo", "Cumaru", "Cupiúba",
    "Curupixa", "Eucalipto", "Freijó", "Goiabão", "Grevílea",
    "Imbuia", "Ipê", "Itaúba", "Jatobá", "Jequitibá",
    "Louro", "Machærium", "Massaranduba", "Mogno", "Louro amarelo",
    "Pau amarelo", "Pau marfim", "Peroba rosa", "Pinus", "Podocarpus",
    "Quaruba", "Roxinho", "Sucupira", "Tatajuba", "Taurí", "Virola"
]

# Caminho do modelo
modelo = load_model(r"G:\Meu Drive\USP\TCC\Application\models\MobileNetv3_small_v1_.keras")

# Função para redimensionar para múltiplos de 224
def ajustar_dimensoes_para_patches(imagem, patch_size=224):
    largura, altura = imagem.size
    nova_largura = (largura // patch_size) * patch_size
    nova_altura = (altura // patch_size) * patch_size

    if nova_largura != largura or nova_altura != altura:
        imagem = imagem.resize((nova_largura, nova_altura))
        ajuste = True
    else:
        ajuste = False

    return imagem, ajuste, nova_largura, nova_altura

# Título
st.title("🌳 Classificador de Espécies de Madeira com Patches")

arquivo = st.file_uploader("Envie uma imagem de alta resolução (ex: 3800x2600)", type=["jpg", "jpeg", "png"])

if arquivo:
    # Abrir imagem
    imagem = Image.open(arquivo).convert("RGB")
    imagem, foi_redimensionada, nova_largura, nova_altura = ajustar_dimensoes_para_patches(imagem)

    st.image(imagem, caption="Imagem ajustada" if foi_redimensionada else "Imagem original", use_container_width=True)

    if foi_redimensionada:
        st.warning(f"A imagem foi redimensionada para {nova_largura}x{nova_altura} para garantir divisão exata em patches de 224x224.")

    # Convertendo imagem para array
    imagem_np = np.array(imagem)

    # Gerando patches com patchify
    patches = patchify(imagem_np, (224, 224, 3), step=224)
    patch_list = [patches[i, j, 0] for i in range(patches.shape[0]) for j in range(patches.shape[1])]

    st.info(f"🔍 Total de patches gerados: {len(patch_list)}")

    # Pré-processamento dos patches
    patches_preproc = np.array([preprocess_input(p) for p in patch_list])

    # Inferência com tempo
    inicio = time.time()
    preds = modelo.predict(patches_preproc, verbose=0)
    fim = time.time()
    tempo_total = fim - inicio

    # Classes previstas
    indices_preditos = np.argmax(preds, axis=1)
    votos = collections.Counter(indices_preditos)
    classe_vencedora = votos.most_common(1)[0][0]
    especie_predita = class_names[classe_vencedora]
    confianca = votos[classe_vencedora] / len(indices_preditos)

    # Resultado principal
    st.subheader("✅ Espécie predominante na imagem:")
    st.success(f"**{especie_predita}** (Classe #{classe_vencedora + 1}) — Confiança: **{confianca:.2%}**")

    # Top 5 classes mais votadas
    top5 = votos.most_common(5)
    top5_data = [(i+1, class_names[i], f"{votos[i]/len(indices_preditos):.4f}") for i, _ in top5]
    df_top5 = pd.DataFrame(top5_data, columns=["Classe #", "Espécie", "Frequência"])
    st.subheader("🔝 Top 5 espécies mais votadas:")
    st.table(df_top5)

    # Tempo
    st.markdown(f"⏱️ **Tempo total de inferência:** `{tempo
