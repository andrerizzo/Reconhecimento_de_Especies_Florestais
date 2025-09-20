# Reconhecimento de Espécies Florestais com Visão Computacional

**Autor:** André Rizzo  
**Afiliado à:** Universidade de São Paulo (USP)  
**Contato:** andre.rizzo@hotmail.com

## 📌 Objetivo

Este projeto visa aplicar técnicas de Deep Learning para o reconhecimento de espécies de madeira amazônica a partir de imagens macroscópicas. O foco é viabilizar a execução do modelo em dispositivos com recursos computacionais limitados, como smartphones de entrada, permitindo sua utilização por agentes de fiscalização ambiental em campo.

## 🧠 Abordagem Técnica

- **Arquiteturas Testadas:**
  - MobileNetV3 Large v1
  - MobileNetV3 Large v2
  - MobileNetV3 Small v1 *(melhor desempenho)*
  
- **Técnica:** Fine-tuning em imagens segmentadas
- **Input:** Imagens de alta resolução (3264x2448) divididas em patches de 224x224
- **Classificação Final:** Votação majoritária entre os patches
- **Total de Imagens:** 2.942 imagens de 41 espécies distintas
- **Total de Patches:** ~348.880

## 📈 Resultados

| Modelo                      | Acurácia | Precisão | Recall | Parâmetros (estimado) | Observações                                                              |
|----------------------------|----------|----------|--------|------------------------|---------------------------------------------------------------------------|
| MobileNetV3 Large v1       | -        | -        | -      | ~5.4M                  | Modelo com maior capacidade, sem ganho de desempenho significativo        |
| MobileNetV3 Large v2       | -        | -        | -      | ~5.4M                  | Versão alternativa do modelo large                                        |
| MobileNetV3 Small v1 (melhor) | 98%   | 98%      | 98%    | ~1.3M                  | Melhor desempenho geral com menor custo computacional                    |

## 🔁 Pipeline do Projeto

1. **Importação das imagens**  
2. **Segmentação em patches (224x224)**  
3. **Pré-processamento (normalização, redimensionamento)**  
4. **Treinamento com MobileNetV3 Small**  
5. **Classificação individual dos patches**  
6. **Agregação por votação majoritária**

## 📁 Estrutura do Projeto

```
project/
├── notebook/              # Notebooks de exploração e análise
├── src/                   # Scripts de processamento e modelagem
├── img/                   # Imagens do projeto (ex: amostras, gráficos)
├── data/                  # Conjunto de dados (estruturado)
├── docs/                  # Documentação adicional
├── models/                # Pesos e artefatos dos modelos treinados
├── Readme.md              # Este arquivo
├── Readme_EN.md           # Versão em inglês (opcional)
├── requirements.txt       # Dependências do projeto
└── .gitignore
```

## 🛠️ Tecnologias Utilizadas

- Python 3.11+
- TensorFlow / Keras
- OpenCV
- NumPy / Pandas
- Google Colab (para treinamento com GPU)
- Matplotlib / Seaborn

## 🌱 Aplicações e Impacto

- Suporte à fiscalização ambiental contra extração ilegal de madeira
- Automação de um processo altamente dependente de especialistas humanos
- Aplicação prática de IA em **ambientes de baixa capacidade computacional**



**André Rizzo**  
Cientista de Dados | Especialista em Visão Computacional  
[LinkedIn](https://www.linkedin.com/in/andrerizzo) • andre.rizzo@hotmail.com  
Rio de Janeiro, RJ – Brasil
