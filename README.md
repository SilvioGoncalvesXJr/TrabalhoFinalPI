# Cat Breed Classifier — Pipeline Completo

Projeto acadêmico de Processamento de Imagens (PI).

O objetivo é desenvolver um pipeline completo de visão computacional para classificação de **12 raças de gatos** utilizando o dataset **Oxford-IIIT Pet Dataset**.

O projeto aplica técnicas clássicas de processamento digital de imagens implementadas manualmente com NumPy, realizando pré-processamento, transformações espaciais e no domínio da frequência, segmentação, extração de descritores e classificação utilizando Machine Learning.

---

## Objetivo

Classificar raças de gatos a partir de imagens digitais analisando características visuais de:

- Cor;
- Textura;
- Forma;
- Frequência.

O pipeline foi desenvolvido seguindo os requisitos da disciplina, priorizando a implementação manual dos algoritmos estudados.

---

# Dataset

Dataset utilizado:

**Oxford-IIIT Pet Dataset**

Tema: Animais e Natureza — Gatos

Licença: 
```
CC BY-SA 4.0
```

O dataset contém imagens anotadas de animais domésticos. Neste projeto foram selecionadas apenas 12 raças de gatos:
- Abyssinian
- Bengal
- Birman
- Bombay
- British Shorthair
- Egyptian Mau
- Maine Coon
- Persian
- Ragdoll
- Russian Blue
- Siamese
- Sphynx

---

# Pipeline Implementado

```text
Imagem
  |
  v
Pré-processamento
  |
  v
Transformações
  |
  v
Segmentação
  |
  v
Extração de características
  |
  v
Classificação
  |
  v
Resultados
```

---

# Técnicas Implementadas

Todas as etapas principais de processamento foram implementadas manualmente utilizando NumPy e OpenCV foi utilizado apenas para leitura, escrita e operações básicas de imagem.

## Pré-processamento

- Conversão RGB → Grayscale utilizando ITU-R BT.601;
- Redimensionamento por Vizinho Próximo;
- Normalização de intensidade;
- Filtro Gaussiano 5×5 por convolução;
- Equalização de histograma.

## Transformações Espaciais

- Filtro Sobel:
  - Gradiente horizontal;
  - Gradiente vertical;
  - Magnitude da borda.

- Filtro LoG (Laplaciano do Gaussiano);

- DCT-II em blocos 8×8 para análise no domínio da frequência.

## Segmentação

- Limiarização de Otsu;
- Morfologia binária:
  - Erosão;
  - Dilatação;
  - Abertura;
  - Fechamento.

- Avaliação por IoU usando os trimaps do dataset.

## Descritores Extraídos

Cada imagem gera um vetor com **99 características**:

### Estatísticas (5)

- Média;
- Desvio padrão;
- Entropia;
- Assimetria;
- Curtose.

### Cor (48)

Histograma HSV:

- 16 bins por canal;
- H, S e V.

### Forma (4)

- Área;
- Perímetro;
- Compacidade;
- Razão de aspecto.

### Frequência (10)

Coeficientes centrais da DCT.

### Textura (32)

Local Binary Pattern (LBP).

---

# Classificação

Foi utilizado `scikit-learn` na etapa de aprendizado.

Modelos:

## KNN

- Teste de K = 1 até 15;
- Validação cruzada 5-fold.

## SVM RBF

Otimização utilizando:

- GridSearchCV;
- Parâmetros:
  - C;
  - Gamma.

Pipeline:

```
StandardScaler
        ↓
SelectKBest (ANOVA)
        ↓
KNN / SVM
```

---

# Estrutura do Projeto

```
TrabalhoFinalPI/

├── data/
│   └── oxford-iiit-pet/
│       ├── images/
│       └── annotations/

├── outputs/
│   ├── features/
│   ├── models/
│   ├── results/
│   └── visuals/

├── notebooks/
│   ├── exploratory_analysis.ipynb

├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── loader.py
│   ├── preprocessing.py
│   ├── transforms.py
│   ├── segmentation.py
│   ├── descriptors.py
│   └── classification.py

├── requirements.txt
└── README.md
```

---

# Instalação

Python recomendado: 3.10+

## Criar ambiente virtual

```bash
python3 -m venv .venv
```

Ativar:

Mac/Linux:

```bash
source .venv/bin/activate
```

Instalar dependências:

```bash
pip install -r requirements.txt
```

---


## Download

Criar pasta:

```bash
mkdir -p data/oxford-iiit-pet
```

Baixar:

```bash
curl -L -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz

curl -L -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```

Extrair:

```bash
tar -xzf images.tar.gz -C data/oxford-iiit-pet/

tar -xzf annotations.tar.gz -C data/oxford-iiit-pet/
```

Após isso:

```
data/oxford-iiit-pet/

├── images/
└── annotations/
```

---

# Execução

Criar diretórios de saída:

```bash
mkdir -p outputs/features outputs/results outputs/models outputs/visuals
```

Executar pipeline completo:

```bash
.venv/bin/python -m src.pipeline
```

O pipeline irá:

1. Carregar imagens;
2. Aplicar processamento;
3. Extrair features;
4. Salvar vetores;
5. Treinar classificadores;
6. Gerar resultados.

---

# Resultados Gerados

Após execução:

## Features

```
outputs/features/features.npz
```

Contém:

- Vetores de características;
- Labels;
- Divisão treino/teste.

---

## Estatísticas

```
outputs/results/
```

Inclui:

- Comparação KNN × SVM;
- Métricas;
- Relatórios.

---

## Visualizações

```
outputs/visuals/
```

Inclui:

- Imagem original;
- Grayscale;
- Filtro Gaussiano;
- Sobel;
- Segmentação;
- Máscara final.

---

## Dataset
Oxford-IIIT Pet — CC BY-SA 4.0 https://www.robots.ox.ac.uk/~vgg/data/pets/

Download direto:

https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

---

# Equipe

- Layza Carneiro
- Pedro Nonato
- Samuel Valente
- Silvio Gonçalves