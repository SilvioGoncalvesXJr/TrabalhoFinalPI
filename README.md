# Cat Breed Classifier — Pipeline Completo

Este repositório contém o projeto aplicado de Processamento de Imagens (PI) desenvolvido como o Trabalho Final da disciplina. O objetivo é desenvolver um pipeline completo de visão computacional para classificação de **12 raças de gatos** utilizando o dataset **Oxford-IIIT Pet Dataset**.

O projeto aplica técnicas clássicas de processamento digital de imagens implementadas manualmente com NumPy, realizando pré-processamento, transformações espaciais e no domínio da frequência, segmentação, extração de descritores e classificação utilizando Machine Learning.

Tema Selecionado: Animais e Natureza — Gatos

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

Dataset utilizado: **Oxford-IIIT Pet Dataset**

Oxford-IIIT Pet — CC BY-SA 4.0 https://www.robots.ox.ac.uk/~vgg/data/pets/

Download direto:
https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

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
├── data/                         # Subconjunto filtrado do dataset (Ignorado no Git)
│   └── oxford-iiit-pet/
│       ├── images/               # Fotos originais (.jpg)
│       └── annotations/          # Trimaps e bounding boxes de referência
├── notebooks/
│   └── exploratory_analysis.ipynb # Análise Exploratória de Dados (Parte A e Parte B)
├── outputs/                      # Artefatos gerados automaticamente pelo pipeline
│   ├── features/                 # Vetores de características salvos (.npz)
│   ├── models/                   # Serialização dos modelos e scalers (.pkl)
│   ├── results/                  # Tabelas comparativas e relatórios de métricas (.csv)
│   └── visuals/                  # Gráficos de erro, heatmaps e matrizes de confusão (.png)
├── src/                          # Código-fonte modularizado em Python
│   ├── __init__.py
│   ├── app.py                    # Aplicação Web Interativa (Streamlit Dashboard)
│   ├── pipeline.py               # Script principal (Orquestrador do Pipeline)
│   ├── loader.py                 # Filtro, leitura e estruturação dos dados
│   ├── preprocessing.py          # Operações espaciais manuais e realce
│   ├── transforms.py             # Filtros de borda (Sobel, LoG) e domínio da frequência (DCT)
│   ├── segmentation.py           # Otsu, Morfologia Binária e validação por IoU
│   ├── classification.py         # Classificador com SVM e KNN
│   └── descriptors.py            # Funções manuais de extração de características
├── docs/
│   └── artigo_pdi_sbc.pdf        # Relatório Científico Final no formato SBC
├── requirements.txt              # Dependências e bibliotecas do ambiente
└── README.md                     # Este arquivo de documentação
```

---

# Análise Exploratória de Dados (AED)
O ciclo de análise de dados foi dividido em duas frentes complementares dentro do arquivo notebooks/exploratory_analysis.ipynb:

## Parte A: Análise Pré-Pipeline (Dados Brutos)
Estudo estatístico focado na consistência geométrica e na integridade das amostras originais:
- Distribuição de frequência volumétrica das 12 raças de gatos para checagem de desbalanceamento de classes.
- Mapeamento de dispersão dimensional (Largura $\times$ Altura) das imagens originais antes da padronização espacial.
- Distribuição do balanço dos splits nativos de treino e teste.

## Parte B: Análise Pós-Pipeline (Features Extraídas)
Estudo sobre o comportamento matemático do vetor de 99 características gerado pelos módulos de PI:
- Análise de relevância e variância dos bins dos histogramas de cor (HSV) e textura (LBP) entre diferentes raças.
- Correlação linear entre os descritores de forma (área, perímetro, compacidade) e as classes.
- Avaliação do comportamento da redução de dimensionalidade por variância (ANOVA) para a escolha das 50 melhores features que alimentam os classificadores.

# Módulo Extra: Aplicação Web Interativa (Streamlit)
Como um artefato funcional bônus, o projeto conta com uma interface gráfica para o usuário realizar testes preditivos em tempo real.

## Funcionalidades:
- Drag-and-drop para upload de imagens externas de felinos (.png, .jpg, .jpeg).
- Renderização visual sequencial lado a lado: Original -> Filtro Gaussiano -> Bordas de Sobel -> Máscara de Otsu -> Segmentação Resultante.
- Inferência instantânea carregando os modelos otimizados (svm_best.pkl) com exibição gráfica da distribuição de probabilidade por classe.

## Demonstração Visual da Interface:
<img width="1374" height="834" alt="Captura de Tela 2026-06-24 às 22 08 44" src="https://github.com/user-attachments/assets/93dc7fc9-e2e7-45db-9883-04e8c607a95a" />

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


## Download e Organização do Dataset

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

# Executar o Pipeline Principal (Treinamento e Extração)

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

# Executar a Aplicação Web (Streamlit)

Após a conclusão do pipeline (garantindo que os arquivos .pkl foram salvos em outputs/models/), inicialize o servidor do dashboard local com o comando:

```bash
streamlit run src/app.py
```

O terminal exibirá um link local (geralmente http://localhost:8501). Abra-o no seu navegador para interagir com o sistema.

---

# Slides da Apresentação
[Classificação Avançada de Raças de Gatos via Processamento Digital de Imagens e Aprendizado Visual.pdf](https://github.com/user-attachments/files/29316126/Classificacao.Avancada.de.Racas.de.Gatos.via.Processamento.Digital.de.Imagens.e.Aprendizado.Visual.pdf)

---

# Equipe

- Layza Carneiro ✉️ E-mail: layza.carneiro@aluno.uece.br
- Pedro Nonato ✉️ E-mail: pedro.nonato@aluno.uece.br
- Samuel Valente ✉️ E-mail: samuel.valente@aluno.uece.br
- Silvio Gonçalves ✉️ E-mail: silvio.goncalves@aluno.uece.br
