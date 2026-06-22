# Cat Breed Classifier — Pré-processamento

Projeto acadêmico de Processamento Digital de Imagens.
Classifica 12 raças de gatos usando o Oxford-IIIT Pet Dataset.

## Técnicas implementadas (manualmente)
- Conversão RGB → Grayscale (pesos ITU-R BT.601)
- Filtro Gaussiano 5×5 (convolução NumPy)
- Equalização de histograma
- Filtro Sobel (detecção de borda)
- Filtro LoG — Laplaciano do Gaussiano
- DCT-II em blocos 8×8 (domínio da frequência)
- Limiarização de Otsu
- Morfologia binária (erosão, dilatação, abertura, fechamento)
- Histograma HSV
- LBP — Local Binary Pattern
- Descritores estatísticos (entropia, assimetria, curtose)

## Pré-requisitos
- Python 3.10+
- Ver requirements.txt

## Instalação
pip install -r requirements.txt

## Download do dataset
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xvf images.tar.gz -C data/oxford-iiit-pet/
tar -xvf annotations.tar.gz -C data/oxford-iiit-pet/

## Execução
python src/pipeline.py

## Saída para a Etapa 2 (classificação)
outputs/features/features.npz
  X      → (N, 99) float64 — vetores de features
  y      → (N,)    int32   — rótulos de raça (0–11)
  breeds → (12,)   str     — nomes das raças
  split  → (N,)    str     — "train" ou "test"

## Dataset
Oxford-IIIT Pet Dataset — CC BY-SA 4.0
https://www.robots.ox.ac.uk/~vgg/data/pets/
