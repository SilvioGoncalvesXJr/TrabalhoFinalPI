# Cat Breed Classifier — Pipeline Completo

Projeto acadêmico de Processamento Digital de Imagens.
Classifica 12 raças de gatos usando o Oxford-IIIT Pet Dataset.

## Técnicas implementadas (manualmente — NumPy)
- Conversão RGB → Grayscale (pesos ITU-R BT.601)
- Filtro Gaussiano 5×5 por convolução
- Equalização de histograma
- Filtro Sobel — detecção de borda e direção
- Filtro LoG — Laplaciano do Gaussiano
- DCT-II em blocos 8×8 (domínio da frequência)
- Limiarização de Otsu
- Morfologia binária (erosão, dilatação, abertura, fechamento)
- Histograma HSV
- LBP — Local Binary Pattern
- Descritores estatísticos (entropia, assimetria, curtose)

## Modelos de classificação (scikit-learn)
- KNN com variação de K (1 a 15) e validação cruzada 5-fold
- SVM RBF com GridSearchCV (C × gamma)

## Pré-requisitos
Python 3.10+

## Instalação
pip install -r requirements.txt

## Download do dataset
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xvf images.tar.gz      -C data/oxford-iiit-pet/
tar -xvf annotations.tar.gz -C data/oxford-iiit-pet/

## Execução
python src/pipeline.py

## Resultados
outputs/results/model_comparison.csv  — tabela comparativa KNN × SVM
outputs/visuals/                       — todas as figuras para o artigo

## Dataset
Oxford-IIIT Pet — CC BY-SA 4.0
https://www.robots.ox.ac.uk/~vgg/data/pets/

Download direto:
- https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
- https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
