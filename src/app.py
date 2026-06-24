import os
import sys

# Correção de escopo de módulo
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import streamlit as st
import cv2
import numpy as np
import joblib
from src.preprocessing import resize_rgb, to_grayscale, normalize, gaussian_filter, equalize_histogram
from src.transforms import sobel_filter, dct_blocks
from src.segmentation import otsu_threshold, morphological_open_close
from src.descriptors import extract_all
from src.loader import CAT_BREEDS

st.set_page_config(page_title="Cat Breed Classifier", layout="wide")

st.title("🐱 Classificador Interativo de Raças de Gatos")
st.write("### Trabalho Final — Processamento Digital de Imagens (Prof. Matheus Araújo)")
st.write("Faça o upload de uma foto de gato para visualizar as etapas do pipeline de PDI clássico e a classificação final.")

# Verificar se os modelos existem antes de rodar
models_ready = True
required_models = ["scaler.pkl", "selector.pkl", "svm_best.pkl"]
for model_name in required_models:
    if not os.path.exists(f"outputs/models/{model_name}"):
        models_ready = False

if not models_ready:
    st.error("🚨 Erro: Os modelos treinados não foram encontrados em `outputs/models/`. Execute o `pipeline.py` primeiro para treinar e salvar os arquivos `.pkl`.")
else:
    # Componente de Upload de Imagem
    uploaded_file = st.file_uploader("Selecione uma imagem de gato (JPG, JPEG ou PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Converter arquivo carregado para matriz OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = img_bgr[:, :, ::-1]

        st.markdown("---")
        st.subheader("🔄 Etapas do Pipeline de Processamento de Imagem (Implementação Manual)")

        # Executar os seus módulos manuais exatamente como no main
        img_resized = resize_rgb(img_rgb, (128, 128))
        img_gray = to_grayscale(img_resized)
        img_norm = normalize(img_gray)
        img_filtered = gaussian_filter(img_norm)
        img_equalized = equalize_histogram(img_filtered)
        
        sobel_mag, _ = sobel_filter(img_equalized)
        dct_blks = dct_blocks(img_equalized)
        
        thresh = otsu_threshold(img_equalized)
        binary_mask = (img_equalized >= thresh).astype(np.uint8)
        clean_mask = morphological_open_close(binary_mask)

        # Exibir as etapas lado a lado na interface em colunas
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(img_resized, caption="1. Original (128x128)", use_column_width=True)
        with col2:
            st.image(img_filtered, caption="2. Filtro Gaussiano", use_column_width=True, clamp=True)
        with col3:
            st.image(sobel_mag, caption="3. Bordas de Sobel", use_column_width=True, clamp=True)
        with col4:
            st.image(clean_mask * 255, caption="4. Máscara (Otsu + Morfologia)", use_column_width=True)
        with col5:
            masked_rgb = img_resized.copy()
            masked_rgb[clean_mask == 0] = 0
            st.image(masked_rgb, caption="5. Segmentação Resultante", use_column_width=True)

        st.markdown("---")
        st.subheader("🧠 Classificação e Inteligência Visual")

        with st.spinner("Extraindo características e computando inferência do SVM..."):
            # 1. Extrair o vetor de 99 características exatamente igual ao dataset de treino
            feats = extract_all(
                gray_image=img_equalized,
                rgb_image=img_resized,
                binary_mask=clean_mask,
                dct_blocks=dct_blks
            ).reshape(1, -1) # Reshape para formato de amostra única (1, 99)

            # 2. Carregar os artefatos de machine learning salvos
            scaler = joblib.load("outputs/models/scaler.pkl")
            selector = joblib.load("outputs/models/selector.pkl")
            svm_model = joblib.load("outputs/models/svm_best.pkl")

            # 3. Aplicar as transformações nos dados da nova imagem
            feats_scaled = scaler.transform(feats)
            feats_selected = selector.transform(feats_scaled)

            # 4. Predizer a raça e pegar as probabilidades de cada uma
            pred_id = svm_model.predict(feats_selected)[0]
            pred_probabilities = svm_model.predict_proba(feats_selected)[0]
            
            detected_breed = CAT_BREEDS[pred_id]
            confidence = pred_probabilities[pred_id] * 100

        # Exibir resultado com destaque visual
        st.success(f"### ✨ Raça Predita: **{detected_breed.replace('_', ' ')}** ({confidence:.2f}% de confiança)")

        # Mostrar gráfico de barras com as probabilidades para cada raça
        st.write("#### Distribuição de Probabilidade do Modelo:")
        chart_data = {CAT_BREEDS[i].replace('_', ' '): float(pred_probabilities[i]) for i in range(len(CAT_BREEDS))}
        st.bar_chart(chart_data)