import os
import logging
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from .loader import load_dataset, CAT_BREEDS
from .preprocessing import to_grayscale, normalize, gaussian_filter, equalize_histogram
from .transforms import sobel_filter, log_filter, dct_blocks
from .segmentation import otsu_threshold, morphological_open_close, calculate_iou, load_trimap
from .descriptors import extract_all
from . import classification

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(module)-20s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("outputs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reprodutibilidade
SEED = 42
np.random.seed(SEED)


def main():
    data_dir = "data/oxford-iiit-pet"
    df = load_dataset(data_dir)

    if len(df) == 0:
        logger.error("Nenhuma imagem carregada! Verifique o dataset.")
        return

    features_list = []
    labels_list = []
    paths_list = []
    split_list = []
    otsu_list = []
    iou_list = []
    breed_iou_dict = {breed: [] for breed in CAT_BREEDS}

    # Para visualizações: selecionar 3 imagens por raça
    sample_paths = []
    for breed in CAT_BREEDS:
        breed_df = df[df["breed_name"] == breed]
        if len(breed_df) >= 3:
            sample_paths.extend(breed_df["image_path"].head(3).tolist())

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processando imagens"):
        img_path = row["image_path"]
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                logger.warning(f"Imagem não carregada: {img_path}")
                continue
            img_rgb = img_bgr[:, :, ::-1]

            # Pré-processamento
            img_resized = cv2.resize(img_rgb, (128, 128))
            img_gray = to_grayscale(img_resized)
            img_norm = normalize(img_gray)
            img_filtered = gaussian_filter(img_norm)
            img_equalized = equalize_histogram(img_filtered)

            # Transformações
            sobel_mag, _ = sobel_filter(img_equalized)
            log_edges = log_filter(img_equalized)
            dct_blks = dct_blocks(img_equalized)

            # Segmentação
            thresh = otsu_threshold(img_equalized)
            binary_mask = (img_equalized >= thresh).astype(np.uint8)
            clean_mask = morphological_open_close(binary_mask)

            # IoU com trimap
            iou = -1
            img_name = Path(img_path).stem
            trimap_path = Path(data_dir) / "annotations" / "trimaps" / f"{img_name}.png"
            if trimap_path.exists():
                true_mask = load_trimap(str(trimap_path))
                true_mask_resized = cv2.resize(true_mask, (128, 128), interpolation=cv2.INTER_NEAREST)
                iou = calculate_iou(clean_mask, true_mask_resized.astype(np.uint8))
                breed_iou_dict[row["breed_name"]].append(iou)

            # Extrair features
            feats = extract_all(
                gray_image=img_equalized,
                rgb_image=img_resized,
                binary_mask=clean_mask,
                dct_blocks=dct_blks
            )

            features_list.append(feats)
            labels_list.append(row["breed_id"])
            paths_list.append(img_path)
            split_list.append(row["split"])
            otsu_list.append(thresh)
            iou_list.append(iou)

            # Salvar visualizações para amostras
            if img_path in sample_paths:
                save_visualization(
                    img_resized, img_gray, img_filtered, sobel_mag, clean_mask,
                    img_path, "outputs/visuals"
                )

        except Exception as e:
            logger.error(f"Erro processando {img_path}: {e}")
            continue

    # Salvar features
    np.savez(
        "outputs/features/features.npz",
        X=np.array(features_list),
        y=np.array(labels_list, dtype=np.int32),
        paths=np.array(paths_list),
        breeds=np.array(CAT_BREEDS),
        split=np.array(split_list)
    )
    logger.info("Features salvas em outputs/features/features.npz")
    logger.info(f"Shape X: {np.array(features_list).shape}")

    # Salvar estatísticas
    save_statistics(df, features_list, otsu_list, iou_list, breed_iou_dict)

    # Salvar outras visualizações (histogramas RGB, LBP, DCT)
    save_other_visuals(df, "outputs/visuals")

    # ETAPA 2: Classificação
    logger.info("=== INICIANDO ETAPA 2: CLASSIFICAÇÃO ===")
    classification.run_classification_pipeline()
    logger.info("=== PIPELINE COMPLETO ===")


def save_visualization(rgb, gray, filtered, sobel, mask, img_path, out_dir):
    """Salva figura com etapas do pipeline para uma imagem."""
    Path(out_dir).mkdir(exist_ok=True)
    base_name = Path(img_path).stem

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(filtered, cmap="gray")
    axes[2].set_title("Gaussian Filter")
    axes[2].axis("off")

    axes[3].imshow(sobel, cmap="gray")
    axes[3].set_title("Sobel")
    axes[3].axis("off")

    axes[4].imshow(mask, cmap="gray")
    axes[4].set_title("Segmentação")
    axes[4].axis("off")

    masked_rgb = rgb.copy()
    masked_rgb[mask == 0] = 0
    axes[5].imshow(masked_rgb)
    axes[5].set_title("Máscara Aplicada")
    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{base_name}_pipeline.png", dpi=150)
    plt.close()


def save_statistics(df, features_list, otsu_list, iou_list, breed_iou_dict):
    """Salva arquivos de estatística em outputs/stats/."""
    out_dir = Path("outputs/stats")
    out_dir.mkdir(exist_ok=True)

    # Otsu thresholds
    otsu_df = pd.DataFrame({
        "image_path": df["image_path"],
        "breed_name": df["breed_name"],
        "threshold": otsu_list
    })
    otsu_df.to_csv(out_dir / "otsu_thresholds.csv", index=False)

    # IoU por raça
    iou_data = []
    for breed in CAT_BREEDS:
        if len(breed_iou_dict[breed]) > 0:
            mean_iou = np.mean(breed_iou_dict[breed])
            std_iou = np.std(breed_iou_dict[breed])
        else:
            mean_iou = -1
            std_iou = -1
        iou_data.append({
            "breed_name": breed,
            "mean_iou": mean_iou,
            "std_iou": std_iou,
            "count": len(breed_iou_dict[breed])
        })
    pd.DataFrame(iou_data).to_csv(out_dir / "iou_per_breed.csv", index=False)

    # Estatísticas descritivas das features
    stats_data = []
    features_array = np.array(features_list)
    # Primeiras 5 features são as estatísticas
    for breed in CAT_BREEDS:
        breed_mask = df["breed_name"] == breed
        breed_feats = features_array[breed_mask]
        if len(breed_feats) > 0:
            stats = {
                "breed_name": breed,
                "mean_mean": np.mean(breed_feats[:, 0]),
                "mean_std": np.mean(breed_feats[:, 1]),
                "mean_entropy": np.mean(breed_feats[:, 2]),
                "mean_skew": np.mean(breed_feats[:, 3]),
                "mean_kurtosis": np.mean(breed_feats[:, 4]),
                "count": len(breed_feats)
            }
            stats_data.append(stats)
    pd.DataFrame(stats_data).to_csv(out_dir / "descriptive_stats.csv", index=False)

    # Resumo das features
    with open(out_dir / "feature_summary.txt", "w") as f:
        f.write("Resumo das características:\n")
        f.write("- Estatísticas: 5 features (média, std, entropia, assimetria, curtose)\n")
        f.write("- Histograma HSV: 48 features (16 bins × 3 canais)\n")
        f.write("- Forma: 4 features (área, perímetro, compacidade, razão de aspecto)\n")
        f.write("- DCT: 10 features (coeficientes do bloco central)\n")
        f.write("- LBP: 32 features (histograma)\n")
        f.write(f"Total: {5 + 48 + 4 + 10 + 32} features\n")


def save_other_visuals(df, out_dir):
    """Salva visualizações adicionais (histogramas RGB médio, etc.)."""
    logger.info("Salvando visualizações adicionais...")
    # Placeholder para visualizações mais complexas (pode ser expandido)
    pass


if __name__ == "__main__":
    main()
