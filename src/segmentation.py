import numpy as np
import logging
import cv2
from pathlib import Path
from .preprocessing import to_grayscale

logger = logging.getLogger(__name__)


def otsu_threshold(image: np.ndarray) -> float:
    """
    Calcula limiar de Otsu manualmente.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem grayscale normalizada em [0, 1].

    Retorna
    -------
    float
        Limiar ótimo em [0, 1].
    """
    img_uint8 = (image * 255).astype(np.uint8)
    hist, _ = np.histogram(img_uint8.flatten(), bins=256, range=[0, 256])
    hist = hist / hist.sum()

    best_thresh = 0
    best_variance = 0

    for thresh in range(256):
        w0 = hist[:thresh].sum()
        w1 = hist[thresh:].sum()

        if w0 == 0 or w1 == 0:
            continue

        mean0 = np.sum(np.arange(thresh) * hist[:thresh]) / w0
        mean1 = np.sum(np.arange(thresh, 256) * hist[thresh:]) / w1

        variance_between = w0 * w1 * (mean0 - mean1)**2

        if variance_between > best_variance:
            best_variance = variance_between
            best_thresh = thresh

    return best_thresh / 255.0


def _erode(image: np.ndarray) -> np.ndarray:
    """Erosão binária manual com elemento estruturante em cruz."""
    # Elemento estruturante cruz (3x3)
    se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    H, W = image.shape
    pad = 1
    padded = np.pad(image, pad, mode="constant")
    output = np.zeros_like(image)

    for i in range(H):
        for j in range(W):
            region = padded[i:i+3, j:j+3]
            # Aplicar SE: verificar se todos os pixels do SE em região são 1
            masked = region * se
            output[i, j] = 1 if np.all(masked == se) else 0
    return output


def _dilate(image: np.ndarray) -> np.ndarray:
    """Dilatação binária manual com elemento estruturante em cruz."""
    # Elemento estruturante cruz (3x3)
    se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    H, W = image.shape
    pad = 1
    padded = np.pad(image, pad, mode="constant")
    output = np.zeros_like(image)

    for i in range(H):
        for j in range(W):
            region = padded[i:i+3, j:j+3]
            # Aplicar SE: verificar se pelo menos um pixel do SE em região é 1
            masked = region * se
            output[i, j] = 1 if np.any(masked == 1) else 0
    return output


def morphological_open_close(binary_mask: np.ndarray) -> np.ndarray:
    """
    Aplica abertura seguida de fechamento para limpar máscara.

    Parâmetros
    ----------
    binary_mask : np.ndarray
        Máscara binária, valores 0 ou 1.

    Retorna
    -------
    np.ndarray
        Máscara limpa.
    """
    opened = _dilate(_erode(binary_mask))
    closed = _erode(_dilate(opened))
    return closed


def calculate_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Calcula IoU (Intersection over Union) entre duas máscaras binárias.

    Parâmetros
    ----------
    pred_mask : np.ndarray
        Máscara predita (0 ou 1).
    true_mask : np.ndarray
        Máscara ground truth (0 ou 1).

    Retorna
    -------
    float
        Valor IoU entre 0 e 1.
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


def load_trimap(trimap_path: str) -> np.ndarray:
    """
    Carrega trimap do dataset e converte para máscara binária.

    Parâmetros
    ----------
    trimap_path : str
        Caminho para o arquivo trimap.png.

    Retorna
    -------
    np.ndarray
        Máscara binária (1 = foreground, 0 = background).
    """
    trimap_bgr = cv2.imread(trimap_path)
    if trimap_bgr is None:
        raise FileNotFoundError(f"Não foi possível carregar o trimap: {trimap_path}")

    trimap_gray = to_grayscale(trimap_bgr[:, :, ::-1])
    trimap = np.round(trimap_gray).astype(np.uint8)

    mask = np.zeros_like(trimap)
    mask[trimap == 1] = 1
    return mask.astype(np.float64)
