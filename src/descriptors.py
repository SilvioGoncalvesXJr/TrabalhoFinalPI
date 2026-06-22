import numpy as np
import logging
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)


def _rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """Converte RGB para HSV manualmente."""
    img_norm = image.astype(np.float64) / 255.0
    r, g, b = img_norm[:, :, 0], img_norm[:, :, 1], img_norm[:, :, 2]

    max_c = np.maximum.reduce([r, g, b])
    min_c = np.minimum.reduce([r, g, b])
    delta = max_c - min_c

    h = np.zeros_like(r)
    s = np.zeros_like(r)
    v = max_c

    # Hue
    mask = delta != 0
    r_mask = r[mask]
    g_mask = g[mask]
    b_mask = b[mask]
    max_mask = max_c[mask]
    delta_mask = delta[mask]

    h[mask] = np.where(max_mask == r_mask,
                       ((g_mask - b_mask) / delta_mask) % 6,
                       h[mask])
    h[mask] = np.where(max_mask == g_mask,
                       ((b_mask - r_mask) / delta_mask) + 2,
                       h[mask])
    h[mask] = np.where(max_mask == b_mask,
                       ((r_mask - g_mask) / delta_mask) + 4,
                       h[mask])
    h = (h * 60) % 360

    # Saturation
    s[max_c != 0] = delta[max_c != 0] / max_c[max_c != 0]

    return np.stack([h, s, v], axis=-1)


def extract_statistical_features(image: np.ndarray) -> np.ndarray:
    """
    Extrai características estatísticas: média, desvio padrão, entropia, assimetria, curtose.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem grayscale normalizada em [0, 1].

    Retorna
    -------
    np.ndarray
        Vetor de 5 características.
    """
    mean_val = np.mean(image)
    std_val = np.std(image)
    img_uint8 = (image * 255).astype(np.uint8)
    hist, _ = np.histogram(img_uint8.flatten(), bins=256, range=[0, 256], density=True)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    skewness_val = skew(image.flatten())
    kurtosis_val = kurtosis(image.flatten())
    return np.array([mean_val, std_val, entropy, skewness_val, kurtosis_val], dtype=np.float64)


def extract_hsv_histogram(image: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Extrai histograma HSV manualmente.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem RGB, shape (H, W, 3).
    bins : int
        Número de bins por canal. Padrão: 16.

    Retorna
    -------
    np.ndarray
        Vetor de 3*bins características normalizadas.
    """
    hsv = _rgb_to_hsv(image)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    h_bins = np.linspace(0, 360, bins + 1)
    s_bins = np.linspace(0, 1, bins + 1)
    v_bins = np.linspace(0, 1, bins + 1)

    h_hist, _ = np.histogram(h.flatten(), bins=h_bins)
    s_hist, _ = np.histogram(s.flatten(), bins=s_bins)
    v_hist, _ = np.histogram(v.flatten(), bins=v_bins)

    h_hist = h_hist / (h_hist.sum() + 1e-10)
    s_hist = s_hist / (s_hist.sum() + 1e-10)
    v_hist = v_hist / (v_hist.sum() + 1e-10)

    return np.concatenate([h_hist, s_hist, v_hist])


def extract_shape_features(mask: np.ndarray) -> np.ndarray:
    """
    Extrai características de forma: área, perímetro, compacidade, razão de aspecto.

    Parâmetros
    ----------
    mask : np.ndarray
        Máscara binária (0 ou 1).

    Retorna
    -------
    np.ndarray
        Vetor de 4 características.
    """
    area = mask.sum()

    # Perímetro
    padded = np.pad(mask, 1)
    perimeter = 0
    H, W = mask.shape
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1:
                neighbors = padded[i:i+3, j:j+3]
                if (neighbors == 0).any():
                    perimeter += 1

    # Bounding box
    coords = np.argwhere(mask == 1)
    if len(coords) == 0:
        return np.zeros(4, dtype=np.float64)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    aspect_ratio = width / height if height != 0 else 0

    # Compacidade
    compactness = (perimeter**2) / (4 * np.pi * area) if area != 0 else 0

    return np.array([area, perimeter, compactness, aspect_ratio], dtype=np.float64)


def extract_dct_features(dct_blocks: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Extrai coeficientes DCT do bloco central.

    Parâmetros
    ----------
    dct_blocks : np.ndarray
        Blocos DCT de transforms.dct_blocks.
    k : int
        Número de coeficientes a extrair. Padrão: 10.

    Retorna
    -------
    np.ndarray
        Vetor de k características normalizadas.
    """
    blocks_h, blocks_w = dct_blocks.shape[:2]
    center_i = blocks_h // 2
    center_j = blocks_w // 2
    center_block = dct_blocks[center_i, center_j]

    dc = center_block[0, 0] if center_block[0, 0] != 0 else 1.0
    flat = center_block.flatten()
    normalized = flat[:k] / dc
    return normalized


def extract_lbp_histogram(image: np.ndarray, num_bins: int = 32) -> np.ndarray:
    """
    Extrai histograma LBP (Local Binary Pattern) manualmente.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem grayscale normalizada em [0, 1].
    num_bins : int
        Número de bins do histograma. Padrão: 32.

    Retorna
    -------
    np.ndarray
        Histograma LBP normalizado.
    """
    img_uint8 = (image * 255).astype(np.uint8)
    H, W = img_uint8.shape
    lbp = np.zeros_like(img_uint8)

    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, 1), (1, 1), (1, 0),
               (1, -1), (0, -1)]

    for i in range(1, H-1):
        for j in range(1, W-1):
            center = img_uint8[i, j]
            val = 0
            for idx, (di, dj) in enumerate(offsets):
                neighbor = img_uint8[i+di, j+dj]
                if neighbor >= center:
                    val |= (1 << idx)
            lbp[i, j] = val

    hist, _ = np.histogram(lbp.flatten(), bins=np.arange(257))
    hist = hist[:num_bins]
    hist = hist / (hist.sum() + 1e-10)
    return hist


def extract_all(gray_image: np.ndarray, rgb_image: np.ndarray,
                binary_mask: np.ndarray, dct_blocks: np.ndarray) -> np.ndarray:
    """
    Extrai todas as características e concatena em um vetor único.

    Parâmetros
    ----------
    gray_image : np.ndarray
        Imagem grayscale equalizada.
    rgb_image : np.ndarray
        Imagem RGB redimensionada.
    binary_mask : np.ndarray
        Máscara binária limpa.
    dct_blocks : np.ndarray
        Blocos DCT.

    Retorna
    -------
    np.ndarray
        Vetor de 99 características.
    """
    feats_stat = extract_statistical_features(gray_image)
    feats_hsv = extract_hsv_histogram(rgb_image)
    feats_shape = extract_shape_features(binary_mask)
    feats_dct = extract_dct_features(dct_blocks)
    feats_lbp = extract_lbp_histogram(gray_image)

    return np.concatenate([feats_stat, feats_hsv, feats_shape, feats_dct, feats_lbp])
