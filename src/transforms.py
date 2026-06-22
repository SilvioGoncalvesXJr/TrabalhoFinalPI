import numpy as np
import logging
from .preprocessing import _convolve_2d

logger = logging.getLogger(__name__)


def sobel_filter(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica filtro Sobel manualmente para detecção de bordas.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem grayscale normalizada em [0, 1].

    Retorna
    -------
    tuple[np.ndarray, np.ndarray]
        (magnitude, direção), ambas shape (H, W), dtype float64.
    """
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    grad_x = _convolve_2d(image, Gx)
    grad_y = _convolve_2d(image, Gy)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude
    direction = np.arctan2(grad_y, grad_x)

    return magnitude, direction


def log_filter(image: np.ndarray) -> np.ndarray:
    """
    Aplica filtro Laplaciano do Gaussiano (LoG) manualmente.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem grayscale normalizada em [0, 1].

    Retorna
    -------
    np.ndarray
        Imagem com bordas LoG, shape (H, W), valores clipados em [0, 1].
    """
    LoG = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0]
    ], dtype=np.float64)

    filtered = _convolve_2d(image, LoG)
    filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min())
    return np.clip(filtered, 0, 1)


def dct_blocks(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Aplica DCT-II em blocos 8×8 sobre a imagem grayscale.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem grayscale normalizada em [0, 1], shape (H, W).
    block_size : int
        Tamanho do bloco para DCT. Padrão: 8.

    Retorna
    -------
    np.ndarray
        Blocos DCT, shape (H//block_size, W//block_size, block_size, block_size).
    """
    H, W = image.shape
    blocks_h = H // block_size
    blocks_w = W // block_size

    dct_result = np.zeros((blocks_h, blocks_w, block_size, block_size), dtype=np.float64)

    alpha = np.zeros(block_size)
    alpha[0] = 1.0 / np.sqrt(block_size)
    alpha[1:] = np.sqrt(2.0 / block_size)

    for i in range(blocks_h):
        for j in range(blocks_w):
            block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            for u in range(block_size):
                for v in range(block_size):
                    sum_val = 0.0
                    for x in range(block_size):
                        for y in range(block_size):
                            cos_u = np.cos((2*x + 1)*u*np.pi / (2*block_size))
                            cos_v = np.cos((2*y + 1)*v*np.pi / (2*block_size))
                            sum_val += block[x, y] * cos_u * cos_v
                    dct_result[i, j, u, v] = alpha[u] * alpha[v] * sum_val

    return dct_result
