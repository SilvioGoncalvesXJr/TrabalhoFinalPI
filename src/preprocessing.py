import numpy as np
import logging

logger = logging.getLogger(__name__)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converte uma imagem RGB para escala de cinza usando pesos ITU-R BT.601.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem RGB, shape (H, W, 3), dtype uint8 ou float.

    Retorna
    -------
    np.ndarray
        Imagem em escala de cinza, shape (H, W), dtype float64, valores [0, 255].
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Imagem deve ser RGB com 3 canais")

    img_float = image.astype(np.float64)
    r = img_float[:, :, 0]
    g = img_float[:, :, 1]
    b = img_float[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255)


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normaliza uma imagem para o intervalo [0.0, 1.0].

    Parâmetros
    ----------
    image : np.ndarray
        Imagem de entrada, valores em [0, 255].

    Retorna
    -------
    np.ndarray
        Imagem normalizada, dtype float64, valores [0, 1].
    """
    return image.astype(np.float64) / 255.0


def _create_gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Cria kernel gaussiano 2D."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    center = kernel_size // 2
    sum_kernel = 0.0

    for x in range(kernel_size):
        for y in range(kernel_size):
            dx = x - center
            dy = y - center
            value = np.exp(-(dx**2 + dy**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
            kernel[x, y] = value
            sum_kernel += value

    return kernel / sum_kernel


def _convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Aplica convolução 2D manualmente."""
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    output = np.zeros_like(image, dtype=np.float64)

    for i in range(H):
        for j in range(W):
            region = padded[i:i + kH, j:j + kW]
            output[i, j] = np.sum(region * kernel)

    return output


def gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Aplica filtro gaussiano manualmente via convolução 2D com NumPy.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem grayscale normalizada em [0, 1], shape (H, W), dtype float64.
    sigma : float
        Desvio padrão do kernel gaussiano. Padrão: 1.0.

    Retorna
    -------
    np.ndarray
        Imagem filtrada, mesmo shape e dtype da entrada.
    """
    kernel = _create_gaussian_kernel(5, sigma)
    return _convolve_2d(image, kernel)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Aplica equalização de histograma manualmente em uma imagem grayscale.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem grayscale em [0, 1], shape (H, W).

    Retorna
    -------
    np.ndarray
        Imagem equalizada em [0, 1], mesmo shape da entrada.
    """
    img_uint8 = (image * 255).astype(np.uint8)
    hist, _ = np.histogram(img_uint8.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min())
    img_equalized = cdf_normalized[img_uint8]
    return img_equalized.astype(np.float64)
