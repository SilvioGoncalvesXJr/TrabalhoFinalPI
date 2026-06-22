import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

CAT_BREEDS = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx"
]

BREED_TO_ID = {breed: idx for idx, breed in enumerate(CAT_BREEDS)}


def load_dataset(data_dir: str) -> pd.DataFrame:
    """
    Carrega e filtra o dataset Oxford-IIIT Pet, mantendo apenas gatos.

    Parâmetros
    ----------
    data_dir : str
        Caminho para a pasta raiz do dataset (ex: "data/oxford-iiit-pet/").

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas [image_path, breed_name, breed_id, split].
    """
    data_path = Path(data_dir)
    annotations_path = data_path / "annotations"
    images_path = data_path / "images"

    dfs = []

    # Processar treino e teste
    for split in ["trainval", "test"]:
        split_file = annotations_path / f"{split}.txt"
        if not split_file.exists():
            logger.error(f"Arquivo de split não encontrado: {split_file}")
            continue

        with open(split_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        split_name = "train" if split == "trainval" else "test"
        rows = []

        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue

            image_name = parts[0]
            species_id = int(parts[2])

            if species_id != 1:
                continue

            breed_name_parts = []
            for part in image_name.split("_"):
                if part.isdigit():
                    break
                breed_name_parts.append(part)
            breed_name = "_".join(breed_name_parts)

            if breed_name not in CAT_BREEDS:
                continue

            image_path = images_path / f"{image_name}.jpg"
            if not image_path.exists():
                logger.warning(f"Imagem não encontrada: {image_path}")
                continue

            rows.append({
                "image_path": str(image_path),
                "breed_name": breed_name,
                "breed_id": BREED_TO_ID[breed_name],
                "split": split_name
            })

        dfs.append(pd.DataFrame(rows))

    df = pd.concat(dfs, ignore_index=True)

    logger.info(f"Dataset carregado com {len(df)} imagens")
    logger.info("Distribuição por raça:")
    breed_counts = df["breed_name"].value_counts()
    for breed, count in breed_counts.items():
        logger.info(f"  {breed}: {count} imagens")
    logger.info("Distribuição treino/teste:")
    split_counts = df["split"].value_counts()
    for split, count in split_counts.items():
        logger.info(f"  {split}: {count} imagens")

    return df
