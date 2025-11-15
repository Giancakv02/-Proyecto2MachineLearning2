import os
import pandas as pd
from .config import TRAIN_META, POSTER_DIR

def load_metadata(csv_path: str = TRAIN_META):
    """
    CSV esperado:
    movieId,title,year,genres,poster_path (opcional)

    Si 'poster_path' no está en el CSV, asumimos automáticamente:
        data/posters/{movieId}.jpg
    """
    df = pd.read_csv(csv_path)

    # Si el CSV no trae una columna 'poster_path', la creamos
    if "poster_path" not in df.columns:
        df["poster_path"] = df["movieId"].apply(
            lambda mid: os.path.join(POSTER_DIR, f"{mid}.jpg")
        )

    return df

def filter_existing_posters(df):
    """
    Filtra solo las filas cuya imagen existe físicamente en disco.
    """
    df = df.copy()
    df = df[df["poster_path"].apply(lambda p: os.path.exists(p))]
    df = df.reset_index(drop=True)
    return df