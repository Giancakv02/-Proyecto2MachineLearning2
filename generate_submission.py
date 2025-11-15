import os
import numpy as np
import pandas as pd

from src.config import (
    FEATURES_PCA,
    PCA_MEAN_NPY,
    PCA_COMPONENTS_NPY,
    TRAIN_META_FILTERED_CSV,
    TEST_WITH_POSTERS,
    SUBMISSION_CSV
)
from src.feature_extractor import extract_features_for_image
from src.dimensionality import pca_transform

TOP_N = 10  # cuántas recomendaciones vas a dar por cada query


def load_artifacts():
    # embeddings de TRAIN ya reducidos con PCA
    X_train = np.load(FEATURES_PCA)  # shape (n_train, k)
    # parámetros del PCA entrenado en train
    pca_mean = np.load(PCA_MEAN_NPY)  # shape (d_original,)
    pca_components = np.load(PCA_COMPONENTS_NPY)  # shape (d_original, k)
    # metadata alineada con X_train (misma orden de filas)
    df_train = pd.read_csv(TRAIN_META_FILTERED_CSV)
    # test enriquecido con poster_path
    df_test = pd.read_csv(TEST_WITH_POSTERS)

    return X_train, pca_mean, pca_components, df_train, df_test


def embed_image_pca(poster_path, pca_mean, pca_components):
    """
    1. Extrae features visuales crudas de la imagen del test (igual que en train).
    2. Proyecta al espacio PCA aprendido (mismas componentes).
    Retorna un vector (1, k) o None si la imagen falla.
    """
    vec = extract_features_for_image(poster_path)
    if vec is None:
        return None
    vec = vec.reshape(1, -1)  # (1, d_original)
    x_proj = pca_transform(vec, pca_mean, pca_components)  # (1, k)
    return x_proj


def top_n_neighbors(x_query, X_train, df_train, n=TOP_N):
    """
    Calcula distancias euclidianas entre la query y todas las pelis de train.
    Devuelve la lista de movieId más cercanos.
    """
    diff = X_train - x_query  # (N, k)
    dists = np.sqrt(np.sum(diff * diff, axis=1))  # (N,)
    order = np.argsort(dists)  # índices ordenados por menor distancia
    movie_ids_ranked = df_train.iloc[order]["movieId"].tolist()
    return movie_ids_ranked[:n]


def main():
    print("[1] Cargando artefactos de entrenamiento y test...")
    X_train, pca_mean, pca_components, df_train, df_test = load_artifacts()
    print(f"   Train embeddings: {X_train.shape}")
    print(f"   Train meta usado: {df_train.shape}")
    print(f"   Test con poster:  {df_test.shape}")

    rows = []

    print("[2] Generando recomendaciones visuales...")
    for i, row in df_test.iterrows():
        q_id = row["movieId"]
        poster_path = row["poster_path"]

        # embed de la película query (test) al mismo espacio PCA
        x_q = embed_image_pca(poster_path, pca_mean, pca_components)
        if x_q is None:
            # si no pudimos extraer features de esa imagen, la saltamos
            continue

        recs = top_n_neighbors(x_q, X_train, df_train, n=TOP_N)

        # armamos las filas en el formato exactamente pedido
        # query_movie_id,recommended_movie_id,position
        for rank, rec_id in enumerate(recs, start=1):
            rows.append({
                "query_movie_id": q_id,
                "recommended_movie_id": rec_id,
                "position": rank
            })

    submission_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(SUBMISSION_CSV), exist_ok=True)
    submission_df.to_csv(SUBMISSION_CSV, index=False)

    print(f"[✔] Submission listo en {SUBMISSION_CSV}")
    print(submission_df.head(20))


if __name__ == "__main__":
    main()