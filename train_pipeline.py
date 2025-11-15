import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    OUTPUT_DIR,
    FEATURES_RAW, FEATURES_PCA, FEATURES_SVD,
    LABELS_KMEANS, LABELS_DBSCAN,
    COORDS_2D,
    EVAL_REPORT,
    N_COMPONENTS,
    KMEANS_K,
    PCA_MEAN_NPY,
    PCA_COMPONENTS_NPY,
    TRAIN_META_FILTERED_CSV,
    TRAIN_WITH_POSTERS
)
from src.data_loader import load_metadata, filter_existing_posters
from src.feature_extractor import extract_features_for_image
from src.dimensionality import pca_fit_transform, svd_fit_transform
from src.clustering import kmeans_fit_predict, dbscan_fit_predict
from src.evaluation import build_eval_report
from src.utils import save_npy, save_json, project_to_2d


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1] Cargando metadatos...")
    # IMPORTANTE: ahora entrenamos con movies_train_with_posters.csv
    df = load_metadata(TRAIN_WITH_POSTERS)
    df = filter_existing_posters(df)
    print(f"   Películas con póster válido: {len(df)}")

    # 2. Extraer features visuales por póster
    feats = []
    valid_idx = []

    print("[2] Extrayendo features...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        vec = extract_features_for_image(row["poster_path"])
        if vec is not None:
            feats.append(vec)
            valid_idx.append(idx)

    if len(feats) == 0:
        print("[ERROR] No se pudo extraer NINGÚN póster válido.")
        print("Revisa que existan imágenes en data/posters/ con nombres que coincidan con movieId.jpg")
        return

    X = np.vstack(feats)
    # alineamos df con las filas válidas (en caso de que alguna imagen haya fallado)
    df = df.iloc[valid_idx].reset_index(drop=True)
    print(f"   Shape features crudas: {X.shape}")
    save_npy(FEATURES_RAW, X)

    # 3. Reducción PCA (propia)
    print("[3] PCA manual...")
    X_pca, pca_mean, pca_components, eigvals = pca_fit_transform(X, n_components=N_COMPONENTS)
    print("   PCA shape:", X_pca.shape)
    save_npy(FEATURES_PCA, X_pca)
    # 🔽 Extra: guardamos también una copia simple del embedding PCA para futuras visualizaciones
    np.save("data/FEATURES_PCA.npy", X_pca)
    print("✅ Archivo FEATURES_PCA.npy guardado en carpeta data/")

    # Guardamos parámetros del PCA para usarlos luego en test
    save_npy(PCA_MEAN_NPY, pca_mean)
    save_npy(PCA_COMPONENTS_NPY, pca_components)

    # 4. Reducción SVD (propia) - opcional análisis
    print("[4] SVD manual...")
    X_svd, svd_mean, svd_components, singvals = svd_fit_transform(X, n_components=N_COMPONENTS)
    print("   SVD shape:", X_svd.shape)
    save_npy(FEATURES_SVD, X_svd)

    # Elegimos PCA reducido para clustering y recomendación
    X_red = X_pca

    # 5. Clustering K-Means (propio)
    print("[5] K-Means manual...")
    labels_km, centroids = kmeans_fit_predict(X_red)
    save_npy(LABELS_KMEANS, labels_km)

    # 6. Clustering DBSCAN (propio)
    print("[6] DBSCAN manual...")
    labels_db = dbscan_fit_predict(X_red)
    save_npy(LABELS_DBSCAN, labels_db)

    # 7. Proyección 2D para visualización en la app
    print("[7] Proyección 2D con PCA(2)...")
    coords2d = project_to_2d(X_red)
    save_npy(COORDS_2D, coords2d)

    # 8. Guardar el dataframe alineado al embedding
    #    Esto es muy importante para poder mapear movieId -> vector PCA después
    df_out = df.copy()
    df_out["cluster_kmeans"] = labels_km
    df_out.to_csv(TRAIN_META_FILTERED_CSV, index=False)

    # 9. Evaluación interna y coherencia de géneros
    print("[8] Evaluación...")
    report = {
        "n_movies": int(len(df)),
        "feature_dim_raw": int(X.shape[1]),
        "feature_dim_reduced": int(X_red.shape[1]),
        "pca_components_used": int(N_COMPONENTS),
        "kmeans_k": int(KMEANS_K),
    }
    from src.evaluation import build_eval_report
    full_eval = build_eval_report(X_red, df, labels_km, labels_db)
    report.update(full_eval)

    save_json(EVAL_REPORT, report)

    print(f"[Listo ✅] Reporte en: {EVAL_REPORT}")
    print(f"[Listo ✅] Metadata alineada guardada en: {TRAIN_META_FILTERED_CSV}")
    print(f"[Listo ✅] Parámetros PCA guardados en: {PCA_MEAN_NPY} y {PCA_COMPONENTS_NPY}")


if __name__ == "__main__":
    main()