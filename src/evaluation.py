import numpy as np
import pandas as pd

def _euclidean_matrix(X):
    """
    Devuelve matriz de distancias NxN.
    """
    norms = np.sum(X**2, axis=1, keepdims=True)
    dist_sq = norms + norms.T - 2 * np.dot(X, X.T)
    dist_sq[dist_sq < 0] = 0
    return np.sqrt(dist_sq)

def silhouette_score_manual(X, labels):
    """
    Calcula silhouette promedio.
    Para cada punto i:
      a_i = promedio dist a puntos mismo cluster
      b_i = menor promedio dist a otro cluster
      s_i = (b_i - a_i) / max(a_i, b_i)
    Retorna promedio de s_i (ignora clusters de tamaño 1).
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    n = X.shape[0]
    unique_clusters = np.unique(labels)

    # si todos están en el mismo cluster o todos -1, silhouette no tiene sentido
    if len(unique_clusters) < 2:
        return None

    D = _euclidean_matrix(X)  # NxN

    silhouettes = []
    for i in range(n):
        li = labels[i]
        same_cluster = np.where(labels == li)[0]
        if len(same_cluster) <= 1:
            # silhouette indefinido para clúster singleton
            continue

        # a_i
        same_cluster_wo_i = same_cluster[same_cluster != i]
        if len(same_cluster_wo_i) == 0:
            continue
        a_i = np.mean(D[i, same_cluster_wo_i])

        # b_i
        b_i_candidates = []
        for other in unique_clusters:
            if other == li:
                continue
            idx_other = np.where(labels == other)[0]
            if len(idx_other) > 0:
                b_i_candidates.append(np.mean(D[i, idx_other]))
        if len(b_i_candidates) == 0:
            continue
        b_i = np.min(b_i_candidates)

        denom = max(a_i, b_i)
        if denom == 0:
            s_i = 0.0
        else:
            s_i = (b_i - a_i) / denom

        silhouettes.append(s_i)

    if len(silhouettes) == 0:
        return None
    return float(np.mean(silhouettes))

def cluster_genre_stats(df_meta, labels):
    """
    Calcula stats de coherencia de género por cluster.
    df_meta necesita columnas 'genres' y 'title'.
    labels es array (n,)
    Retorna dict:
      cluster_id: {
         "size": ...,
         "top_genre": ...,
         "genre_purity": ...,
         "sample_titles": [...]
      }
    """
    out = {}
    tmp = df_meta.copy()
    tmp["cluster"] = labels

    for cl in sorted(tmp["cluster"].unique()):
        sub = tmp[tmp["cluster"] == cl]
        genres_list = []
        for g in sub["genres"]:
            if isinstance(g, str):
                genres_list += g.split("|")

        if len(genres_list) == 0:
            top_genre = None
            purity = None
        else:
            counts = pd.Series(genres_list).value_counts()
            top_genre = counts.index[0]
            purity = counts.iloc[0] / len(sub)

        out[int(cl)] = {
            "size": int(len(sub)),
            "top_genre": top_genre,
            "genre_purity": float(purity) if purity is not None else None,
            "sample_titles": sub["title"].head(5).tolist()
        }
    return out

def build_eval_report(X_reduced, df_meta, labels_kmeans, labels_dbscan):
    """
    Junta métricas para el JSON final.
    """
    sil_km = silhouette_score_manual(X_reduced, labels_kmeans)
    sil_db = silhouette_score_manual(X_reduced, labels_dbscan)

    genre_stats_km = cluster_genre_stats(df_meta, labels_kmeans)
    genre_stats_db = cluster_genre_stats(df_meta, labels_dbscan)

    report = {
        "metrics": {
            "kmeans": {
                "silhouette": sil_km
            },
            "dbscan": {
                "silhouette": sil_db
            }
        },
        "genre_stats": {
            "kmeans": genre_stats_km,
            "dbscan": genre_stats_db
        }
    }
    return report