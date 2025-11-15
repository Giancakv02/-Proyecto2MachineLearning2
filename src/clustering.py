import numpy as np
from .config import (
    KMEANS_K, KMEANS_MAX_ITERS, KMEANS_TOL,
    DBSCAN_EPS, DBSCAN_MIN_SAMPLES
)

# ---------- helpers ----------
def _euclidean(a, b):
    # distancias fila-a-fila, pero aquí nos sirve para 1 punto vs N
    return np.sqrt(np.sum((a - b)**2, axis=1))

# ---------- K-MEANS ----------
def kmeans_fit_predict(X,
                       k=KMEANS_K,
                       max_iters=KMEANS_MAX_ITERS,
                       tol=KMEANS_TOL,
                       random_state=0):
    """
    Implementación básica de K-Means:
    1. inicializar centroides aleatorios de X
    2. asignar puntos al centroide más cercano
    3. recomputar centroides
    4. repetir hasta converger o max_iters
    Retorna: labels (n,), centroids (k,d)
    """
    rng = np.random.RandomState(random_state)
    n, d = X.shape

    # init: escoger k filas aleatorias
    idx0 = rng.choice(n, size=k, replace=False)
    centroids = X[idx0].copy()

    for it in range(max_iters):
        # asignación
        # distancias de cada punto a cada centroide
        # shape -> (n, k)
        dists = np.zeros((n, k), dtype=np.float32)
        for j in range(k):
            diff = X - centroids[j]
            dists[:, j] = np.sqrt(np.sum(diff * diff, axis=1))

        labels = np.argmin(dists, axis=1)

        # recomputar centroides
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            pts = X[labels == j]
            if len(pts) > 0:
                new_centroids[j] = np.mean(pts, axis=0)
            else:
                # si un cluster se queda vacío, re-seed aleatorio
                new_centroids[j] = X[rng.choice(n)]

        shift = np.sqrt(np.sum((centroids - new_centroids)**2))
        centroids = new_centroids

        if shift < tol:
            break

    return labels, centroids


# ---------- DBSCAN ----------
def dbscan_fit_predict(X, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    """
    Implementación DBSCAN básica.
    Etiquetas:
      -1 = ruido
      0..C-1 = id de cluster
    """
    n = X.shape[0]
    labels = np.full(n, -1, dtype=int)  # todo empieza como ruido
    visited = np.zeros(n, dtype=bool)

    cluster_id = 0

    # pre-computamos distancias NxN para acelerar vecinos
    # O(n^2), está bien para datasets medianos
    dist_matrix = _pairwise_distances(X)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = np.where(dist_matrix[i] <= eps)[0]

        if len(neighbors) < min_samples:
            # sigue como ruido
            continue

        # crear nuevo cluster
        labels[i] = cluster_id

        # expandir
        seeds = list(neighbors)
        seeds_idx = 0
        while seeds_idx < len(seeds):
            p = seeds[seeds_idx]
            if not visited[p]:
                visited[p] = True
                p_neighbors = np.where(dist_matrix[p] <= eps)[0]
                if len(p_neighbors) >= min_samples:
                    # anexar vecinos densos
                    for pn in p_neighbors:
                        if pn not in seeds:
                            seeds.append(pn)

            # asignar cluster si aún no tiene
            if labels[p] == -1:
                labels[p] = cluster_id

            seeds_idx += 1

        cluster_id += 1

    return labels

def _pairwise_distances(X):
    """
    Distancia euclidiana cuadrada expandida:
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    Luego sqrt.
    """
    # (n,d)
    norms = np.sum(X**2, axis=1, keepdims=True)  # (n,1)
    # dist^2 = norms + norms.T - 2 X X^T
    dist_sq = norms + norms.T - 2 * np.dot(X, X.T)
    # por estabilidad numérica: valores negativos muy pequeños -> 0
    dist_sq[dist_sq < 0] = 0
    return np.sqrt(dist_sq)