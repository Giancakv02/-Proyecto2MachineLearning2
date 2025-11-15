import numpy as np
from .config import N_COMPONENTS

def pca_fit_transform(X, n_components=N_COMPONENTS):
    """
    PCA manual:
    1. centrar
    2. cov = X^T X / (n-1)
    3. autovalores/vectores
    4. ordenar por energía
    5. proyectar
    Retorna: X_proj, mean_vec, components (matriz de autovectores)
    """
    # centramos
    mean_vec = np.mean(X, axis=0, keepdims=True)
    Xc = X - mean_vec

    # covarianza (dxd)
    cov = np.dot(Xc.T, Xc) / (Xc.shape[0] - 1)

    # autovalores / autovectores
    vals, vecs = np.linalg.eigh(cov)  # eigh porque cov es simétrica

    # ordenar de mayor a menor autovalor
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # truncar
    k = min(n_components, vecs.shape[1])
    components = vecs[:, :k]              # d x k
    X_proj = np.dot(Xc, components)       # n x k

    return X_proj, mean_vec, components, vals[:k]

def pca_transform(X, mean_vec, components):
    """
    Proyecta nuevos datos usando PCA ya entrenado.
    """
    Xc = X - mean_vec
    return np.dot(Xc, components)

def svd_fit_transform(X, n_components=N_COMPONENTS):
    """
    SVD manual (usando numpy.linalg.svd):
    1. centrar
    2. SVD de Xc = U S V^T
    3. tomar primeras k columnas de V
    4. X_proj = Xc @ V_k
    Retorna: X_proj, mean_vec, V_k, singular_vals[:k]
    """
    mean_vec = np.mean(X, axis=0, keepdims=True)
    Xc = X - mean_vec

    # SVD completa
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Vt: k x d  (filas son componentes principales tipo PCA también)
    k = min(n_components, Vt.shape[0])
    V_k = Vt[:k, :].T           # d x k
    X_proj = np.dot(Xc, V_k)    # n x k

    return X_proj, mean_vec, V_k, S[:k]

def pca_2d_projection(X):
    """
    PCA for visualization ONLY (2D).
    Siempre devuelve n x 2.
    """
    X2d, _, _, _ = pca_fit_transform(X, n_components=2)
    return X2d