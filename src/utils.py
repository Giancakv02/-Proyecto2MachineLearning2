import os
import json
import numpy as np
from .dimensionality import pca_2d_projection

def save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def load_npy(path):
    return np.load(path)

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def euclidean_distances_from_idx(X, idx):
    """
    Devuelve distancias euclidianas del punto idx al resto.
    """
    x0 = X[idx]
    diff = X - x0
    d = np.sqrt(np.sum(diff * diff, axis=1))
    return d

def project_to_2d(X):
    """
    Para visualización en app: PCA a 2D (hecho a mano).
    """
    return pca_2d_projection(X)