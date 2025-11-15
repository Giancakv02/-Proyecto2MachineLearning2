# generate_figures.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Configuración de estilo IEEE
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# ==== Carga de datos ====

# Cargamos el CSV alineado que sale del pipeline final (debe incluir 'cluster_kmeans')
df_train = pd.read_csv("outputs/train_meta_used.csv")

# Asegurar que exista la columna de cluster. Si no existe, creamos una dummy.
if "cluster_kmeans" not in df_train.columns:
    print("⚠️ 'cluster_kmeans' no existe en el CSV. Usaremos un cluster ficticio (0) para todas las películas.")
    df_train["cluster_kmeans"] = 0
else:
    print("✅ 'cluster_kmeans' encontrado en el CSV.")

# Intentamos cargar la proyección 2D precomputada si existe
coords2d = None
try:
    coords2d = np.load("data/COORDS_2D.npy")
    print("✅ Cargado COORDS_2D.npy")
except FileNotFoundError:
    print("⚠️ data/COORDS_2D.npy no existe. Usando las dos primeras PCs como proyección 2D.")
    # fallback: usamos las primeras 2 columnas del embedding PCA de 50 dims
    # Este archivo lo genera tu pipeline como FEATURES_PCA.npy
    features_pca = np.load("data/FEATURES_PCA.npy")
    # features_pca tiene forma (N, 50). Nos quedamos con PC1 y PC2.
    coords2d = features_pca[:, :2]

# ==== 1. Proyección PCA ====
x = coords2d[:, 0]
y = coords2d[:, 1]
labels = df_train["cluster_kmeans"].astype(int)

plt.figure(figsize=(6, 4.5))
scatter = plt.scatter(x, y, c=labels, cmap='tab10', s=12, alpha=0.8, edgecolors='none')
plt.title("Proyección PCA 2D de los pósters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
plt.tight_layout()
plt.savefig("figures/pca_map.png")
plt.close()

# ==== 2. Distribución de Clusters ====
counts = df_train["cluster_kmeans"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(counts.index, counts.values, color="#4b2db8", edgecolor="black", alpha=0.85)
plt.title("Distribución del tamaño de los clusters (K-Means)")
plt.xlabel("Cluster ID")
plt.ylabel("Número de películas")
plt.xticks(counts.index)
plt.grid(axis='y', linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/clusters_distribution.png")
plt.close()

# ==== 3. Interfaz Streamlit (manual) ====
# Captura tu pantalla como screenshot_streamlit.png en raíz del proyecto
try:
    img = Image.open("screenshot_streamlit.png")
    canvas = Image.new("RGB", img.size, "white")
    canvas.paste(img, mask=None)
    canvas.save("figures/ui_streamlit.png", "PNG")
    print("✅ Imagen de interfaz guardada como ui_streamlit.png")
except FileNotFoundError:
    print("⚠️ No se encontró screenshot_streamlit.png. Captura manualmente la app Streamlit.")