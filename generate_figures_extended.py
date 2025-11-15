# eda_extra_figs.py  (o al final de generate_figures_extended.py)
import os, re, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, TRAIN_META_FILTERED_CSV
from src.feature_extractor import extract_features_for_image

RAW_FEATS_NPY = os.path.join(OUTPUT_DIR, "features_raw.npy")

# --- helpers ---------------------------------------------------------------
def _nice_42_names():
    hue = [f"H{i}" for i in range(32)]
    rest = ["LBP_mean", "GLCM_contrast", "GLCM_homogeneity"] + [f"Hu{i+1}" for i in range(7)]
    return hue + rest  # 32 + 1 + 2 + 7 = 42

def _standardize(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + 1e-8)

# --- Figure 1: feature correlation heatmap --------------------------------
def figure_feature_corr_heatmap(max_feats=256, seed=0):
    assert os.path.exists(RAW_FEATS_NPY), f"No existe {RAW_FEATS_NPY}"
    X = np.load(RAW_FEATS_NPY)  # (N, d)
    d = X.shape[1]
    print(f"[Info] d = {d} features")

    # sample columns if too many
    rng = np.random.default_rng(seed)
    cols = np.arange(d) if d <= max_feats else rng.choice(d, size=max_feats, replace=False)
    Xsub = X[:, cols]
    Xz = _standardize(Xsub)
    C = np.corrcoef(Xz, rowvar=False)

    # names
    if d == 42:
        base = _nice_42_names()
        names = [base[i] for i in cols]
    else:
        names = [f"f{c}" for c in cols]

    plt.figure(figsize=(9, 7), dpi=200)
    im = plt.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="corr")

    # Si hay muchas columnas, no saturar con ticks
    if len(cols) <= 60:
        plt.xticks(np.arange(len(cols)), names, rotation=90, fontsize=7)
        plt.yticks(np.arange(len(cols)), names, fontsize=7)
    else:
        plt.xticks([]); plt.yticks([])

    plt.title("Correlation heatmap (standardized features)")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "feature_corr_heatmap.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[OK] {out}")

# --- Figure 2: LBP mean vs GLCM contrast ----------------------------------
def figure_lbp_glcm_scatter(sample_size=600, seed=0):
    rng = np.random.default_rng(seed)

    # Intento 1: si el vector es 42-D, usar índices directos
    try:
        X = np.load(RAW_FEATS_NPY)
        if X.shape[1] >= 34 and X.shape[1] == 42:
            lbp = X[:, 32]
            glcm_contrast = X[:, 33]
        else:
            raise ValueError("No es 42-D, paso a extraer en una muestra.")
    except Exception:
        # Intento 2: re-extraer features (42-D) en una muestra de pósters
        df = pd.read_csv(TRAIN_META_FILTERED_CSV)
        paths = df["poster_path"].dropna().tolist()
        if len(paths) == 0:
            raise RuntimeError("No hay poster_path en el CSV.")
        paths = rng.choice(paths, size=min(sample_size, len(paths)), replace=False)

        lbp_vals, glcm_vals = [], []
        for p in paths:
            if not os.path.exists(p): 
                continue
            v = extract_features_for_image(p)  # debe retornar 42-D
            if v is None or len(v) < 34: 
                continue
            lbp_vals.append(v[32])
            glcm_vals.append(v[33])

        if len(lbp_vals) < 10:
            raise RuntimeError("Muy pocos puntos para el scatter LBP vs GLCM.")
        lbp = np.array(lbp_vals); glcm_contrast = np.array(glcm_vals)

    plt.figure(figsize=(7, 6), dpi=200)
    hb = plt.hexbin(lbp, glcm_contrast, gridsize=40, bins='log', mincnt=1)
    plt.colorbar(hb, label="log(count)")
    plt.xlabel("LBP mean")
    plt.ylabel("GLCM contrast (d=1, θ=0°)")
    plt.title("LBP mean vs. GLCM contrast")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "lbp_glcm_scatter.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[OK] {out}")

# --- Figure 3: Posters by decade ------------------------------------------
def figure_decade_hist():
    df = pd.read_csv(TRAIN_META_FILTERED_CSV)
    years = []
    for t in df["title"].astype(str):
        m = re.search(r"\((\d{4})\)\s*$", t)
        years.append(int(m.group(1)) if m else np.nan)
    s = pd.Series(years).dropna().astype(int)
    decades = (s // 10) * 10
    counts = decades.value_counts().sort_index()

    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(counts.index.astype(int).astype(str), counts.values)
    plt.xlabel("Decade"); plt.ylabel("Poster count")
    plt.title("Posters by release decade")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "decade_hist.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[OK] {out}")

def generate_extra_eda():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figure_feature_corr_heatmap(max_feats=256)  # seguro y legible con 8210-D
    figure_lbp_glcm_scatter(sample_size=600)
    figure_decade_hist()
    print("[OK] Extra EDA figures saved in outputs")

if __name__ == "__main__":
    generate_extra_eda()