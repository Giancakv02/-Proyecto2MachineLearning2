import os

# === RUTAS BASE ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
POSTER_DIR = os.path.join(DATA_DIR, "posters")

TRAIN_META = os.path.join(DATA_DIR, "movies_train.csv")
TEST_META = os.path.join(DATA_DIR, "movies_test.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
APP_ARTIFACTS_DIR = os.path.join(BASE_DIR, "app_artifacts")

FEATURES_RAW = os.path.join(OUTPUT_DIR, "features_raw.npy")
FEATURES_PCA = os.path.join(OUTPUT_DIR, "features_reduced_pca.npy")
FEATURES_SVD = os.path.join(OUTPUT_DIR, "features_reduced_svd.npy")

LABELS_KMEANS = os.path.join(OUTPUT_DIR, "clustering_labels_kmeans.npy")
LABELS_DBSCAN = os.path.join(OUTPUT_DIR, "clustering_labels_dbscan.npy")

COORDS_2D = os.path.join(OUTPUT_DIR, "coords2d_pca.npy")

EVAL_REPORT = os.path.join(OUTPUT_DIR, "eval_report.json")

# === PARÁMETROS DE FEATURES ===
IMG_SIZE = (256, 256)  # resize de cada póster antes de extraer features
HIST_BINS = 32         # bins por canal HSV

# === REDUCCIÓN DE DIMENSIONALIDAD ===
N_COMPONENTS = 50      # dimensión objetivo para PCA / SVD

# === CLUSTERING ===
KMEANS_K = 12          # num clusters para K-Means
KMEANS_MAX_ITERS = 100
KMEANS_TOL = 1e-4

DBSCAN_EPS = 0.8       # radio de vecindad
DBSCAN_MIN_SAMPLES = 5
# === MODELO / ARTEFACTOS ===
PCA_MEAN_NPY = os.path.join(OUTPUT_DIR, "pca_mean.npy")
PCA_COMPONENTS_NPY = os.path.join(OUTPUT_DIR, "pca_components.npy")
TRAIN_META_FILTERED_CSV = os.path.join(OUTPUT_DIR, "train_meta_used.csv")
FEATURES_PCA_FALLBACK = os.path.join(APP_ARTIFACTS_DIR, "features_reduced_pca.npy")
COORDS_2D_FALLBACK = os.path.join(APP_ARTIFACTS_DIR, "coords2d_pca.npy")
PCA_MEAN_NPY_FALLBACK = os.path.join(APP_ARTIFACTS_DIR, "pca_mean.npy")
PCA_COMPONENTS_NPY_FALLBACK = os.path.join(APP_ARTIFACTS_DIR, "pca_components.npy")
TRAIN_META_FILTERED_CSV_FALLBACK = os.path.join(APP_ARTIFACTS_DIR, "train_meta_used.csv")

SUBMISSION_CSV = os.path.join(OUTPUT_DIR, "submission.csv")

TEST_WITH_POSTERS = os.path.join(DATA_DIR, "movies_test_with_posters.csv")
TRAIN_WITH_POSTERS = os.path.join(DATA_DIR, "movies_train_with_posters.csv")
