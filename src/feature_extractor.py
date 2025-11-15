import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from .config import IMG_SIZE, HIST_BINS

def _resize_and_gray(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None, None
    img_bgr = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, gray

def _color_hist_hsv(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hists = []
    for ch in range(3):
        hist = cv2.calcHist([img_hsv],[ch],None,[HIST_BINS],[0,256])
        hist = cv2.normalize(hist, hist).flatten()
        hists.append(hist)
    return np.concatenate(hists, axis=0)

def _lbp_hist(gray):
    # LBP uniforme 8 vecinos, radio 1
    P = 8
    R = 1
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    (hist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, P + 3),
        range=(0, P + 2)
    )
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist

def _glcm_feats(gray):
    # reducimos resolución y cuantizamos gris para tener niveles pequeños
    small = cv2.resize(gray, (64,64))
    small_q = (small / 4).astype(np.uint8)  # niveles 0-63
    glcm = graycomatrix(
        small_q,
        distances=[1],
        angles=[0],
        levels=64,
        symmetric=True,
        normed=True
    )
    feats = [
        graycoprops(glcm, 'contrast')[0,0],
        graycoprops(glcm, 'homogeneity')[0,0],
        graycoprops(glcm, 'energy')[0,0],
        graycoprops(glcm, 'correlation')[0,0]
    ]
    return np.array(feats, dtype=np.float32)

def _hog_feats(gray):
    # HOG clásico
    h = hog(
        gray,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        orientations=9,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return h.astype(np.float32)

def extract_features_for_image(path):
    img_bgr, gray = _resize_and_gray(path)
    if img_bgr is None or gray is None:
        return None

    color_vec = _color_hist_hsv(img_bgr)
    lbp_vec   = _lbp_hist(gray)
    glcm_vec  = _glcm_feats(gray)
    hog_vec   = _hog_feats(gray)

    feat_vec = np.concatenate([color_vec, lbp_vec, glcm_vec, hog_vec], axis=0)
    return feat_vec