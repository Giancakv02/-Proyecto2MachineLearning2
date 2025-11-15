# Recomendador Visual de Películas (Clustering de Pósters)

## Descripción
Este proyecto construye un sistema de recomendación de películas **basado únicamente en similitud visual entre pósters/fotogramas**, usando técnicas clásicas de visión por computadora y **aprendizaje no supervisado implementado desde cero (sin scikit-learn)**.

### Pipeline general
1. **Extracción de características visuales** (src/feature_extractor.py):
   - Histogramas de color HSV (color global)
   - LBP + GLCM (textura)
   - HOG (forma/bordes)
2. **Reducción de dimensionalidad** (src/dimensionality.py):
   - PCA implementado manualmente con autovectores de la matriz de covarianza
   - SVD implementado manualmente con `numpy.linalg.svd`
3. **Clustering no supervisado** (src/clustering.py):
   - K-Means implementado a mano
   - DBSCAN implementado a mano
4. **Evaluación** (src/evaluation.py):
   - Silhouette score implementado a mano
   - Coherencia de géneros dentro de cada cluster (pureza de género dominante)
5. **Visualizador interactivo** (app.py con Streamlit):
   - Buscar una película y sugerir similares visualmente
   - Ver qué películas caen en cada cluster
   - Visualizar distribución 2D (PCA 2D implementado manualmente)
   - Filtrar por género y año

## Datos
Colocar en `data/`:
- `movies_train.csv` con columnas mínimas:
  - `movie_id` (identificador único)
  - `title`
  - `year`
  - `genres` (por ejemplo `"Action|Sci-Fi"`)
  - `poster_path` (opcional; si falta, se asume `data/posters/<movie_id>.jpg`)
- Carpeta `data/posters/` con las imágenes de cada película.

## Cómo correr
1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt# -Proyecto2MachineLearning2
# -Proyecto2MachineLearning2
