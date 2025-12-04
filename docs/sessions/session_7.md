# Sesión 7 — Modelización con Machine Learning

## Objetivos

En esta sesión daremos una visión completa del flujo de **machine learning** aplicado a datos tabulares, tanto en problemas supervisados como no supervisados.

Al finalizar la sesión, serás capaz de:
- Entender el **pipeline completo**: preprocesamiento → entrenamiento → evaluación.
- Aplicar técnicas de **validación** (train/test y validación cruzada) y razonar sobre **overfitting** y **underfitting**.
- Entrenar y evaluar modelos supervisados de **regresión** y **clasificación**.
- Aplicar **clustering** (K-Means) a problemas de segmentación y exploración de patrones.
- Integrar preprocesamiento y modelo en **pipelines de scikit-learn**.

---

## Introducción

Hasta ahora hemos trabajado en:

- Importar y manipular datos (`pandas`).
- Limpiar y preprocesar (valores perdidos, outliers, tipos, escalado y codificación).
- Visualizar y explorar (EDA).

En esta sesión pasamos a la **modelización**: queremos que un algoritmo aprenda patrones a partir de datos para:

- **Predecir** una variable objetivo (regresión o clasificación).
- **Descubrir grupos** o estructuras ocultas (clustering).

Es importante ver la modelización como parte de un **proceso repetitivo**:
1. Preprocesar datos.
2. Entrenar modelo.
3. Evaluar.
4. Ajustar (hiperparámetros, features, limpieza).
5. Volver al paso 1–3 si es necesario.

---

## Fundamentos del pipeline de Machine Learning

### División train/test y su importancia

Siempre necesitamos evaluar el modelo en **datos que no ha visto en el entrenamiento**.

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["objetivo"])
y = df["objetivo"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # opcional, útil en clasificación
)
```

- `train` → se usa para **entrenar** el modelo (ajustar parámetros).
- `test` → se usa solo al final para **evaluar**.
- `stratify=y` mantiene proporciones de clases (útil en clasificación).

Nunca debemos “mirar” los resultados de test para ajustar hiperparámetros de manera sistemática (eso sería fuga de información).

### Validación cruzada (cross-validation)

La **validación cruzada** mejora la estimación del rendimiento dividiendo el conjunto de entrenamiento en varios “folds”.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

print("Scores en cada fold:", scores)
print("Media R2:", scores.mean())
```

- `cv=5` → 5 particiones.
- `cross_val_score`:
  - Entrena y evalúa en cada fold.
  - Devuelve una lista de puntuaciones.

Esto nos ayuda a detectar si el modelo es muy sensible a la división de los datos.

### Overfitting y underfitting

- **Underfitting**:
  - El modelo es **demasiado simple**.
  - No captura los patrones de los datos.
  - Mal rendimiento en **train y test**.
- **Overfitting**:
  - El modelo es **demasiado complejo**.
  - Aprende ruido y detalles específicos del train.
  - Muy buen rendimiento en **train**, peor en **test**.

Queremos un modelo que tenga un **equilibrio**: lo bastante flexible para aprender, pero sin memorizar el ruido.

Ejemplos típicos:

- Árbol de decisión muy profundo → riesgo de overfitting.
- Regresión lineal en un problema no lineal → underfitting.

### Integración de preprocesamiento + modelo en un pipeline

Como en la sesión 4, usamos `Pipeline` y `ColumnTransformer` para asegurar que:

- Todas las transformaciones (imputación, escalado, codificación) se hacen de forma consistente.
- No hay fuga de información entre train y test.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

num_cols = ["edad", "ingresos"]
cat_cols = ["ciudad", "tipo_vivienda"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

clf_pipe = Pipeline([
    ("preproc", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

clf_pipe.fit(X_train, y_train)
y_pred = clf_pipe.predict(X_test)
```

---

## Aprendizaje Supervisado: Regresión

En problemas de **regresión**, la variable objetivo es numérica (precio, temperatura, ingresos, etc.).

### Regresión lineal multivariante

La regresión lineal multivariante asume que la relación entre las features y el objetivo es aproximadamente lineal.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
```

### Métricas de regresión

- **MAE (Mean Absolute Error)**:
  - Media del valor absoluto de los errores.
  - Fácil de interpretar (misma unidad que la variable objetivo).
- **RMSE (Root Mean Squared Error)**:
  - Raíz cuadrada del error cuadrático medio.
  - Penaliza más los errores grandes.
- **R² (coeficiente de determinación)**:
  - Indica qué proporción de la varianza del objetivo explica el modelo.
  - 1 → perfecto, 0 → no mejor que predecir la media.

---

## Aprendizaje Supervisado: Clasificación

En problemas de **clasificación**, la variable objetivo es categórica (ej. “spam/no spam”, “aprobado/suspenso”, tipo de cliente).

### Regresión logística

La **regresión logística** es un modelo lineal para clasificación binaria (y extensiones a multiclase).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="binary")  # o 'macro'/'weighted' en multiclase
recall = recall_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
```

### k-Nearest Neighbors (k-NN)

k-NN clasifica una muestra según las clases de sus **k vecinos más cercanos**.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
```

- Sensible a la **escala** de las variables → importante usar escalado (por ejemplo, en un pipeline con `StandardScaler`).
- Hiperparámetro clave: `n_neighbors`.

### Árboles de decisión

Los **árboles de decisión** dividen recursivamente el espacio de features.

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
```

- Interpretables: podemos visualizar reglas (aunque árboles grandes se vuelven difíciles).
- No requieren escalado de variables.
- Riesgo de overfitting si `max_depth` es muy grande o sin restricciones.

### Métricas de clasificación y matriz de confusión

- **Accuracy**: proporción de aciertos.
- **Precision**: de los que predije como positivos, ¿cuántos lo eran?
- **Recall**: de los positivos reales, ¿cuántos detecté?
- **F1-score**: media armónica de precision y recall (útil cuando hay clases desbalanceadas).

Matriz de confusión:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matriz de confusión")
plt.show()
```

Interpretación (para binaria):
- TP (True Positives), FP (False Positives).
- TN (True Negatives), FN (False Negatives).

Dependiendo del problema, puede interesar más **minimizar falsos positivos** o **falsos negativos**.

---

## Aprendizaje No Supervisado: Clustering

En **clustering**, no hay etiqueta/objetivo. Queremos agrupar observaciones similares.

### K-Means clustering

`KMeans` agrupa los datos en `k` clusters, intentando que los puntos de cada cluster estén lo más cerca posible de su centro.

```python
from sklearn.cluster import KMeans

X_clust = df[["feature1", "feature2"]]  # seleccionar variables numéricas

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clust)

labels = kmeans.labels_          # cluster asignado a cada muestra
centers = kmeans.cluster_centers_
```

Visualización en 2D:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.scatter(X_clust["feature1"], X_clust["feature2"], c=labels, cmap="tab10", alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centroides")
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.title("K-Means con 3 clusters")
plt.legend()
plt.show()
```

### Elección del número óptimo de clusters

Dos enfoques habituales:

1. **Método del codo (Elbow)**:
   - Calculamos la suma de distancias intra-cluster (`inertia_`) para varios valores de `k`.
   - Buscamos un “codo” en la gráfica.

```python
inertias = []
K = range(1, 10)

for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_clust)
    inertias.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K, inertias, marker="o")
plt.xlabel("Número de clusters k")
plt.ylabel("Inercia (within-cluster SSE)")
plt.title("Método del codo para elegir k")
plt.show()
```

2. **Coeficiente de silhouette**:
   - Mide qué tan bien separado está cada punto de otros clusters.
   - Valores cercanos a 1 → buena separación; cercanos a 0 → solapamiento; negativos → mala asignación.

```python
from sklearn.metrics import silhouette_score

for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    labels_k = km.fit_predict(X_clust)
    sil = silhouette_score(X_clust, labels_k)
    print(f"k={k}, silhouette={sil:.3f}")
```

### Interpretación y casos de uso

Una vez asignados los clusters:

- Analizar características medias por cluster (por ejemplo, perfil de cliente):

```python
df_clusters = df.copy()
df_clusters["cluster"] = labels
print(df_clusters.groupby("cluster").mean())
```

Casos de uso típicos:
- **Segmentación de clientes** (por comportamiento de compra, uso de servicios, etc.).
- **Detección de patrones** en sensores, datos de uso de aplicaciones, etc.
- Como paso previo para modelos supervisados (por ejemplo, añadir el cluster como feature).

---

## Ejercicios sugeridos

1. **Regresión supervisada**:
   - Tomar un dataset real (por ejemplo, viviendas, precio de coches).
   - Entrenar un modelo de **regresión lineal** (posiblemente dentro de un pipeline con preprocesamiento).
   - Evaluar con MAE, RMSE y R².

2. **Clasificación supervisada**:
   - Entrenar al menos **dos modelos de clasificación** (por ejemplo, regresión logística y KNN, o logística y árbol de decisión).
   - Comparar métricas: accuracy, precision, recall, F1.
   - Analizar la matriz de confusión y comentar errores típicos.

3. **Clustering (K-Means)**:
   - Aplicar K-Means a un dataset adecuado para **segmentación** (por ejemplo, clientes, alumnos, dispositivos).
   - Probar distintos valores de `k` y usar el **método del codo** y/o **silhouette** para justificar una elección.
   - Visualizar los clusters resultantes (si es posible en 2D) e interpretar las diferencias entre grupos.

4. **Pipelines completos**:
   - Integrar en un `Pipeline`:
     - Preprocesamiento (imputación, escalado, codificación).
     - Modelo (regresión o clasificación).
   - Usar validación cruzada para estimar el rendimiento del pipeline.

---

## Conclusiones de la sesión

- La modelización con Machine Learning se basa en un **pipeline completo** donde datos limpios y bien preprocesados son tan importantes como el modelo.
- La **división train/test** y la **validación cruzada** son esenciales para estimar la capacidad de generalización y evitar engañarnos con resultados sobre-entrenados.
- En **regresión**, hemos trabajado con la regresión lineal y métricas como MAE, RMSE y R².
- En **clasificación**, hemos visto modelos básicos (regresión logística, k-NN, árboles de decisión) y métricas como accuracy, precision, recall, F1 y la matriz de confusión.
- En **clustering**, K-Means nos permite descubrir grupos en datos sin etiquetas y es muy útil para segmentación y exploración.
- Integrar preprocesamiento y modelo en **pipelines** facilita la reproducibilidad, evita fugas de información y acerca nuestro trabajo a un entorno profesional.
