# Sesión 3 — Preprocesamiento y Preparación de Datos para Modelado

## Introducción
Después de explorar y limpiar nuestros datos, el siguiente paso es **prepararlos para el modelado**. El preprocesamiento es crucial porque los algoritmos de machine learning requieren datos consistentes, normalizados y codificados de manera adecuada. Un preprocesamiento correcto mejora el rendimiento del modelo y evita errores durante el entrenamiento.

En esta sesión aprenderemos a utilizar técnicas de **escalado de variables, codificación de variables categóricas, imputación de valores faltantes** y cómo combinar estos pasos en un **pipeline reproducible con scikit-learn**. Además, repasaremos conceptos básicos de modelado para contextualizar por qué estas transformaciones son necesarias.

---

## Escalado de variables
Muchos algoritmos de machine learning requieren que las características tengan la misma escala para funcionar correctamente, especialmente aquellos basados en distancia (como KNN o clustering).

### StandardScaler
```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

# DataFrame de ejemplo
data = {
    'altura': [1.60, 1.75, 1.80, 1.55],
    'peso': [55, 70, 80, 50]
}
df = pd.DataFrame(data)

# Escalar características
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
print(df_scaled)
```
El resultado es un array donde cada columna tiene media 0 y desviación estándar 1, listo para entrenar modelos.

---

## Codificación de variables categóricas
Los modelos de machine learning no entienden texto, por lo que debemos transformar las categorías en valores numéricos.

### One-Hot Encoding
```python
from sklearn.preprocessing import OneHotEncoder

# DataFrame con categoría
df = pd.DataFrame({'color': ['rojo', 'azul', 'verde', 'rojo']})
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['color']])
print(encoded)
```
Esto crea columnas binarias para cada categoría, permitiendo que el modelo las use correctamente.

---

## Imputación de valores faltantes
Los valores nulos pueden causar errores en el entrenamiento. La imputación consiste en reemplazarlos por un valor adecuado.

```python
from sklearn.impute import SimpleImputer

# DataFrame con valores faltantes
df = pd.DataFrame({'edad': [25, 30, None, 40]})
imputer = SimpleImputer(strategy='mean')
df['edad'] = imputer.fit_transform(df[['edad']])
print(df)
```
La estrategia puede ser la media, la mediana, la moda o un valor constante.

---

## Selección de características
Seleccionar las variables más relevantes mejora la interpretabilidad y rendimiento del modelo.

```python
from sklearn.feature_selection import SelectKBest, f_regression

X = df[['altura', 'peso']]
y = pd.Series([1, 2, 3, 4])  # variable objetivo

selector = SelectKBest(score_func=f_regression, k=1)
X_new = selector.fit_transform(X, y)
print(X_new)
```
Esto permite reducir dimensionalidad y concentrarse en las características más informativas.

---

## Pipelines en scikit-learn
Para combinar múltiples pasos de preprocesamiento de manera ordenada y reproducible, podemos usar **Pipeline**.

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

X_preprocessed = pipeline.fit_transform(df)
print(X_preprocessed)
```
Esto asegura que cada paso se aplique en el orden correcto y facilita la integración con modelos.

---

## Introducción al modelado de datos
Una vez preprocesados los datos, podemos entrenar modelos. Algunos ejemplos básicos:
- **Regresión lineal**: para predecir valores continuos.
- **Regresión logística**: para predecir categorías.
- **Árboles de decisión y Random Forest**: para clasificación y regresión.
- **KMeans**: para clustering.
- **MLPClassifier**: redes neuronales básicas.

### Ejemplo con regresión lineal
```python
from sklearn.linear_model import LinearRegression

X = df[['altura', 'peso']]
y = pd.Series([60, 75, 80, 55])

model = LinearRegression()
model.fit(X, y)
print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)
```
Esto permite ver cómo las variables escaladas afectan la predicción.

---

## Ejercicio guiado
1. Cargar un dataset limpio de la sesión anterior.
2. Identificar columnas numéricas y categóricas.
3. Aplicar imputación de valores faltantes.
4. Escalar las variables numéricas y codificar las categóricas.
5. Seleccionar las características más relevantes.
6. Crear un pipeline que combine todos estos pasos.
7. Entrenar un modelo simple de regresión o clasificación y evaluar sus resultados.

---

## Conclusiones de la sesión
- El **preprocesamiento de datos** es esencial antes de entrenar modelos de machine learning.
- Escalado, codificación y imputación garantizan que los algoritmos funcionen correctamente.
- Los pipelines permiten reproducibilidad y organización profesional del flujo de trabajo.
- Esta preparación abre el camino para análisis más complejos y modelado avanzado en futuras sesiones.

