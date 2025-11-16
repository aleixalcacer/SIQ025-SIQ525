# Sesión 2 — Exploración y Limpieza de Datos con pandas (EDA)

## Introducción
Una vez que tenemos los datos cargados en Python, el siguiente paso es **explorarlos y limpiarlos**. Esta fase se conoce como **Exploratory Data Analysis (EDA)** y es fundamental para comprender la estructura, las características y la calidad de los datos antes de realizar cualquier análisis avanzado o modelado. EDA nos ayuda a identificar errores, valores faltantes, outliers, tendencias y relaciones importantes que pueden influir en nuestras decisiones o modelos.

En esta sesión profundizaremos en el uso de **pandas** y **NumPy**, además de utilizar librerías de visualización como **Seaborn** y **Matplotlib** para generar gráficos que nos permitan entender mejor los datos.

---

## Importación de datos
Para comenzar, necesitamos importar los datos a un DataFrame de pandas. Los datos pueden provenir de archivos CSV, Excel, Parquet u otras fuentes.

```python
import pandas as pd

# Cargar un archivo CSV
df = pd.read_csv('data/ejemplo.csv')

# Mostrar las primeras filas
df.head()
```
Esto nos permite obtener una vista rápida de los datos y familiarizarnos con las columnas y tipos de información.

---

## Exploración básica de los datos
pandas ofrece múltiples métodos para inspeccionar rápidamente un dataset:
- `df.head(n)` → Muestra las primeras n filas.
- `df.tail(n)` → Muestra las últimas n filas.
- `df.info()` → Información general sobre columnas, tipos y valores nulos.
- `df.describe()` → Estadísticas descriptivas de columnas numéricas.
- `df.shape` → Número de filas y columnas.
- `df.columns` → Nombres de las columnas.

```python
print(df.shape)
print(df.columns)
df.info()
df.describe()
```
Esta exploración inicial nos permite identificar rápidamente columnas con datos faltantes o tipos de datos incorrectos.

---

## Limpieza de datos
Los datos reales casi siempre contienen errores, duplicados o valores faltantes. Algunas técnicas básicas de limpieza incluyen:

### 1. Valores nulos
```python
# Contar valores nulos por columna
print(df.isnull().sum())

# Eliminar filas con valores nulos
df_limpio = df.dropna()

# Rellenar valores nulos con un valor específico
df['columna'] = df['columna'].fillna(0)
```

### 2. Duplicados
```python
# Eliminar filas duplicadas
df = df.drop_duplicates()
```

### 3. Reemplazar valores
```python
# Reemplazar valores específicos
df['columna'] = df['columna'].replace({'antiguo_valor': 'nuevo_valor'})
```

### 4. Conversión de tipos
```python
# Convertir columna a numérica
df['columna'] = pd.to_numeric(df['columna'], errors='coerce')
```
Estas acciones aseguran que los datos estén en un formato adecuado para análisis y modelado posterior.

---

## Filtrado y selección de datos
Seleccionar subconjuntos de datos según condiciones es muy común en EDA:
```python
# Filtrar filas donde columna > 10
df_filtrado = df[df['columna'] > 10]

# Filtrar usando varias condiciones
df_filtrado = df[(df['columna1'] > 10) & (df['columna2'] == 'valor')]

# Seleccionar columnas específicas
sub_df = df[['columna1', 'columna3']]
```
Esto permite concentrarse en los datos de interés y preparar conjuntos de datos específicos para análisis más detallados.

---

## Visualización básica
Visualizar los datos ayuda a detectar patrones y relaciones que no se ven fácilmente en tablas.

### Histogramas y distribuciones
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['columna_numerica'], bins=20)
plt.show()
```

### Diagramas de caja (boxplot)
```python
sns.boxplot(x='categoria', y='valor', data=df)
plt.show()
```

### Gráficos de dispersión (scatterplot)
```python
sns.scatterplot(x='columna1', y='columna2', hue='categoria', data=df)
plt.show()
```
Estas visualizaciones permiten detectar outliers, tendencias y relaciones entre variables.

---

## Ejercicio guiado
1. Cargar un dataset CSV proporcionado.
2. Explorar sus primeras filas y obtener estadísticas descriptivas.
3. Identificar y tratar valores nulos o duplicados.
4. Filtrar un subconjunto de datos según condiciones específicas.
5. Generar un histograma, un boxplot y un scatterplot para explorar distribuciones y relaciones.

Este ejercicio permite practicar todas las técnicas de exploración y limpieza de datos vistas en la sesión.

---

## Conclusiones de la sesión
- La **exploración y limpieza de datos** es un paso crítico en cualquier análisis de datos.
- pandas y NumPy ofrecen herramientas potentes para manipular datos de manera eficiente.
- Seaborn y Matplotlib permiten crear visualizaciones informativas para entender patrones, detectar errores y presentar resultados.
- Mantener un flujo de trabajo organizado y reproducible desde el inicio facilita el análisis y la comunicación de los resultados.

En la próxima sesión se profundizará en **preprocesamiento avanzado y preparación de datos para modelado**, utilizando técnicas como escalado, codificación de variables categóricas y pipelines de scikit-learn.

