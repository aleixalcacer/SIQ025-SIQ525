# Sesión 4 — Preprocesamiento y transformación de datos

## Objetivos

En esta sesión aprenderemos a **preparar datos para modelado** aplicando transformaciones estándar y a construir pipelines reproducibles.

Al finalizar la sesión, serás capaz de:
- Aplicar **escalado** y **normalización** a variables numéricas.
- Codificar variables categóricas mediante **One-Hot**, **Label Encoding** y estrategias como **Target Encoding**.
- Realizar **feature engineering** básico para crear variables derivadas.
- Construir pipelines de preprocesamiento con scikit-learn y combinar limpieza y transformación en un flujo reproducible.

---

## Introducción

En las sesiones anteriores hemos visto cómo **cargar** y **limpiar** datos. El siguiente paso antes de entrenar modelos es **transformar** esos datos a una forma que los algoritmos puedan aprovechar bien.

A este conjunto de pasos lo llamamos **preprocesamiento**. Incluye, entre otros:

- Ajustar la **escala** de las variables numéricas.
- Convertir variables categóricas (texto) a **números**.
- Crear nuevas variables que resuman o combinen información (**feature engineering**).
- Integrar todo en un flujo reproducible (**pipelines**) que podamos aplicar tanto en entrenamiento como en producción.

La idea clave:  
> El mismo conjunto de pasos que aplicamos sobre los datos de entrenamiento debe poder aplicarse después, exactamente igual, a nuevos datos (por ejemplo, datos de test o datos en producción).

---

## Escalado y normalización de variables numéricas

Muchos modelos de machine learning **suponen** o **agradecen** que las variables numéricas tengan una escala comparable.  
Ejemplos:

- KNN (distancias): si una variable está medida en miles y otra entre 0 y 1, la primera dominará la distancia.
- Regresión lineal / logística con regularización: la penalización se ve afectada por la escala.
- SVM, PCA, redes neuronales… también suelen mejorar cuando las variables están escaladas.

En pandas y scikit-learn el escalado se hace con transformadores como `StandardScaler` o `MinMaxScaler`.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# df: DataFrame con columnas numéricas
num_cols = ['edad', 'ingresos', 'area']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Alternativa: Min-Max
mms = MinMaxScaler()
df[num_cols] = mms.fit_transform(df[num_cols])
```

### ¿Cuándo usar cada tipo de escalado?

- **StandardScaler**
  - Qué hace: resta la media y divide por la desviación estándar → cada columna queda con media ≈ 0 y desviación ≈ 1.
  - Cuándo usarlo:
    - Cuando las variables tienen una distribución más o menos **gaussiana**.
    - Cuando usamos modelos como **regresión lineal/logística**, **SVM**, **KNN**, **PCA**.
  - Ventaja: mantiene la forma de la distribución (solo cambia centro y escala).

- **MinMaxScaler**
  - Qué hace: transforma cada columna para que sus valores estén en un rango dado (por defecto [0, 1]).
  - Cuándo usarlo:
    - Cuando queremos mantener **proporciones** y trabajar en un rango acotado (por ejemplo, redes neuronales sencillas).
    - Cuando sabemos que los datos no tienen outliers muy extremos.
  - Inconveniente: es **sensible a outliers** (un valor extremo estira todo el rango).

- **Normalizer** (normalización por norma)
  - Qué hace: escala **cada fila** (cada muestra) para que su vector tenga norma 1 (por ejemplo, norma L2).
  - Cuándo usarlo:
    - Cuando nos importa más la **dirección** del vector que su magnitud.
    - Típico en problemas de texto con **TF-IDF**, donde se compara la similitud de documentos.
  - Importante: no es lo mismo que escalar columnas; aquí se normaliza por **filas**.

- **RobustScaler**
  - Qué hace: usa mediana e IQR (rango intercuartílico) para escalar, por lo que es **robusto a outliers**.
  - Cuándo usarlo:
    - Cuando hay valores atípicos (muy grandes o muy pequeños) que podrían distorsionar StandardScaler o MinMaxScaler.

- **Cuándo NO hace falta escalar**
  - Modelos basados en árboles (DecisionTree, RandomForest, GradientBoosting, XGBoost…) **no requieren** escalado para funcionar bien:
    - Dividen el espacio según umbrales, no por distancias.
    - Escalar o no suele cambiar poco o nada en su rendimiento.
  - Aun así, en pipelines mixtos (con otras técnicas) puede ser útil mantener un tratamiento homogéneo.

---

## Codificación de variables categóricas

Los modelos de scikit-learn trabajan, en general, con **números**, no con texto.  
Por tanto, las variables categóricas (por ejemplo, `ciudad = "Madrid"`) deben convertirse a representaciones numéricas.

Hay varias estrategias, con ventajas e inconvenientes:

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

cat_cols = ['ciudad', 'tipo_vivienda']

# One-Hot (útil para muchos modelos, especialmente lineales)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe_arr = ohe.fit_transform(df[cat_cols])

# Label / Ordinal encoding (si el orden tiene sentido o para tree models)
ord_enc = OrdinalEncoder()
df['tipo_vivienda_ord'] = ord_enc.fit_transform(df[['tipo_vivienda']])
```

### One-Hot Encoding

- Crea una columna binaria (0/1) por cada categoría.
- Ejemplo: `color = ["rojo", "azul", "verde"]` → columnas `color_rojo`, `color_azul`, `color_verde`.
- Ventajas:
  - No introduce un **orden artificial** entre categorías.
  - Bien entendido por **modelos lineales**, redes neuronales, etc.
- Inconvenientes:
  - Si hay muchas categorías, se generan muchas columnas (problema de **dimensionalidad**).
- Parámetros clave:
  - `handle_unknown='ignore'` evita errores cuando aparece una categoría nueva en predicción.

### Label / Ordinal Encoding

- Asigna un número entero a cada categoría: `"piso" → 0`, `"casa" → 1`, `"ático" → 2`, etc.
- Ventajas:
  - Muy sencillo y compacto (una sola columna).
- Inconvenientes:
  - Introduce un **orden artificial** entre categorías que puede no existir (¿es “ático” > “piso”?).
  - Para modelos lineales, esto puede llevar a interpretaciones erróneas.

Se usa con más sentido cuando:
- La variable tiene un orden natural (por ejemplo: `["bajo", "medio", "alto"]`).
- O cuando usamos **modelos de árboles**, que no interpretan el número como distancia, sino solo para comparar.

### Target Encoding (codificación por el objetivo)

En problemas supervisados (tenemos una variable objetivo `y`, por ejemplo el precio o si algo es fraude), podemos codificar categorías usando información del objetivo.

Ejemplo simple con pandas:

```python
# Encoding por media del objetivo por categoría (simple y con smoothing en práctica)
target_mean = df.groupby('ciudad')['precio'].mean()
df['ciudad_te'] = df['ciudad'].map(target_mean)
```

- Idea: cada categoría se sustituye por la **media del objetivo** para esa categoría.
  - Ejemplo: si en “Madrid” el precio medio es 300k y en “Sevilla” es 200k, entonces:
    - `ciudad_te = 300000` para registros de Madrid,
    - `ciudad_te = 200000` para registros de Sevilla.
- Ventajas:
  - Muy potente cuando hay **muchas categorías** y algunas tienen pocos datos.
- Riesgo importante:
  - **Fuga de información (data leakage)** si calculas las medias usando **todo** el dataset (incluyendo test).
  - Para evitarlo, debe calcularse **solo con datos de entrenamiento** y, mejor todavía, con esquemas tipo K-fold.

---

## Feature engineering básico

**Feature engineering** es el proceso de crear nuevas variables (features) a partir de las existentes para dar más información útil al modelo.

La idea es:  
> A veces, una combinación adecuada de columnas explica mucho mejor el problema que las columnas originales por separado.

Ejemplos:

```python
# Ejemplos
df['area_m2'] = df['largo'] * df['ancho']
df['precio_por_m2'] = df['precio'] / df['area_m2']
df['anio_antiguedad'] = 2025 - df['anio_construccion']
```

- `area_m2`: a partir de `largo` y `ancho`.
  - Útil en problemas de viviendas, terrenos, etc.
- `precio_por_m2`: normaliza el precio respecto al tamaño.
  - Permite comparar viviendas de distinto tamaño de forma más justa.
- `anio_antiguedad`: transforma un año en una medida de “cuántos años tiene”.
  - Más interpretable para el modelo que el año en bruto.

Buenas prácticas:
- Crear features que tengan **sentido de negocio** (no combinaciones aleatorias).
- Evaluar su utilidad:
  - Mirando correlaciones.
  - Probando modelos con y sin ellas y comparando métricas.

---

## Imputación de valores faltantes en scikit-learn

En la **sesión 3** ya vimos cómo detectar y tratar valores faltantes utilizando **pandas**, por ejemplo con métodos como `.isna()`, `.fillna()` o `dropna()`.  
Esa aproximación es muy útil para una primera limpieza manual y exploratoria de los datos.

En esta sesión damos un paso más: veremos cómo realizar la **imputación de valores faltantes dentro de los pipelines de scikit-learn**, de forma que el proceso quede integrado y sea completamente **reproducible**. La idea es que:

- El cálculo de los valores de imputación (media, mediana, moda, etc.) se haga **solo con los datos de entrenamiento**.
- Esos mismos parámetros se apliquen automáticamente a cualquier dato nuevo (validación, test, producción).

Para esto utilizamos el transformador `SimpleImputer`:

```python
from sklearn.impute import SimpleImputer
import pandas as pd

df = pd.read_csv("data/ejemplo.csv")

# Columnas numéricas
num_cols = ["edad", "ingresos"]

imputer_num = SimpleImputer(strategy="median")
df[num_cols] = imputer_num.fit_transform(df[num_cols])

# Columnas categóricas (ejemplo)
cat_cols = ["ciudad"]
imputer_cat = SimpleImputer(strategy="most_frequent")
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
```

- Para **numéricas**, suele ser habitual usar:
  - `strategy="mean"` (media) o
  - `strategy="median"` (mediana, más robusta a outliers).
- Para **categóricas**, una opción sencilla es:
  - `strategy="most_frequent"` (valor más frecuente).

Más adelante, integraremos estos `SimpleImputer` dentro de un **`Pipeline`** y de un **`ColumnTransformer`**, de forma que la imputación se aplique junto con el resto de transformaciones (escalado, codificación, etc.).

---

## Pipelines de preprocesamiento con scikit-learn

Hasta ahora hemos visto transformaciones por separado. En proyectos reales, queremos:

1. Definir **todas las transformaciones** (imputación, escalado, codificación, etc.).
2. Aplicarlas siempre en el **mismo orden**.
3. Integrarlas con el **modelo** para que todo se entrene junto.

Para esto usamos dos componentes clave de scikit-learn:

- `ColumnTransformer`: permite aplicar **transformaciones distintas a grupos de columnas diferentes** (numéricas vs categóricas).
- `Pipeline`: encadena varios pasos secuenciales (preprocesamiento + modelo).

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_cols = ['edad', 'ingresos', 'area']
cat_cols = ['ciudad', 'tipo_vivienda']

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

model_pipe = Pipeline([
    ('preproc', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Entrenar pipeline completo
X = df.drop(columns=['precio'])
y = df['precio']
model_pipe.fit(X, y)
```

### ¿Qué está pasando aquí?

1. **`num_pipe`**:
   - Primero imputa valores faltantes numéricos con la **mediana**.
   - Luego aplica `StandardScaler` para escalar esas columnas.

2. **`cat_pipe`**:
   - Imputa valores faltantes categóricos con el **valor más frecuente**.
   - Aplica `OneHotEncoder` para pasar texto a variables binarias.

3. **`preprocessor` (ColumnTransformer)**:
   - Aplica `num_pipe` solo a `num_cols`.
   - Aplica `cat_pipe` solo a `cat_cols`.
   - Devuelve una matriz combinada con todas las nuevas columnas.

4. **`model_pipe` (Pipeline)**:
   - Paso 1: `preproc` → aplica todo el preprocesamiento anterior.
   - Paso 2: `model` → entrena un `RandomForestRegressor` con los datos ya preprocesados.

Ventajas de este enfoque:

- **Reproducibilidad**: entrenar y predecir usan exactamente los mismos pasos.
- **Evita fugas de información**:
  - `fit` de los transformadores se hace solo con los datos de entrenamiento dentro del pipeline.
- **Simplicidad**:
  - Para predecir nuevas muestras solo llamamos a `model_pipe.predict(nuevos_datos)`.

---

## Integración de limpieza y transformación en un flujo reproducible

En un proyecto real, el flujo típico sería:

1. Cargar datos brutos.
2. Aplicar **limpieza básica** (por ejemplo, corrección de tipos, eliminación de registros imposibles).
3. Definir un **pipeline de preprocesamiento**:
   - Imputación de nulos.
   - Escalado numérico.
   - Codificación categórica.
   - (Opcional) creación de nuevas features.
4. Integrar ese pipeline con uno o varios **modelos**.
5. Guardar el pipeline entrenado para reutilizarlo (por ejemplo, con `joblib`).

Buenas prácticas:

- Separar claramente:
  - Limpieza “física” de datos (formato, columnas imposibles) → puede hacerse antes del pipeline.
  - Transformaciones estadísticas (imputación, escalado, codificación) → mejor dentro del pipeline.
- Ajustar (`fit`) los transformadores siempre solo con **entrenamiento**, no con test.
- Documentar las decisiones: por qué elegimos una estrategia de imputación, por qué este scaler, etc.
- Versionar datos, código y modelos (por ejemplo, en Git).

---

## Ejercicios sugeridos

1. Transformar un dataset completo aplicando:
   - Imputación de valores faltantes en variables numéricas y categóricas.
   - Escalado de las variables numéricas.
   - Codificación de las variables categóricas.
2. Crear un **pipeline de preprocesamiento** que integre escalado y codificación y conectarlo con un modelo (por ejemplo, regresión lineal o RandomForest).
3. Comparar el rendimiento del mismo modelo:
   - Sin preprocesamiento (solo variables crudas).
   - Con preprocesamiento completo (pipeline).

---

## Conclusiones de la sesión

- El **preprocesamiento** es una parte esencial del flujo de machine learning: condiciona fuertemente la calidad de los modelos.
- Escalar y codificar adecuadamente las variables permite que los algoritmos trabajen de forma más **estable y eficiente**.
- El **feature engineering** puede aportar mucha información extra al modelo y mejorar su rendimiento.
- Los **pipelines** y `ColumnTransformer` son herramientas clave para crear flujos de trabajo **reproducibles**, evitar errores y facilitar el despliegue del modelo.
- A partir de esta sesión podremos construir modelos más robustos y profesionales en las siguientes etapas del curso.
