# Sesión 3 — Limpieza de datos

## Objetivos

En esta sesión aprenderemos técnicas profesionales de **limpieza de datos**, centrándonos en identificar y resolver problemas comunes que aparecen en datasets reales.

Al finalizar la sesión, serás capaz de:
- Aplicar distintas estrategias de **gestión de valores perdidos** (eliminación e imputación).
- Detectar y tratar **valores atípicos (outliers)**.
- Identificar y corregir **duplicados y datos inconsistentes**.
- Validar y corregir **tipos de datos y formatos**.
- Realizar una **exploración inicial de la calidad de los datos (EDA básico)**.

---

## Introducción

En la mayoría de proyectos reales, los datos no vienen “limpios”: hay valores perdidos, formatos inconsistentes, errores de introducción, duplicados y valores extremos que pueden distorsionar los análisis. La limpieza de datos es un paso crítico antes de cualquier análisis o modelado, y suele consumir una parte importante del tiempo del proyecto.

En esta sesión trabajaremos con pandas para identificar estos problemas y aplicar soluciones sistemáticas, documentando siempre las decisiones tomadas para garantizar la reproducibilidad y la transparencia del análisis.

---

## Gestión de valores perdidos

Los valores perdidos pueden aparecer como `NaN`, cadenas vacías o códigos especiales (por ejemplo `-999`). Es importante **detectarlos**, **cuantificarlos** y decidir una estrategia de tratamiento.

### Detección de valores perdidos

```python
import pandas as pd

df = pd.read_csv("data/datos_ejemplo.csv")

# Conteo de valores nulos por columna
print(df.isna().sum())

# Porcentaje de nulos por columna
porcentaje_nulos = df.isna().mean() * 100
print(porcentaje_nulos)
```

### Eliminación de valores perdidos

```python
# Eliminar filas con cualquier valor nulo
df_drop_any = df.dropna()

# Eliminar filas solo si una columna concreta es nula
df_drop_col = df.dropna(subset=["columna_importante"])
```

### Imputación simple de valores perdidos

```python
# Rellenar con un valor constante
df["columna_numerica"] = df["columna_numerica"].fillna(0)

# Rellenar con la media
media = df["columna_numerica"].mean()
df["columna_numerica"] = df["columna_numerica"].fillna(media)

# Rellenar categóricas con la moda
moda = df["columna_categorica"].mode().iloc[0]
df["columna_categorica"] = df["columna_categorica"].fillna(moda)

# Rellenar hacia adelante (forward fill), completando con el último valor válido
df["columna_temporal"] = df["columna_temporal"].fillna(method="ffill")
```

La elección de estrategia depende del contexto del problema; es importante justificarla y documentarla.

---

## Detección de outliers y estrategias de tratamiento

Los **outliers** son valores extremos que pueden ser errores o casos raros pero válidos. Pueden distorsionar medias, desviaciones estándar y modelos.

### Detección basada en rangos (IQR)

```python
col = "columna_numerica"
q1 = df[col].quantile(0.25)
q3 = df[col].quantile(0.75)
iqr = q3 - q1

limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]
print(outliers[[col]])
```

### Estrategias de tratamiento

```python
# 1) Eliminar filas con outliers
df_sin_outliers = df[(df[col] >= limite_inferior) & (df[col] <= limite_superior)]

# 2) Recortar (winsorizar) a los límites
df_recortado = df.copy()
df_recortado[col] = df_recortado[col].clip(limite_inferior, limite_superior)
```

No siempre es correcto eliminar outliers; depende del dominio del problema y del objetivo del análisis.

---

## Detección de duplicados y datos inconsistentes

Los **duplicados** pueden inflar estadísticas y sesgar resultados, mientras que los valores inconsistentes dificultan el análisis (por ejemplo, "Madrid", "MADRID", "mad").

### Duplicados

```python
# Detectar filas duplicadas completas
duplicadas = df[df.duplicated()]
print(duplicadas)

# Eliminar duplicados manteniendo la primera aparición
df_sin_duplicados = df.drop_duplicates()

# Duplicados según un subconjunto de columnas
df_sin_duplicados_id = df.drop_duplicates(subset=["id_registro"])
```

### Datos categóricos inconsistentes

```python
# Explorar valores únicos de una columna categórica
print(df["ciudad"].unique())

# Normalizar texto: minúsculas y espacios
df["ciudad"] = df["ciudad"].str.strip().str.lower()

# Reemplazar variantes específicas
df["ciudad"] = df["ciudad"].replace(
    {"bcn": "barcelona", "mad": "madrid"}
)
```

---

## Validación de tipos de datos y formatos

Es habitual que números se carguen como texto o fechas como cadenas. Validar y corregir tipos es clave para poder operar con los datos.

### Conversión de tipos

```python
# Ver tipos actuales
print(df.dtypes)

# Convertir a numérico (forzando errores a NaN)
df["columna_numerica"] = pd.to_numeric(df["columna_numerica"], errors="coerce")

# Convertir a fecha
df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d", errors="coerce")
```

### Validación básica de formatos

```python
# Comprobar si hay valores no convertibles (NaT o NaN creados al convertir)
fechas_invalidas = df[df["fecha"].isna()]
print(fechas_invalidas[["fecha"]])

valores_no_numericos = df[df["columna_numerica"].isna()]
print(valores_no_numericos[["columna_numerica"]])
```

Este tipo de inspección ayuda a localizar registros con formatos erróneos o datos corruptos.

---

## Exploración inicial de calidad de datos (EDA básico)

Antes de decidir cómo limpiar, es útil tener una visión general de la **calidad de los datos**.

```python
# Información general
df.info()

# Estadísticos descriptivos para numéricas
print(df.describe())

# Estadísticos para categóricas
print(df["columna_categorica"].value_counts())

# Distribución rápida de una variable numérica
df["columna_numerica"].hist(bins=30)
```

Algunas preguntas útiles:
- ¿Qué porcentaje de valores perdidos hay por columna?
- ¿Hay valores fuera de rangos razonables?
- ¿Hay categorías con muy pocos casos o inconsistentes?
- ¿Hay duplicados evidentes?

Estas observaciones guían las decisiones de limpieza.

---

## Ejercicio guiado

1. Cargar un dataset real (por ejemplo, de calidad del aire, viviendas, etc.) con problemas de calidad.
2. Realizar una **EDA básica**:
   - Revisar `info()`, `describe()`, `isna().sum()` y `value_counts()` para varias columnas.
3. Identificar:
   - Columnas con **valores perdidos** y decidir para cada una si eliminar filas o imputar, justificando la decisión.
   - Posibles **outliers** en al menos dos variables numéricas y decidir cómo tratarlos.
   - **Duplicados** y **valores categóricos inconsistentes**.
4. Corregir **tipos de datos y formatos** (por ejemplo, convertir fechas y números almacenados como texto).
5. Guardar el dataset limpio en un nuevo archivo (por ejemplo, `data/datos_limpios.csv`).
6. Documentar las decisiones de limpieza tomadas (por ejemplo, en un notebook o en un pequeño informe de texto).

---

## Conclusiones de la sesión

- La **limpieza de datos** es un paso imprescindible antes de cualquier análisis o modelado.
- La gestión adecuada de **valores perdidos**, **outliers**, **duplicados** y **tipos de datos** mejora la calidad y la fiabilidad de los resultados.
- La **exploración inicial de la calidad de datos** ayuda a priorizar qué problemas abordar y cómo hacerlo.
- Es fundamental **documentar todas las decisiones de limpieza** para garantizar la reproducibilidad y permitir que otras personas entiendan las transformaciones realizadas.

