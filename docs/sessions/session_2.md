# Sesión 2 — Importación y manipulación de datos

## Objetivos

En esta sesión nos centraremos en **manipular datos con pandas**, aprendiendo a trabajar de forma eficiente con DataFrames y a aplicar transformaciones habituales en análisis de datos reales. Los objetivos principales son:
- Dominar la estructura de datos principal de pandas: el **DataFrame**.
- Realizar **transformaciones** y operaciones comunes en análisis de datos reales.
- Practicar **importación**, **filtrado**, **selección**, **ordenación** y **agregaciones por grupos**.

---

## Qué es un DataFrame

Un **DataFrame** es una estructura de datos bidimensional similar a una tabla, donde los datos están organizados en filas y columnas. Cada columna puede tener un tipo de dato diferente (números, texto, fechas, etc.), lo que lo hace muy flexible para manejar datos heterogéneos.

Es similar a una hoja de cálculo o una tabla SQL, y es la estructura principal que utilizaremos para almacenar y manipular datos. Los DataFrames permiten realizar operaciones complejas de manera eficiente y con una sintaxis intuitiva.

## Importación de datos (CSV, Excel)

Para comenzar, necesitamos importar los datos a un DataFrame de pandas. Los datos pueden provenir de archivos CSV o Excel, entre otros formatos.

```python
import pandas as pd

# Cargar un archivo CSV
df_csv = pd.read_csv('data/ejemplo.csv')

# Cargar un archivo de Excel
df_excel = pd.read_excel('data/ejemplo.xlsx', sheet_name='Hoja1')

# Mostrar las primeras filas
df_csv.head()
```

---

## Estructura básica de un DataFrame

pandas ofrece múltiples métodos para inspeccionar rápidamente un DataFrame y entender su estructura:

```python
print(df_csv.shape)      # Filas y columnas
print(df_csv.columns)    # Nombres de columnas
df_csv.info()            # Tipos de datos y nulos
df_csv.head()            # Primeras filas
df_csv.tail()            # Últimas filas
```

---

## Filtrado y selección de datos

Seleccionar filas y columnas es una operación central en la manipulación de datos.

```python
# Seleccionar una columna
col = df_csv['columna1']

# Seleccionar varias columnas
sub_df = df_csv[['columna1', 'columna2']]

# Filtrar filas según condición
df_filtrado = df_csv[df_csv['columna_numerica'] > 10]

# Varias condiciones (AND / OR)
df_filtrado = df_csv[
    (df_csv['columna1'] > 10) & (df_csv['columna2'] == 'valor')
]
```

### Selección por etiquetas e índices: loc e iloc

```python
# Selección por etiqueta (nombre de fila/columna)
df_loc = df_csv.loc[0:10, ['columna1', 'columna2']]

# Selección por posición (índices enteros)
df_iloc = df_csv.iloc[0:10, 0:2]
```

---

## Ordenación de datos

Ordenar datos es útil para priorizar o revisar registros específicos.

```python
# Ordenar por una columna ascendente
df_ordenado = df_csv.sort_values(by='columna_numerica')

# Ordenar por varias columnas y orden descendente
df_ordenado = df_csv.sort_values(
    by=['columna_categoria', 'columna_numerica'],
    ascending=[True, False]
)
```

---

## Transformaciones comunes en DataFrames

Algunas transformaciones típicas en análisis reales incluyen crear nuevas columnas, limpiar valores o cambiar tipos.

```python
# Crear una nueva columna a partir de otras
df_csv['ratio'] = df_csv['columna_a'] / df_csv['columna_b']

# Reemplazar valores específicos
df_csv['columna_categoria'] = df_csv['columna_categoria'].replace(
    {'antiguo': 'nuevo'}
)

# Conversión de tipos
df_csv['columna_numerica'] = pd.to_numeric(
    df_csv['columna_numerica'],
    errors='coerce'
)
```

---

## Agregaciones y estadísticas por grupos (groupby)

El método `groupby` permite calcular estadísticas por grupos, algo muy usado en análisis reales (por ejemplo, ventas por región, media por categoría, etc.).

```python
# Agrupar por una columna y calcular la media
media_por_grupo = df_csv.groupby('categoria')['valor'].mean()

# Varios agregados por grupo
agg_por_grupo = df_csv.groupby('categoria').agg({
    'valor': ['mean', 'sum', 'count'],
    'otra_columna': 'max'
})

# Agrupar por varias columnas
multi_grupo = df_csv.groupby(['categoria', 'subcategoria'])['valor'].sum()
```

También podemos resetear el índice para obtener un DataFrame “plano”:

```python
agg_por_grupo = agg_por_grupo.reset_index()
```

---

## Ejercicio guiado

1. Importar un dataset desde un archivo CSV (y, si es posible, otro desde Excel).
2. Explorar la estructura: usar `shape`, `columns`, `info()` y `head()`.
3. Aplicar filtrados: seleccionar filas según una o varias condiciones.
4. Seleccionar subconjuntos de columnas y ordenar el DataFrame por una o varias columnas.
5. Crear al menos una nueva columna derivada de otras.
6. Utilizar `groupby` para calcular estadísticas (media, suma, conteo) por una columna categórica.

---

## Conclusiones de la sesión

- La **manipulación de datos con pandas** se basa en dominar los DataFrames.
- Saber **importar**, **filtrar**, **seleccionar**, **ordenar** y **agrupar** datos es esencial para cualquier análisis real.
- El método `groupby` permite obtener estadísticas por grupos de forma clara y eficiente.
- Estas operaciones son la base sobre la que construiremos análisis más avanzados y modelos en sesiones posteriores.
