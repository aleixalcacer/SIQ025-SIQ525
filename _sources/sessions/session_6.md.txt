# Sesión 6 — Visualización de datos (Parte 2)

## Objetivos

En esta sesión daremos un paso más en visualización de datos, centrándonos en gráficos **multivariables** y en la preparación de visualizaciones listas para informes.

Al finalizar la sesión, serás capaz de:
- Crear **visualizaciones multivariables complejas** usando faceting (`FacetGrid`, `PairGrid`) y combinaciones de gráficos.
- Utilizar **violin plots** y otras extensiones avanzadas para comparar distribuciones.
- Diseñar **gráficos orientados a comunicar resultados** de análisis.
- Explorar la **visualización interactiva** con Plotly (opcional).
- **Exportar gráficos** con calidad profesional para informes y presentaciones.

---

## Introducción

En la sesión anterior vimos cómo representar:

- Distribuciones (histogramas, densidad, boxplots).
- Relaciones simples entre dos variables (scatterplots).
- Correlaciones con mapas de calor.

En esta sesión queremos acercarnos más a lo que ocurre en proyectos reales:

- Explorar **varias variables a la vez** (por ejemplo, cómo cambia una relación según el día de la semana y el sexo).
- Crear gráficos pensados para un **informe** o una **presentación**.
- Dar un vistazo a la visualización **interactiva**, útil para dashboards y exploración.

Trabajaremos principalmente con `seaborn` y, de forma opcional, con `plotly.express`.

---

## Preparación: librerías y dataset de ejemplo

Usaremos de nuevo el dataset `tips` de seaborn, que ya conoces de la sesión anterior.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = sns.load_dataset("tips")
df.head()
```

Columnas relevantes:
- `total_bill`: total de la cuenta.
- `tip`: propina.
- `sex`, `smoker`, `day`, `time`, `size`, etc.

---

## Faceting y visualización multivariable (FacetGrid, relplot, PairGrid)

### ¿Qué es el faceting?

El **faceting** consiste en dividir un gráfico en **subgráficos (facetas)** según una o más variables categóricas.  
Ejemplos:

- Un scatterplot de `total_bill` vs `tip`, separado por día.
- Histogramas de una variable, uno por cada categoría de otra variable.

Esto permite ver **cómo cambia un patrón** en distintos subgrupos.

### FacetGrid básico

```python
g = sns.FacetGrid(df, col="day")
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip")
g.set_axis_labels("Total de la cuenta", "Propina")
g.fig.suptitle("Relación total_bill vs tip por día", y=1.02)
plt.show()
```

Explicación:
- `col="day"` → una columna de facetas por cada día.
- `map_dataframe` aplica `sns.scatterplot` sobre cada subconjunto de datos.

Podemos facetear por filas y columnas:

```python
g = sns.FacetGrid(df, row="time", col="smoker")
g.map_dataframe(sns.histplot, x="total_bill", bins=20)
g.set_axis_labels("Total de la cuenta", "Frecuencia")
g.fig.suptitle("Distribución de total_bill por time y smoker", y=1.02)
plt.show()
```

### Atajos con `relplot` y otros

`sns.relplot` es un atajo que crea un `FacetGrid` internamente:

```python
sns.relplot(
    data=df,
    x="total_bill",
    y="tip",
    hue="sex",
    col="day",
    kind="scatter",
    height=4
)
plt.show()
```

- `kind="scatter"` → scatterplot.
- `hue="sex"` → color por sexo.
- `col="day"` → una columna de facetas por día.

### PairGrid y pairplot

Para explorar **todas las combinaciones de variables numéricas** podemos usar `pairplot` (atajo de `PairGrid`):

```python
sns.pairplot(
    data=df,
    vars=["total_bill", "tip", "size"],
    hue="sex",
    diag_kind="kde"
)
plt.show()
```

Interpretación:
- Diagonal: distribuciones de cada variable.
- Fuera de la diagonal: scatterplots entre pares de variables.
- `hue="sex"` permite ver diferencias por grupo.

---

## Violin plots y combinaciones avanzadas

Los **violin plots** combinan la idea del boxplot con una estimación suave de la densidad (KDE). Son útiles para comparar distribuciones entre grupos de manera más detallada.

### Violin plot básico

```python
plt.figure(figsize=(6, 4))
sns.violinplot(
    data=df,
    x="day",
    y="total_bill"
)
plt.title("Distribución de total_bill por día (violin plot)")
plt.show()
```

Interpretación:
- La “forma” del violín indica dónde se concentran más los valores.
- Se puede ver si la distribución es unimodal, bimodal, etc.

### Combinar violin plot y boxplot

Podemos combinarlo con un **boxplot** o **swarmplot** para ver puntos individuales:

```python
plt.figure(figsize=(6, 4))
sns.violinplot(
    data=df,
    x="day",
    y="total_bill",
    inner=None,          # quitamos resumen interno
    color="lightgray"
)
sns.boxplot(
    data=df,
    x="day",
    y="total_bill",
    width=0.2
)
plt.title("Violin + Boxplot de total_bill por día")
plt.show()
```

O con `swarmplot`:

```python
plt.figure(figsize=(6, 4))
sns.violinplot(
    data=df,
    x="day",
    y="total_bill",
    inner=None,
    color="lightgray"
)
sns.swarmplot(
    data=df,
    x="day",
    y="total_bill",
    size=3,
    color="black"
)
plt.title("Violin + puntos individuales de total_bill por día")
plt.show()
```

Esta combinación es muy útil para **comunicar resultados**: se ve la forma global y también cada observación.

---

## Gráficos para comunicar resultados de análisis

Hasta ahora nos hemos centrado en **explorar**.  
Cuando queremos **comunicar** resultados (a profesores, clientes, equipo), conviene:

- Reducir el número de gráficos a los que realmente responden preguntas clave.
- Simplificar: menos es más.
- Cuidar:
  - Título: debe explicar qué se ve.
  - Ejes: nombres claros y unidades.
  - Leyendas: evitar duplicar información.
  - Notas o anotaciones: resaltar hallazgos importantes.

Ejemplo de gráfico “para informe”:

```python
plt.figure(figsize=(6, 4))
sns.regplot(
    data=df,
    x="total_bill",
    y="tip",
    scatter_kws={"alpha": 0.6},
    line_kws={"color": "red"}
)
plt.title("Relación entre total de la cuenta y propina")
plt.xlabel("Total de la cuenta ($)")
plt.ylabel("Propina ($)")
plt.tight_layout()
plt.show()
```

Ideas:
- Usar `regplot` para mostrar la tendencia.
- Ajustar `alpha` para que los puntos no saturen.
- Añadir comentarios en el texto del informe sobre:
  - Si la relación es aproximadamente lineal.
  - Si hay zonas donde la relación cambia.
  - Outliers llamativos.

---

## Visualización interactiva con Plotly (opcional)

Las visualizaciones interactivas permiten:

- Hacer zoom.
- Ver valores exactos al pasar el ratón.
- Activar/desactivar series.

`plotly.express` ofrece una API sencilla para crear gráficos interactivos en Jupyter.

### Instalación (si no lo tienes)

```bash
pip install plotly
```

### Ejemplo básico con scatter interactivo

```python
import plotly.express as px

fig = px.scatter(
    df,
    x="total_bill",
    y="tip",
    color="day",
    size="size",
    hover_data=["sex", "smoker", "time"],
    title="Relación total_bill vs tip (interactivo)"
)
fig.show()
```

Notas:
- `hover_data` muestra columnas adicionales al pasar el ratón.
- `size` controla el tamaño de los puntos según una variable numérica.

### Histograma interactivo

```python
fig = px.histogram(
    df,
    x="total_bill",
    color="day",
    nbins=20,
    barmode="overlay",
    opacity=0.7,
    title="Distribución de total_bill por día (interactivo)"
)
fig.show()
```

El uso de Plotly es opcional en la asignatura, pero útil si quieres explorar datos de forma más flexible o crear dashboards.

---

## Exportación de gráficos de calidad profesional

Una vez que tenemos gráficos útiles, necesitamos **exportarlos** para:

- Incluirlos en informes (Word, LaTeX, Markdown, PDF).
- Añadirlos a diapositivas (PowerPoint, Keynote).

### Exportar con matplotlib

```python
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="day", y="tip")
plt.title("Propina por día de la semana")
plt.tight_layout()

# Guardar en alta resolución
plt.savefig("results/propina_por_dia.png", dpi=300)
plt.savefig("results/propina_por_dia.pdf")
plt.close()
```

Recomendaciones:
- Usar `dpi=300` para impresión.
- Preferir formatos vectoriales (`.pdf`, `.svg`) cuando sea posible (para gráficos simples).

### Exportar con Plotly

```python
# Guardar como HTML interactivo
fig.write_html("results/scatter_total_bill_tip.html")

# Para exportar a PNG/SVG puede ser necesario instalar kaleido:
# pip install -U kaleido
fig.write_image("results/scatter_total_bill_tip.png", scale=2)
```

Guardar como HTML es muy útil para compartir gráficos interactivos (por ejemplo, por correo o en una intranet).

---

## Ejercicios sugeridos

1. **Faceting**:
   - Crear un `FacetGrid` que muestre la relación entre dos variables numéricas (por ejemplo, `total_bill` y `tip`) facetada por una o dos variables categóricas (`day`, `time`, `smoker`).
   - Interpretar qué cambia entre facetas.

2. **Visualización multivariable**:
   - Usar `pairplot` o `PairGrid` para explorar varias variables numéricas de un dataset y comentar las relaciones más evidentes.

3. **Violin plots y combinaciones**:
   - Crear violin plots comparando una variable numérica entre grupos (por ejemplo, `tip` por `day` o `sex`).
   - Combinar violin plot con boxplot o swarmplot para mostrar observaciones individuales.

4. **Plotly (opcional)**:
   - Crear al menos un gráfico interactivo (scatter o histograma) con `plotly.express`.
   - Probar a filtrar datos y observar cómo cambia la visualización.

5. **Informe visual**:
   - Preparar un pequeño conjunto de 3–5 gráficos que:
     - Resuman las distribuciones principales.
     - Muestren una o dos relaciones importantes.
   - Guardar los gráficos en formato `.png` o `.pdf` y anotar brevemente (en un notebook o documento) qué cuenta cada gráfico.

---

## Conclusiones de la sesión

- Las visualizaciones **multivariables** (faceting, PairGrid) permiten entender cómo cambian los patrones según distintos subgrupos.
- Los **violin plots** y combinaciones avanzadas ayudan a comparar distribuciones con más detalle.
- Diseñar gráficos pensando en la **comunicación** (títulos, ejes, leyendas) es tan importante como el código que los genera.
- La visualización **interactiva** con Plotly abre la puerta a exploración más flexible y a dashboards.
- Saber **exportar gráficos con buena calidad** es clave para integrarlos en informes y presentaciones profesionales.
