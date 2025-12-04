# Sesión 5 — Visualización de datos (Parte 1)

## Objetivos

En esta sesión aprenderemos a **representar datos gráficamente** de forma clara y profesional utilizando `matplotlib` y `seaborn`.

Al finalizar la sesión, serás capaz de:
- Construir **gráficos claros, informativos y significativos**.
- Utilizar `matplotlib` y `seaborn` para crear gráficos estáticos de alta calidad.
- Representar **distribuciones** (histogramas, densidad, boxplots).
- Analizar **relaciones entre variables** con scatterplots.
- Visualizar **correlaciones** con mapas de calor.
- Aplicar **paletas de color y estilos** profesionales en `seaborn`.

---

## Introducción: por qué visualizar datos

La visualización es una parte central del análisis de datos:

- Permite detectar **patrones**, **tendencias**, **outliers** y **errores** que no se ven en tablas.
- Ayuda a **comunicar resultados** a otras personas (no técnicas y técnicas).
- Complementa la estadística numérica: una media o una desviación estándar no cuentan toda la historia.

En esta sesión seguiremos principios básicos inspirados en Edward Tufte y otros autores de visualización efectiva.

---

## Principios de visualización efectiva (inspirados en Edward Tufte)

Al diseñar un gráfico, no solo importa el código, sino también **cómo** contamos la historia de los datos.

Algunas ideas clave:

- **Claridad por encima de todo**:
  - El gráfico debe responder una pregunta clara: “¿qué quiero que el lector vea?”.
- **Evitar el “chartjunk”**:
  - Quitar elementos innecesarios (fondos recargados, efectos 3D, decoraciones que no aportan información).
- **Maximizar la relación “tinta/datos”**:
  - La mayor parte de los elementos del gráfico deben representar datos, no ruido visual.
- **Elegir el tipo de gráfico adecuado**:
  - Histogramas / densidad / boxplots → distribuciones.
  - Scatterplots → relación entre dos variables.
  - Mapas de calor → matrices de correlación, intensidad.
- **Usar escalas y ejes honestos**:
  - No truncar ejes ni cambiar escalas sin avisar.
- **Cuidar colores y etiquetas**:
  - Paletas que se vean bien en pantalla y en impresión.
  - Etiquetas de ejes, títulos y leyendas claros y concisos.

---

## Preparación: importar librerías y cargar datos de ejemplo

En la mayoría de ejemplos usaremos `pandas`, `matplotlib` y `seaborn`.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo básico de seaborn
sns.set(style="whitegrid")

# Cargar un dataset de ejemplo de seaborn
df = sns.load_dataset("tips")  # datos de propinas en un restaurante
df.head()
```

`tips` incluye columnas como:
- `total_bill`: importe total de la cuenta.
- `tip`: cantidad de la propina.
- `sex`, `smoker`, `day`, `time`, `size`, etc.

---

## Gráficos de distribución: histogramas, densidad y boxplots

### Histogramas

Los histogramas muestran cómo se distribuyen los valores de una variable numérica.

```python
plt.figure(figsize=(6, 4))
plt.hist(df["total_bill"], bins=20, edgecolor="black")
plt.xlabel("Total de la cuenta")
plt.ylabel("Frecuencia")
plt.title("Histograma de total_bill")
plt.show()
```

Con `seaborn`:

```python
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="total_bill", bins=20, kde=False)
plt.title("Histograma de total_bill (seaborn)")
plt.show()
```

### Densidad (KDE: Kernel Density Estimate)

Es una versión suavizada del histograma, útil para ver la forma general de la distribución.

```python
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="total_bill", fill=True)
plt.title("Densidad de total_bill")
plt.show()
```

Podemos comparar distribuciones entre grupos:

```python
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="total_bill", hue="day", fill=True, common_norm=False)
plt.title("Densidad de total_bill por día")
plt.show()
```

### Boxplots

Los boxplots (diagramas de caja) resumen la distribución y muestran outliers de forma compacta.

```python
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="day", y="total_bill")
plt.title("Distribución de total_bill por día (boxplot)")
plt.show()
```

Interpretación:
- La caja representa el rango intercuartílico (Q1–Q3).
- La línea central es la mediana.
- Los puntos fuera de los “bigotes” se consideran posibles **outliers**.

---

## Scatterplots y relaciones entre variables

Los scatterplots (diagramas de dispersión) muestran la relación entre dos variables numéricas.

```python
plt.figure(figsize=(6, 4))
plt.scatter(df["total_bill"], df["tip"])
plt.xlabel("Total de la cuenta")
plt.ylabel("Propina")
plt.title("Relación entre total_bill y tip")
plt.show()
```

Con `seaborn`:

```python
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="total_bill", y="tip")
plt.title("Relación entre total_bill y tip (seaborn)")
plt.show()
```

Podemos añadir información categórica mediante color (`hue`) o forma (`style`):

```python
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip",
    hue="time",       # comida / cena
    style="smoker"    # fumador / no fumador
)
plt.title("Relación total_bill vs tip, por time y smoker")
plt.show()
```

Consejos:
- Usar transparencia (`alpha`) cuando hay muchos puntos solapados.
- Comprobar si la relación parece lineal, curvada, con outliers claros, etc.

---

## Matrices de correlación y mapas de calor

Para ver cómo se relacionan **varias** variables numéricas entre sí, calculamos la **matriz de correlación** y la visualizamos con un **mapa de calor (heatmap)**.

```python
# Seleccionar solo columnas numéricas
num_cols = df.select_dtypes(include="number").columns
corr = df[num_cols].corr()

print(corr)
```

Visualización con `seaborn`:

```python
plt.figure(figsize=(6, 4))
sns.heatmap(
    corr,
    annot=True,      # muestra el valor numérico
    fmt=".2f",       # formato de los números
    cmap="coolwarm", # paleta de color
    vmin=-1, vmax=1  # rango de la correlación
)
plt.title("Matriz de correlación")
plt.show()
```

Interpretación básica:
- Valores cercanos a **1** → fuerte correlación positiva.
- Valores cercanos a **-1** → fuerte correlación negativa.
- Valores cercanos a **0** → poca relación lineal.

La matriz de correlación ayuda a:
- Detectar variables muy correlacionadas (posible redundancia).
- Intuir relaciones interesantes a explorar con otros gráficos.

---

## Paletas de color y estilos profesionales en Seaborn

`seaborn` facilita definir estilos y paletas de color coherentes, lo que ayuda a:

- Mejorar la legibilidad del gráfico.
- Mantener una estética profesional y consistente en todo el informe.

### Estilos globales

```python
sns.set_theme(style="whitegrid")  # otros: "darkgrid", "white", "dark", "ticks"
```

También podemos cambiar el contexto (tamaño de fuentes):

```python
sns.set_context("talk")   # otros: "paper", "notebook", "poster"
```

### Paletas de color

```python
# Paleta categórica
sns.set_palette("Set2")

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="day", y="total_bill")
plt.title("Boxplot con paleta Set2")
plt.show()
```

Algunas paletas útiles:
- Categóricas: `"Set1"`, `"Set2"`, `"Paired"`, `"tab10"`.
- Secuenciales: `"Blues"`, `"Greens"`, `"Oranges"`.
- Divergentes: `"coolwarm"`, `"RdBu"`, `"Spectral"`.

También se pueden crear paletas manuales:

```python
custom_palette = ["#1b9e77", "#d95f02", "#7570b3"]
sns.set_palette(custom_palette)
```

### Ajustes finos con matplotlib

Aunque usemos `seaborn`, muchas veces queremos personalizar detalles con `matplotlib`:

```python
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="total_bill", y="tip", hue="day")
plt.title("Relación total_bill vs tip por día")
plt.xlabel("Total de la cuenta (€)")
plt.ylabel("Propina (€)")
plt.legend(title="Día de la semana", loc="best")
plt.tight_layout()
plt.show()
```

---

## Ejercicios sugeridos

1. Cargar un dataset (puede ser el de `tips` u otro, por ejemplo, un dataset de viviendas).
2. Realizar un **EDA visual**:
   - Graficar histogramas y curvas de densidad para varias variables numéricas.
   - Crear boxplots para comparar distribuciones entre grupos categóricos (por ejemplo, día de la semana, tipo de vivienda).
3. Crear **scatterplots** para analizar la relación entre:
   - Dos variables numéricas principales (por ejemplo, precio y superficie, total_bill y tip).
   - Añadiendo color (`hue`) y estilo (`style`) según una variable categórica.
4. Calcular la **matriz de correlación** de las variables numéricas y representarla con un mapa de calor (`heatmap`).
5. Probar **diferentes paletas de color y estilos** (`set_theme`, `set_palette`) y elegir una combinación que:
   - Sea clara y legible.
   - No abuse de colores llamativos.
   - Sea adecuada para un informe profesional.

---

## Conclusiones de la sesión

- La visualización es esencial para **entender y comunicar** los datos.
- `matplotlib` y `seaborn` permiten construir gráficos estáticos de alta calidad con pocas líneas de código.
- Hemos visto cómo representar **distribuciones**, **relaciones entre variables** y **correlaciones**.
- Las **paletas de color** y los **estilos** influyen mucho en la claridad del mensaje.
- En sesiones posteriores profundizaremos en visualizaciones más avanzadas y en cómo integrarlas en informes y dashboards.
