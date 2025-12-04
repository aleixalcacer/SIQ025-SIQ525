# Sesión 8 — Comunicación y presentación de resultados

## Objetivos

En esta sesión trabajaremos cómo **contar la historia de los datos** de forma clara y profesional.

Al finalizar la sesión, serás capaz de:
- **Presentar un análisis** de forma estructurada y comprensible para distintos públicos.
- Preparar **informes reproducibles y visualmente atractivos** a partir de notebooks.
- **Exportar gráficos y tablas** con calidad suficiente para informes y presentaciones.
- Aplicar **buenas prácticas de documentación** de código y análisis.
- Tener una primera idea de cómo crear **mini‑apps interactivas** con Streamlit (opcional).

---

## Introducción

Hemos dedicado varias sesiones a:

- Cargar, limpiar y transformar datos.
- Visualizar distribuciones y relaciones.
- Entrenar y evaluar modelos.

Todo esto pierde valor si no sabemos **comunicar los resultados** de forma efectiva. En muchos proyectos, la “entrega” final no es el notebook ni el modelo, sino:

- Un **informe** (PDF, HTML, documento).
- Una **presentación**.
- Un **dashboard** o pequeña aplicación interactiva.

En esta sesión nos centraremos en cómo pasar de un notebook de trabajo a un **producto final comprensible y reproducible**.

---

## Cómo estructurar un informe analítico claro y reproducible

Un buen informe no es solo una colección de gráficos; debe seguir una **narrativa**. Una estructura típica:

1. **Introducción / Contexto**
   - ¿Qué problema queremos resolver?
   - ¿De dónde vienen los datos?
   - ¿Qué preguntas queremos contestar?

2. **Datos y metodología**
   - Descripción básica del dataset (origen, tamaño, variables importantes).
   - Pasos principales de limpieza y preprocesamiento.
   - Modelos o técnicas utilizadas.

3. **Resultados**
   - Gráficos clave de EDA (distribuciones, relaciones importantes).
   - Resultados de modelos (métricas, comparaciones).
   - Segmentaciones o clusters, si aplica.

4. **Conclusiones e implicaciones**
   - Resumen de hallazgos.
   - Recomendaciones o decisiones posibles.
   - Limitaciones del análisis (datos, supuestos, etc.).

5. **Anexos (opcional)**
   - Detalles técnicos adicionales.
   - Tablas largas, código más avanzado.

### Notebooks como base del informe

En un Jupyter Notebook podemos reflejar esa estructura usando:

- **Títulos y subtítulos** (Markdown: `#`, `##`, `###`).
- Texto explicativo en lenguaje natural.
- Celdas de código bien comentadas.
- Gráficos y tablas integrados en el flujo.

Ejemplo de cabecera de informe en un notebook (celda Markdown):

```markdown
# Análisis de la calidad del aire en Ciudad X

## 1. Introducción
En este informe analizamos los niveles de contaminación en Ciudad X durante el año 2024...
```

---

## Exportación de gráficos y tablas de calidad

A lo largo del curso hemos creado gráficos con `matplotlib` y `seaborn`. Para usarlos en informes:

### Guardar gráficos con matplotlib

```python
import matplotlib.pyplot as plt
import seaborn as sns

# ... crear gráfico ...
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="variable", bins=30)
plt.title("Distribución de 'variable'")
plt.tight_layout()

# Guardar el gráfico en alta resolución
plt.savefig("results/distribucion_variable.png", dpi=300)
plt.savefig("results/distribucion_variable.pdf")
plt.close()
```

Recomendaciones:

- Usar `dpi=300` para calidad de impresión.
- Preferir formatos **vectoriales** (`.pdf`, `.svg`) cuando el destino es impresión o zoom alto.
- Asegurar:
  - Títulos claros.
  - Ejes etiquetados con unidades.
  - Leyendas si hay varias series.

### Exportar tablas con pandas

Podemos exportar tablas resumen a distintos formatos:

```python
# Tabla resumen
tabla_resumen = df.groupby("categoria")["valor"].agg(["mean", "sum", "count"])
tabla_resumen.to_csv("results/tabla_resumen.csv", index=True)

# También a Excel
tabla_resumen.to_excel("results/tabla_resumen.xlsx", sheet_name="Resumen")
```

En informes:

- Incluir solo las **tablas clave**.
- Evitar tablas enormes; para detalles, usar anexos o archivos adjuntos (CSV/Excel).

---

## De Notebooks a HTML/PDF

Una ventaja de Jupyter es que podemos convertir un notebook en un **documento estático** (HTML, PDF) sin esfuerzo manual.

### Conversión con la interfaz de Jupyter / Colab

- En Jupyter clásico:  
  `File` → `Download as` → `HTML (.html)` o `PDF via LaTeX`.
- En Google Colab:  
  `File` → `Download` → `Download .ipynb` o “Print” para generar PDF.

### Conversión con nbconvert (línea de comandos)

Si trabajas localmente:

```bash
jupyter nbconvert \
    --to html \
    --output informe_eda.html \
    notebooks/eda_calidad_aire.ipynb
```

Para PDF (requiere tener LaTeX instalado):

```bash
jupyter nbconvert \
    --to pdf \
    --output informe_eda.pdf \
    notebooks/eda_calidad_aire.ipynb
```

Buenas prácticas antes de exportar:

- Ejecutar **todo el notebook de arriba a abajo** (`Restart & Run All`) para asegurar reproducibilidad.
- Limpiar celdas innecesarias (pruebas, salidas muy largas).
- Comprobar que no quedan **errores** en celdas.

---

## Buenas prácticas de documentación de código y análisis

Para que tu trabajo lo entienda otra persona (o tú mismo en unas semanas), conviene:

### Comentarios y nombres claros

- Usar **nombres descriptivos** para variables y funciones:
  - `df_clientes`, `media_propina`, `predicciones_test`…
- Añadir comentarios breves donde la lógica no sea obvia:

```python
# Filtramos solo clientes activos en 2024
df_activos = df[df["anio"] == 2024].copy()
```

### Separar celdas lógicas

- Una celda = una “idea”:
  - Cargar datos.
  - Limpieza básica.
  - Visualización concreta.
- Evitar celdas enormes con muchos pasos mezclados.

### Documentar decisiones

Usar Markdown para justificar:

- Por qué se eliminan ciertas filas o columnas.
- Por qué se elige una métrica en lugar de otra.
- Supuestos importantes (por ejemplo, “ignoramos valores anteriores a 2010 por cambios en el sistema de medición”).

Ejemplo (celda Markdown):

```markdown
### Decisión: tratar valores por encima de 500 como outliers

Los sensores tienen un rango máximo de 500. Cualquier valor por encima probablemente
se deba a errores de calibración, por lo que se recortan a 500 (winsorización).
```

---

## Introducción a Streamlit (opcional)

**Streamlit** permite convertir scripts de Python en **aplicaciones web interactivas** de forma muy sencilla. Es útil para:

- Crear pequeños **dashboards**.
- Permitir que usuarios no técnicos exploren modelos o resultados.

### Instalación

```bash
pip install streamlit
```

### Ejemplo mínimo de app

Crea un archivo `app.py`:

```python
import streamlit as st
import pandas as pd
import seaborn as sns

st.title("Dashboard simple de propinas")

df = sns.load_dataset("tips")

st.write("Vista previa de los datos:")
st.dataframe(df.head())

variable = st.selectbox("Elige una variable numérica", ["total_bill", "tip", "size"])

st.write(f"Histograma de {variable}:")
st.bar_chart(df[variable])
```

Ejecutar la app:

```bash
streamlit run app.py
```

Esto abrirá una ventana en el navegador con:

- Un título.
- Una tabla interactiva.
- Un selector de variable y un gráfico que cambia según la selección.

Aunque Streamlit es opcional en este curso, ofrece una vía directa para **presentar resultados de forma interactiva** sin tener que desarrollar una web desde cero.

---

## Ejercicios sugeridos

1. **Informe de EDA**:
   - Tomar un dataset (por ejemplo, calidad del aire, viviendas, tips).
   - Crear un notebook estructurado como informe:
     - Introducción y objetivos.
     - Descripción de datos.
     - EDA visual (gráficos de distribución y relaciones).
     - Conclusiones principales.
   - Usar títulos Markdown y texto explicativo.

2. **Exportar notebook a HTML**:
   - Asegurarse de que el notebook se ejecuta de principio a fin sin errores.
   - Exportarlo a formato **HTML** (desde la interfaz o con `nbconvert`).
   - Verificar que los gráficos y tablas se ven correctamente en el HTML generado.

3. **Exportación de gráficos y tablas**:
   - Guardar al menos:
     - Un gráfico en `.png` y `.pdf`.
     - Una tabla resumen en `.csv` o `.xlsx`.
   - Insertar el gráfico en un documento (por ejemplo, un informe en Word, LaTeX o Markdown).

4. **(Opcional) Mini‑dashboard con Streamlit**:
   - Crear una pequeña app que:
     - Cargue un dataset.
     - Permita seleccionar una variable numérica y muestre un gráfico (histograma, boxplot, etc.).
   - Ejecutar la app localmente y explorarla.

---

## Conclusiones de la sesión

- La **comunicación de resultados** es una parte esencial del trabajo en análisis de datos.
- Un buen informe combina:
  - Estructura clara.
  - Texto explicativo.
  - Gráficos y tablas bien seleccionados y legibles.
- Los notebooks permiten crear análisis **reproducibles** que luego podemos exportar a HTML o PDF.
- Las **buenas prácticas de documentación** (nombres claros, comentarios, decisiones justificadas) facilitan la colaboración y el mantenimiento.
- Herramientas como **Streamlit** permiten dar un paso más y convertir análisis en **aplicaciones interactivas**, acercando los resultados a usuarios no técnicos.
