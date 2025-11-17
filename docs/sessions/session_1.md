# Sesión 1 — Introducción al Análisis de Datos con Python


## Objetivos

En esta sesión aprenderemos los conceptos básicos del análisis de datos y cómo usar Python y Jupyter Notebook para realizar análisis sencillos. Al finalizar la sesión, serás capaz de:
- Entender el **flujo general del análisis de datos**: recolección, limpieza, exploración, modelado y comunicación.
- Utilizar **Jupyter Notebook** para documentar y ejecutar análisis de datos de manera interactiva.
- Conocer los **tipos de datos fundamentales** en Python y las **estructuras básicas** como listas y diccionarios.

## Introducción
El análisis de datos se ha convertido en una habilidad esencial en múltiples disciplinas, desde la investigación científica hasta la toma de decisiones empresariales. La capacidad de recolectar, limpiar, explorar, modelar y comunicar datos de manera efectiva permite extraer información valiosa que de otro modo permanecería oculta en grandes volúmenes de información. En esta asignatura aprenderemos a usar herramientas profesionales para realizar análisis de datos de manera rigurosa y reproducible.

Python es uno de los lenguajes más usados en el análisis de datos. Es muy versátil y su ecosistema de librerías científicas permite realizar desde la manipulación básica de datos hasta modelos avanzados de machine learning.

En esta primera sesión nos centraremos en Python y en el entorno Jupyter Notebook, que será la herramienta principal de todo el curso. Aprenderemos los conceptos básicos del lenguaje, la manipulación de estructuras de datos y los primeros pasos en pandas y NumPy, fundamentales para cualquier análisis posterior.

---

## Software y estructura de proyecto
**Jupyter Notebook** es un entorno interactivo que permite combinar código, resultados, gráficos y texto explicativo en un único documento. Esto facilita la comprensión del análisis paso a paso y permite documentar todo el proceso de manera clara y reproducible.

**Google Colab** es una plataforma en línea que permite ejecutar Jupyter Notebooks sin necesidad de instalar nada localmente, facilitando el acceso y la colaboración.

Una buena práctica profesional es mantener una **estructura de proyecto organizada**, de manera que los datos, scripts, notebooks y resultados estén claramente separados. Una estructura recomendada podría ser:
```
data/       → datasets utilizados en el análisis
notebooks/  → cuadernos Jupyter donde se desarrolla el análisis
src/        → scripts de Python para funciones o cálculos repetitivos
results/    → gráficos, tablas y reportes exportados
README.md   → descripción general del proyecto y documentación
```
Mantener esta estructura desde el inicio es clave para garantizar la reproducibilidad y la claridad del proyecto.

---

## Conceptos básicos de Python
Antes de empezar a manipular datos, es importante conocer los tipos de datos fundamentales y cómo interactuar con ellos.

### Tipos de datos
- **int**: números enteros.
- **float**: números decimales.
- **str**: cadenas de texto.
- **bool**: valores lógicos `True` o `False`.

Ejemplo:
```python
x = 5        # número entero
pi = 3.14    # número decimal
nombre = "Ana" # cadena de texto
activo = True  # valor booleano
```
Estos tipos de datos se usan para almacenar información y realizar operaciones básicas, y constituyen la base para trabajar con estructuras más complejas.

### Operadores
Python permite realizar operaciones aritméticas, comparaciones y combinaciones lógicas:
- **Aritméticos**: `+ - * / ** %` (suma, resta, multiplicación, división, potencia y módulo).
- **Comparación**: `== != > < >= <=` (igual, diferente, mayor, menor, mayor o igual, menor o igual).
- **Lógicos**: `and`, `or`, `not` para combinar condiciones booleanas.

### Estructuras de datos
- **Listas**: colecciones ordenadas y modificables.
```python
numeros = [1, 2, 3]
numeros.append(4)  # añadir un elemento
```
- **Diccionarios**: colecciones de pares clave-valor.
```python
persona = {"nombre": "Ana", "edad": 21}
print(persona["nombre"])  # acceder al valor de la clave 'nombre'
```
Estas estructuras permiten almacenar y organizar datos antes de pasarlos a librerías como pandas para su análisis.

---

## Introducción a NumPy y pandas
**NumPy** es una librería especializada en cálculos numéricos eficientes sobre arrays y matrices. Facilita operaciones matemáticas complejas de manera rápida y sencilla.

**pandas** permite manipular datos tabulares mediante **DataFrames**, estructuras similares a hojas de cálculo pero mucho más flexibles y potentes. Nos permite filtrar, seleccionar, agregar y transformar datos de manera profesional.

Ejemplo básico:
```python
import pandas as pd
import numpy as np

# Lista de números y cálculo de media
numeros = [2, 4, 6, 8, 10]
media = np.mean(numeros)
print("Media:", media)

# Crear un DataFrame simple
data = {
    "Nombre": ["Ana", "Luis", "Marta"],
    "Nota": [7.5, 8.0, 9.0]
}
df = pd.DataFrame(data)
print(df)
print("Nota media:", df["Nota"].mean())
```

Con este ejemplo los alumnos comienzan a ver cómo Python y pandas permiten calcular métricas básicas y organizar la información de manera eficiente.

---

## Ejercicio guiado
1. Crear una lista de 10 números y calcular su media manualmente.
2. Convertir la lista en un array de NumPy y calcular la media con `.mean()`.
3. Crear un DataFrame con nombres y notas de 5 alumnos.
4. Calcular la nota media usando pandas.

Estos ejercicios permiten aplicar los conceptos aprendidos y familiarizarse con la sintaxis y las herramientas que se usarán durante todo el curso.

---

## Conclusiones de la sesión
- El análisis de datos consiste en recolectar, limpiar, explorar, modelar y comunicar datos.
- Python y Jupyter Notebook permiten desarrollar análisis completos de manera interactiva y reproducible.
- Hemos visto los tipos de datos básicos, estructuras fundamentales y los primeros pasos con pandas y NumPy.
- En la próxima sesión se profundizará en la **exploración de datos (EDA)** y en la **limpieza de datos** para preparar datasets para análisis más avanzados.

