# Sesión 4 — Entrenamiento de Modelos y Evaluación

## Introducción
Una vez que los datos están preprocesados y listos, podemos pasar a la fase de **entrenamiento de modelos**. Esta etapa consiste en aplicar algoritmos de machine learning para que aprendan patrones a partir de los datos y puedan hacer predicciones sobre datos nuevos.

No todos los modelos funcionan de la misma manera, por lo que es fundamental **evaluar su desempeño** mediante métricas específicas y técnicas de validación. Esta sesión se centra en cómo entrenar modelos básicos, medir su precisión y preparar los resultados de manera reproducible.

---

## Dividir los datos en entrenamiento y prueba
Antes de entrenar un modelo, es importante separar los datos en conjuntos de **entrenamiento** y **prueba**, para poder evaluar cómo se comporta el modelo con datos no vistos.

```python
from sklearn.model_selection import train_test_split

X = df_preprocessed  # variables de entrada preprocesadas
y = df['objetivo']  # variable objetivo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Esta división permite medir la capacidad de generalización del modelo y evita sobreajuste.

---

## Entrenamiento de modelos básicos
### Regresión lineal
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predicciones = model.predict(X_test)
```

### Regresión logística
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predicciones = model.predict(X_test)
```

### Árbol de decisión
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predicciones = model.predict(X_test)
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
predicciones = model.predict(X_test)
```

Cada algoritmo tiene ventajas y limitaciones. La elección depende del tipo de problema, la cantidad de datos y la complejidad de las relaciones.

---

## Evaluación de modelos
Evaluar un modelo es fundamental para entender su desempeño y compararlo con otros modelos.

### Métricas para clasificación
- **Accuracy**: proporción de predicciones correctas.
- **Recall**: proporción de positivos reales correctamente identificados.
- **F1-score**: media armónica de precision y recall.

```python
from sklearn.metrics import accuracy_score, recall_score, f1_score

accuracy = accuracy_score(y_test, predicciones)
recall = recall_score(y_test, predicciones, average='macro')
f1 = f1_score(y_test, predicciones, average='macro')

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
```

### Métricas para regresión
- **MAE**: error absoluto medio.
- **MSE**: error cuadrático medio.
- **R2**: coeficiente de determinación.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predicciones)
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print(f'MAE: {mae}, MSE: {mse}, R2: {r2}')
```

Estas métricas permiten cuantificar la precisión de los modelos y orientar mejoras.

---

## Validación cruzada
La **validación cruzada** permite evaluar el modelo de manera más robusta dividiendo los datos en múltiples subconjuntos y calculando el desempeño promedio.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print("Puntajes de cada fold:", scores)
print("Media:", scores.mean())
```
Esta técnica reduce el riesgo de que los resultados dependan de una única división de los datos.

---

## Ejercicio guiado
1. Tomar el dataset preprocesado de la sesión anterior.
2. Dividir los datos en entrenamiento y prueba.
3. Entrenar al menos dos modelos diferentes (por ejemplo, regresión logística y Random Forest).
4. Evaluar los modelos usando métricas adecuadas según el tipo de problema.
5. Aplicar validación cruzada y comparar los resultados.
6. Analizar cuál modelo parece más robusto y por qué.

---

## Conclusiones de la sesión
- Entrenar modelos requiere separar datos en entrenamiento y prueba para evaluar correctamente la capacidad de generalización.
- Diferentes algoritmos ofrecen distintas ventajas; la elección depende del problema y del tipo de datos.
- Evaluar el modelo con métricas adecuadas y usar validación cruzada permite medir desempeño de forma objetiva.
- Estos conceptos preparan a los alumnos para sesiones más avanzadas de optimización y despliegue de modelos.

