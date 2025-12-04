# Sesión 9 (Bonus) — Control de versiones con Git

## Objetivos

En esta sesión introductoria veremos cómo usar **Git** para versionar proyectos de análisis de datos.

Al finalizar la sesión, serás capaz de:
- Entender qué es el **control de versiones** y por qué es clave para la **reproducibilidad**.
- Crear y utilizar un **repositorio local** con Git.
- Registrar cambios mediante **commits** y revisar el **historial**.
- Usar los comandos básicos de Git en el contexto de proyectos de datos.

---

## Introducción: versionado y reproducibilidad

A lo largo del curso has trabajado con:

- Notebooks (`.ipynb`) con código, gráficos y texto.
- Datos (`data/`), resultados (`results/`), scripts (`src/`).
- Pipelines de preprocesamiento y modelos.

Para que un proyecto sea **reproducible** y fácil de mantener, no basta con guardar archivos sueltos:

- Necesitamos saber **qué cambió, cuándo y por qué**.
- Necesitamos poder **volver atrás** si algo se rompe.
- Necesitamos compartir el proyecto con otras personas sin perder el historial.

Aquí entra Git, un **sistema de control de versiones** que nos permite:

- Guardar “fotografías” del estado del proyecto (commits).
- Comparar versiones.
- Trabajar en equipo sin pisarnos el trabajo.

---

## Concepto básico: repositorios, commits e historial

### Repositorio

Un **repositorio Git** es una carpeta de proyecto con un historial de cambios asociado.

- Normalmente coincide con la raíz de tu proyecto:
  - `data/`, `notebooks/`, `src/`, `results/`, etc.
- Git guarda su información en una subcarpeta oculta llamada `.git`.

```
my_project/
├── data/
├── notebooks/
├── src/
├── results/
└── **.git/**
```

### Commits

Un **commit** es como una “foto” del estado de tus archivos en un momento concreto, junto con:

- Un identificador único.
- Un **mensaje** descriptivo.
- La información de autor y fecha.

Ejemplo de mensajes de commit:

- `Añade EDA inicial de calidad del aire`
- `Corrige imputación de valores perdidos en columna PM10`

### Historial

El **historial de commits** te permite:

- Ver la evolución del proyecto.
- Saber qué cambió entre versiones.
- Recuperar un estado anterior si algo sale mal.

---

## Crear y preparar un repositorio local

Supongamos que tu proyecto está en:

```text
/Users/aalcacer/siq025/
```

### Inicializar un repositorio

En la terminal:

```bash
cd /Users/aalcacer/siq025
git init
```

Esto crea la carpeta oculta `.git` y convierte el directorio actual en un repositorio Git.

### Comprobar el estado

```bash
git status
```

Muestra:

- Archivos no rastreados (new files).
- Archivos modificados.
- Archivos listos para commit (staged).

---

## Flujo básico: add → commit → log

### 1. Añadir archivos al área de preparación (staging area)

Primero decidimos qué cambios queremos incluir en el próximo commit:

```bash
# Añadir un archivo concreto
git add README.md

# Añadir varios archivos
git add notebooks/eda_calidad_aire.ipynb src/preprocesamiento.py

# Añadir todos los cambios (con cuidado)
git add .
```

### 2. Crear un commit

Una vez añadidos los archivos:

```bash
git commit -m "Añade edición inicial del README con descripción del proyecto"
```

Recomendaciones para mensajes:

- Ser **breve pero descriptivo**.
- Escribir en **imperativo** corto (“Añade”, “Corrige”, “Refactoriza”…).

### 3. Ver el historial

```bash
git log
```

Salida típica:

```text
commit 1a2b3c4d...
Author: Tu Nombre <tu.email@example.com>
Date:   2025-01-10 10:15:32 +0100

    Añade edición inicial del README con descripción del proyecto

commit 9f8e7d6c...
Author: Tu Nombre <tu.email@example.com>
Date:   2025-01-09 18:02:10 +0100

    Crea estructura inicial del proyecto
```

Para un historial más compacto:

```bash
git log --oneline --graph --decorate
```

Esto permite ver de un vistazo la secuencia de commits y facilita la navegación por el historial.

---

## Comandos básicos que usarás todo el tiempo

### `git status`

Para ver el estado actual del repositorio:

```bash
git status
```

- Archivos no rastreados (en rojo).
- Archivos modificados.
- Archivos en staging listos para commit (en verde).

### `git diff`

Para ver qué ha cambiado:

```bash
# Diferencias sin preparar aún (working directory vs última versión)
git diff

# Diferencias ya preparadas (staged) para commit
git diff --cached
```

Muy útil para revisar cambios antes de confirmar un commit.

### `git log`

Como hemos visto, muestra el historial. Puedes combinar con opciones:

```bash
git log --oneline --graph --all
```

---

## Ignorar archivos innecesarios: .gitignore

En proyectos de datos **no queremos** versionar:

- Archivos temporales.
- Resultados generados que se pueden volver a calcular.
- Archivos grandes (datasets originales, si son muy pesados o sensibles).

Para eso usamos un archivo llamado `.gitignore` en la raíz del proyecto:

```text
# .gitignore (ejemplo)
data/raw/
data/*.csv
results/
*.log
.ipynb_checkpoints/
__pycache__/
.env
```

- Cada línea indica un patrón a ignorar.
- Esto evita que estos archivos aparezcan en `git status` o se añadan por error.

En análisis de datos es común:
- Versionar **scripts, notebooks, notebooks “limpios” e informes**.
- No versionar datasets muy pesados ni datos confidenciales.

---

## (Opcional) Conectar con un repositorio remoto

Aunque esta sesión se centra en repositorios locales, es útil conocer la idea:

- Un **repositorio remoto** (por ejemplo, en GitHub/GitLab) permite:
  - Copia de seguridad.
  - Colaboración.
  - Publicar código.

### Clonar un repositorio existente

```bash
git clone https://github.com/usuario/mi-proyecto.git
```

### Enviar cambios a un remoto (push)

Tras configurar el remoto:

```bash
# Ver remotos configurados
git remote -v

# Enviar commits locales a la rama principal (main o master)
git push origin main
```

Estos pasos ya dependen más de la plataforma (GitHub, GitLab…), pero la idea general es:

1. Hacer commits localmente.
2. Sincronizar con el remoto mediante `push` (y `pull` para traer cambios).

---

## Cómo se relaciona Git con todo lo visto en el curso

Git encaja naturalmente con el flujo de análisis de datos:

- **Sesión 1–2**: Estructura de proyecto y primeros notebooks.
  - Commits para:
    - “Crea estructura inicial del proyecto”.
    - “Añade notebook de introducción a pandas”.

- **Sesión 3–4**: Limpieza y preprocesamiento.
  - Commits para:
    - “Implementa limpieza de valores perdidos y outliers”.
    - “Añade pipeline de preprocesamiento con escalado y codificación”.

- **Sesión 5–6**: Visualización.
  - Commits para:
    - “Crea EDA visual de dataset de viviendas”.
    - “Añade visualizaciones multivariables con FacetGrid y PairGrid”.

- **Sesión 7**: Modelización.
  - Commits para:
    - “Entrena modelos de regresión y clasificación con validación cruzada”.
    - “Aplica K-Means para segmentación de clientes”.

- **Sesión 8**: Informes y comunicación.
  - Commits para:
    - “Prepara informe HTML de EDA”.
    - “Añade notebook final con resultados y conclusiones”.

De este modo, puedes:

- Volver a versiones intermedias si un cambio rompe algo.
- Ver cómo ha evolucionado tu enfoque.
- Compartir el proyecto con otras personas que puedan reproducir y revisar el trabajo.

---

## Ejercicios sugeridos

1. **Inicializar un repositorio local**:
   - En la carpeta de tu proyecto del curso (`/Users/aalcacer/siq025` o similar).
   - Ejecutar `git init`.
   - Crear un primer commit con la estructura básica y alguna sesión.

2. **Practicar el flujo básico**:
   - Modificar un notebook o una sesión (por ejemplo, añadir una gráfica nueva).
   - Usar `git status` para ver los cambios.
   - Hacer `git add` de los archivos relevantes.
   - Crear un commit con un mensaje descriptivo.

3. **Revisar historial y diferencias**:
   - Ejecutar `git log` y `git log --oneline`.
   - Usar `git diff` para ver las diferencias entre la versión actual y el último commit.

4. **Configurar un `.gitignore`**:
   - Crear un archivo `.gitignore` en la raíz del proyecto.
   - Añadir patrones para ignorar:
     - Directorios de `results/`, `data/raw/`, `.ipynb_checkpoints/`, etc.
   - Verificar con `git status` que estos archivos ya no aparecen como cambios.

5. **(Opcional) Subir el proyecto a GitHub/GitLab**:
   - Crear un repositorio vacío en GitHub/GitLab.
   - Añadirlo como remoto (`git remote add origin ...`).
   - Hacer `git push` para subir tus commits.

---

## Conclusiones de la sesión

- Git es una herramienta fundamental para **versionar** proyectos de análisis de datos y garantizar la **reproducibilidad**.
- El flujo básico (`status` → `add` → `commit` → `log`) cubre la mayoría de necesidades en proyectos individuales.
- Un buen uso de `.gitignore` evita versionar archivos temporales o datos pesados/sensibles.
- Integrar Git desde el inicio del proyecto facilita:
  - Volver a estados anteriores.
  - Entender la evolución del análisis.
  - Colaborar con otras personas o compartir tu trabajo.
- Aunque esta sesión es un bonus, dominar estos conceptos te acerca a la forma de trabajar en **proyectos profesionales** de ciencia de datos y desarrollo de software.

