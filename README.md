# SIQ025/SIQ525 - Software para el Análisis de Datos con Python

Esta es una web interactiva con los apuntes del módulo 2: "Software para el Análisis de Datos con Python" (SIQ025/SIQ525 — Máster en Matemática Computacional, UJI).

Contenido:
- Portada: descripción general del curso
- 10 páginas (una por sesión) con apuntes, ejemplos y ejercicios

Requisitos

1. Python 3.12+
2. Crear un entorno virtual (venv o conda)

Instalación

```bash
uv init
uv sync
```

Generar la documentación localmente:

```bash
# build HTML
uv run sphinx-build -b html docs _build/html
# Live reload durante edición (requiere sphinx-autobuild)
uv run sphinx-autobuild docs _build/html
```

