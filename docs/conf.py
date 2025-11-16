import os
import sys

# Add project root to sys.path (optional, if autodoc or imports are needed)
sys.path.insert(0, os.path.abspath('..'))

project = 'SIQ025 — Programari Professional d\'Anàlisi de Dades'
extensions = [
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Source suffix: accept Markdown
source_suffix = {
    '.md': 'markdown',
}

# MyST configuration
myst_enable_extensions = [
    'deflist',
    'html_admonition',
    'colon_fence',
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
