import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

project = "Deep Learning Tutorial"
author = "Deep Learning Tutorial Contributors"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "alabaster"
html_static_path = ["_static"]
