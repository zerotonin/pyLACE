"""Sphinx configuration for pyLACE docs.

Built locally with::

    pip install -r docs/requirements.txt
    sphinx-build -b html docs/ docs/_build/html

CI build is in ``.github/workflows/docs.yml``; the output ships to
GitHub Pages on every push to ``master``.
"""

from __future__ import annotations

import importlib.metadata as _md

project = "pyLACE"
author = "Bart Geurten"
copyright = "2026, Bart Geurten"

try:
    release = _md.version("pylace")
except _md.PackageNotFoundError:
    release = "0.0.0"
version = release.split("+", 1)[0]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Heavy / GUI / native deps that the docs runner does not need to
# import. autodoc still pulls docstrings; the modules just import
# successfully under a mock.
autodoc_mock_imports = [
    "cv2",
    "h5py",
    "matplotlib",
    "PyQt6",
    "scipy",
    "yaml",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []
html_title = f"pyLACE {version}"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}
