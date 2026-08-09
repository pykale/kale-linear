"""Sphinx configuration for kalelinear documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

project = "kalelinear"
author = "The PyKale team"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

source_suffix = ".rst"
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "generated/*", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


def skip_sklearn_metadata_request_methods(app, what, name, obj, skip, options):
    """Hide inherited scikit-learn metadata-routing helpers from API docs."""
    metadata_request_methods = {
        "get_metadata_routing",
        "set_fit_request",
        "set_inverse_transform_request",
        "set_partial_fit_request",
        "set_predict_proba_request",
        "set_predict_request",
        "set_score_request",
        "set_transform_request",
    }
    if name in metadata_request_methods:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_sklearn_metadata_request_methods)
