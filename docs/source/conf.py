# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
# sys.path.insert(0, os.path.abspath('..'))
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('../../mcf'))
sys.path.insert(0, os.path.abspath('../../mcf/mcf')) # added on 22.08


# -- Project information -----------------------------------------------------

project = 'mcf 0.8.0'
copyright = '2024, Michael Lechner'
author = 'Michael Lechner'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'sphinx.ext.githubpages',
    'sphinx.ext.doctest',
    'sphinx_design'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files. This pattern also
# affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames. You can specify multiple suffix as a list
# of string:
source_suffix = ['.rst', '.md']

# This will allow your docs to import the example code without requiring those
# modules be installed
autodoc_mock_imports = ['bs4', 'requests', 'pandas', 'mcf', 'time', 'copy', 'datetime', 'os']

# This ensures that the autoclass directive will only include the class'
# docstring without the docstring of the __init__method.
autoclass_content = 'class'

# This will generate stub documentation pages for items included in autosummary
# directives (even if those autosummary directives are "commented out"!).
autosummary_generate = True

# Ignore lines in the docstrings that are enclosed by the following lines:
# <NOT-ON-API>
# ...
# </NOT-ON-API>
from sphinx.ext.autodoc import between

def setup(app):
    app.connect('autodoc-process-docstring', between('^.*</?NOT-ON-API>.*$', exclude=True))
    return app

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'pydata_sphinx_theme'

# This removes the primary (left) sidebar from all pages of the documentation:
html_sidebars = {
  "**": []
}

html_theme_options = {
    # Depth of the table of contents shown in the secondary (right) sidebar
    "show_toc_level": 3,
    # Links in the navigation bar to GitHub and PyPI
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MCFpy/mcf",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/mcf/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ]
}
