# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('../../mcf'))


# -- Project information -----------------------------------------------------

project = 'mcf 0.4.4'
copyright = '2023, ML'
author = 'ML'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx_collapse',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'sphinx.ext.githubpages',
    'sphinx.ext.doctest',
]

napoleon_use_ivar = False

# This will allow your docs to import the example code without requiring those modules be installed
autodoc_mock_imports = ['bs4', 'requests']

autoclass_content = 'class'
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'

# This removes the primary (left) sidebar from all pages of the documentation:
html_sidebars = {
  "**": []
}

# Custom sidebar templates, maps page names to templates.
# html_sidebars = {
#    "index": [
#        "sidebar_versions.html",
#        ]}

html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 4,
    "show_toc_level": 5,
    "show_version_warning_banner": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MCFpy/mcf",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://github.com/MCFpy/mcf",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },

    ]
}



# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
