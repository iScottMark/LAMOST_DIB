import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../../dibkit'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LAMOST DIB'
copyright = '2025, Scott Mark'
author = 'Scott Mark'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
              'sphinx.ext.autodoc',       # Automatically document from docstrings 
              'sphinx.ext.napoleon',      # Support for Google-style docstrings
              'sphinx.ext.viewcode',      # Add links to highlighted source code
              'nbsphinx',                 # Integrate Jupyter Notebooks
              'sphinx_copybutton'         # Add "copy" buttons to code blocks
              ]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
   "icon_links": [
      {
            "name": "GitHub",
            "url": "https://github.com/iScottMark/LAMOST_DIB",  # Replace with your GitHub repository URL
            "icon": "fa-brands fa-github",  # FontAwesome GitHub icon
            "type": "fontawesome",
      }
   ]
}
html_context = {
   # ...
   "default_mode": "dark"
}
html_static_path = ['_static']
