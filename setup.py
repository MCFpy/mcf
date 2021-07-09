from distutils.core import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'mcf',
  packages = ['mcf'],
  version = '0.0.1',
  license='MIT',
  description = 'mcf is a powerful package to estimate heterogeneous treatment effects for multiple treatment models in a selection-on-observables setting',
  author = 'mlechner',
  author_email = 'michael.lechner@unisg.ch',
  url = 'https://github.com/MCFpy/mcf',
  keywords = ['causal machine learning, heterogeneous treatment effects, causal forests'],
  long_description=read('README.txt'),
  classifiers=[
    'Development Status :: 4 - Beta',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8'
  ],
  install_requires[
  'futures>=v3.3.0',
   'multiprocess>=v0.70.5',
   'numpy>=v1.20.2',
   'pandas>=v1.2.5',
   'matplotlib>=v1.5.3',
   'scipy>=v1.2.1',
   'ray>=v1.4.0',
   'numba>=v0.54',
   'scikit-learn>=v0.23.2',
   'psutil>=v5.4.5',
   'importlib>=v1.0.4',
   'sympy>=v1.8',
   'pathlib>=v1.0.1'
  ]
)
