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
    'Development Status :: 5',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8'
  ],
  install_requires=[
   'futures>=3.3.0',
   'multiprocess>=0.70.5',
   'numpy>=1.20.2',
   'pandas>=1.2.5',
   'matplotlib>=1.5.3',
   'scipy>=1.2.1',
   'ray>=1.4.0',
   'numba>=0.54',
   'scikit-learn>=0.23.2',
   'psutil>=5.4.5',
   'importlib>=1.0.4',
   'sympy>=1.8',
   'pathlib>=1.0.1']
)
