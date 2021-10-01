from distutils.core import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'mcf',
  packages = ['mcf'],
  version = '0.1.1',
  license='MIT',
  description = 'mcf is a powerful package to estimate heterogeneous treatment effects for multiple treatment models in a selection-on-observables setting and learn optimal policy rules',
  author = 'mlechner',
  author_email = 'michael.lechner@unisg.ch',
  url = 'https://github.com/MCFpy/mcf',
  keywords = ['causal machine learning, heterogeneous treatment effects, causal forests, optimal policy learning'],
  long_description=read('README.txt'),
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8'
  ],
  install_requires=[
   'numpy>=1.20.0',
   'pandas>=1.3.0',
   'matplotlib>=3.4.2',
   'scipy>=1.7.0',
   'ray>=1.4.0',
   'numba>=0.53.1',
   'scikit-learn>=0.24.2',
   'psutil>=5.8.0',
   'importlib>=1.0.4',
   'sympy>=1.8',
   'pathlib>=1.0.1']
)
