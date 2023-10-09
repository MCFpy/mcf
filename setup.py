from distutils.core import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'mcf',
  packages = ['mcf'],
  version = '0.4.2',
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
    'Programming Language :: Python :: 3.11'
  ],
  install_requires=[
    'ray[default]>=2.7.0',
     'pandas>=2.1.1',
     'matplotlib>=3.8.0',
     'numba>=0.58.0',
     'sympy>=1.12',
     'scikit-learn>=1.3.1',
     'scipy>=1.11.2'
     ]
)
