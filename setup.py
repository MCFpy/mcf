from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'mcf',
  packages = ['mcf'],
  version = '0.7.0',
  license='MIT',
  description = 'The Python package mcf implements the Modified Causal Forest introduced by Lechner (2018). This package allows you to estimate heterogeneous treatment effects for binary and multiple treatments from experimental or observational data. Additionally, mcf offers the capability to learn optimal policy allocations.',
  author = 'mlechner',
  author_email = 'michael.lechner@unisg.ch',
  url = 'https://mcfpy.github.io/mcf/#/',
  keywords = ['causal machine learning, heterogeneous treatment effects, causal forests, optimal policy learning'],
  long_description=read('README.txt'),
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.12'
  ],
  install_requires=[
    'ray>=2.36.0',
     'pandas>=2.2.2',
     'matplotlib>=3.9.2',
     'numba>=0.60.0',
     'sympy>=1.13.3',
     'scikit-learn>=1.5.2',
     'scipy>=1.14.1',
     'torch>=2.4.1',
     'fpdf2>=2.7.9'
     ]
)
