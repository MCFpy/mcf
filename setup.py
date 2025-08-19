from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'mcf',
  packages = ['mcf'],
  version = '0.8.0',
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
    'ray==2.48.0',
     'pandas==2.3.1',
     'matplotlib==3.10.5',
     'numba==0.61.2',
     'sympy==1.14.0',
     'scikit-learn==1.7.1',
     'scipy==1.16.1',
     'torch==2.8.0',
     'fpdf2==2.8.4', 
     'seaborn==0.13.2', 
     'statsmodels==0.14.5'
     ]
)
