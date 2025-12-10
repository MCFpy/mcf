from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'mcf',
  packages = ['mcf'],
  version = '0.7.4',
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
     'ray==2.52.1',
     'pandas==2.3.3',
     'matplotlib==3.10.7',
     'numba==0.62.1',
     'sympy==1.14.0',
     'scikit-learn==1.7.2',
     'scipy==1.16.3',
     'torch==2.9.1',
     'fpdf2==2.8.5', 
     'seaborn==0.13.2', 
     'statsmodels==0.14.6'
     ]
)
