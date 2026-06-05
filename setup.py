from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'mcf',
  packages = ['mcf'],
  version = '0.10.0',
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
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',      
  ],
  install_requires=[
     # Python 3.12: Ray on all platforms
     'ray==2.55.1; python_version == "3.12"',

     # Python 3.13: Ray only on non-Windows (Linux/macOS)
     'ray==2.55.1; python_version == "3.13" and sys_platform != "win32"',
      
     'pandas==3.0.3',
     'matplotlib==3.10.9',
     'numba==0.65.1',
     'sympy==1.14.0',
     'scikit-learn==1.8.0',
     'scipy==1.17.1',
     'torch==2.12.0',
     'fpdf2==2.8.7', 
     'seaborn==0.13.2', 
     'statsmodels==0.14.6'
     ]
)
