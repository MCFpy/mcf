from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'mcf',
  packages = ['mcf'],
  version = '0.11.0',
  license='MIT',
  description = 'Modified Causal Forest in Python',
  author = 'mlechner',
  author_email = 'michael.lechner@unisg.ch',
  url = 'https://mcfpy.github.io/mcf/#/',
  keywords = ['causal machine learning, heterogeneous treatment effects, causal forests, optimal policy learning'],
  long_description=read('README.txt'),
  long_description_content_type="text/markdown",  

  python_requires=">=3.12,<3.15",

  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
    'Programming Language :: Python :: 3.14',
  ],
  install_requires=[
     # Python 3.12: Ray on all platforms
     'ray==2.59.0; python_version == "3.12"',

     # Python 3.13: Ray only on non-Windows (Linux/macOS)
     'ray==2.59.0; python_version == "3.13" and sys_platform != "win32"',
      
     # Python 3.14: no Ray dependency; parallel processing uses Joblib

     # Runtime dependencies: matching versions in both supplied environments
     'pandas==3.0.6',
     'matplotlib==3.11.2',
     'numba==0.68.0',
     'sympy==1.14.0',
     'scikit-learn==1.9.1',
     'scipy==1.18.1',
     'torch==2.14.1',
     'fpdf2==2.8.9', 
     'seaborn==0.13.2', 
     'statsmodels==0.15.0',
     'numpy==2.5.3',
     'joblib==1.6.0',
     'psutil==7.2.2'
     ]
)
