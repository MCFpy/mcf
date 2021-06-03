from distutils.core import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'mcf',
  packages = ['mcf'],
  version = '1.1.0',
  license='MIT',
  description = 'mcf is a powerful package to estimate heterogeneous treatment effects for multiple treatment models in a selection-on-observables setting',
  author = 'mlechner',                   
  author_email = 'michael.lechner@unisg.ch',      
  url = 'https://github.com/MCFpy/mcf',      
  keywords = ['causal machine learning, heterogeneous treatment effects, causal forests'],  
  long_description=read('README'),
  classifiers=[
    'Development Status :: 4 - Beta',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3.8'
  ],
)
