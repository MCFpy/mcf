from distutils.core import setup
setup(
  name = 'mcf',
  packages = ['mcf'],
  version = '0.0.0.3',
  license='MIT',
  description = 'mcf lets you estimate heterogeneous treatment effects for multiple treatment models in a selection-on-observables setting.',
  author = 'mlechner',                   
  author_email = 'michael.lechner@unisg.ch',      
  url = 'https://github.com/MCFpy/mcf',   
  download_url = 'https://github.com/MCFpy/mcf/archive/refs/tags/v0.0.0.3.tar.gz',    
  keywords = ['causal machine learning, heterogeneous treatment effects, causal forests'],  
  classifiers=[
    'Development Status :: 4 - Beta',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3.8'
  ],
)
