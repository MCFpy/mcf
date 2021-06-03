from distutils.core import setup
setup(
  name = 'mcf',
  packages = ['mcf'],
  version = '0.0.5',
  license='MIT',
  description = 'mcf is a powerful package to estimate heterogeneous treatment effects for multiple treatment models in a selection-on-observables setting',
  author = 'mlechner',                   
  author_email = 'michael.lechner@unisg.ch',      
  url = 'https://github.com/MCFpy/mcf',   
  download_url = 'https://github.com/MCFpy/mcf/archive/refs/tags/0.0.5.tar.gz',    
  keywords = ['causal machine learning, heterogeneous treatment effects, causal forests'],  
  long_description="""# Project Description \n The mcf is a Python package that provides a fast and easily implementation of the Modified Causal Forest (MCF). The scope of the MCF is to estimate heterogeneous treatment effects for multiple treatment models in a selection-on-observables setting. For more details refer to the official documentation, which can be accessed [here](https://mcfpy.github.io/mcf/#/).\n""",
  long_description_content_type='text/markdown',
  classifiers=[
    'Development Status :: 4 - Beta',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3.8'
  ],
)
