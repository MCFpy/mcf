from distutils.core import setup
setup(
  name = 'mcf',
  packages = ['mcf'],
  version = '0.0.0.0',
  license='MIT',
  description = 'mcf lets you estimate heterogeneous treatment effects for multiple treatment models in a selection-on-observables setting.',
  author = 'mlechner',                   # Type in your name
  author_email = 'michael.lechner@unisg.ch',      # Type in your E-Mail
  url = 'https://github.com/MCFpy/mcf',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/MCFpy/mcf/archive/refs/tags/v0.0.0.0.tar.gz',    # I explain this later on
  keywords = ['causal machine learning, heterogeneous treatment effects, causal forests'],   # Keywords that define your package best
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Research',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.8'
  ],
)
