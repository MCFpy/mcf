# MCF

## Installing from PyPI

The mcf can be installed from PyPI as follows: 

```
pip install mcf
```

The current mcf runs with Python 3.11. 


We recommend setting up a virtual environment designated to the mcf to rule out dependency conflicts. To set up a new environment with anaconda, you can specify


```conda
conda create --name mcf_environment
```


Before pip installing, make sure to activate the correct environment, i.e.

```conda
conda activate mcf_environment
```

It is always a good idea to first ``conda install`` packages and only then ``pip install`` packages. Should you work on a Windows machine and wish to deploy Spyder as an IDE, make sure to first ``conda install spyder`` and then ``pip install mcf``. Otherwise, you may run into errors. 
