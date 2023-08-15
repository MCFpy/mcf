# MCF

## Installing from PyPI

The mcf can be installed from PyPI. 

```
pip install mcf
```

The current mcf runs with Python 3.11. We recommend setting up a virtual environment designated to the mcf to rule out dependency conflicts. It is always a good idea to first ``conda install`` packages and only then ``pip install`` packages. Should you work on a Windows machine and wish to deploy Spyder as an IDE, make sure to first ``conda install spyder`` and then ``pip install mcf``. Otherwise, you may run into errors. 

To set up a new environment with anaconda, you can specify


```conda
conda create --name mcf_environment
```


Before pip installing, make sure to activate the correct environment, i.e.

```conda
conda activate mcf_environment
```

Should you work with ``virtualenv``, set up a new environment as follows

```bash
virtualenv mcf_environment
```

Also here, activate the correct environment:

On Windows:

```bash
mcf_environment\Scripts\activate
```

On MacOS and Linux

```bash
source mcf_environment/bin/activate
```
