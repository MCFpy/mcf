# MCF

## Installing from PyPI

The mcf can be installed from PyPI. Please follow the installation steps, as outlined here. Otherwise, you may encounter errors.

First,

```
pip install mcf
```

Second,

```
pip install -U --force-reinstall charset-normalizer==3.2.0
```

The current mcf runs with Python 3.11. We recommend setting up a virtual environment designated to the mcf to rule out dependency conflicts. Should you work with anaconda, you can specify


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
