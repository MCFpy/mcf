# Modified Causal Forest 

## Quick Start

The package `mcf` is accompanied by synthetic data, which we  use in our tutorial. The data can be downloaded from the [Github repository](https://github.com/MCFpy/mcf/tree/main/data). Sticking to most default values, the programme can be as quickly started as follows:

```python
from mcf import modified_causal_forest

# If paths are not specified, the current directory will be used
outpfad = 'your/output/goes/here'
datpfad = 'your/data/is/here'

indata = 'dgp_mcfN1000S5'           # csv for estimation

d_name = ['d']          # Treatment: Must be discrete
y_name = ['y']          # List of outcome variables
x_name_ord = ['cont0']
```

Now, to run the programme simply type:

```python

modified_causal_forest(outpfad=outpfad, datpfad=datpfad, indata=indata,
                       d_name=d_name, y_name=y_name, x_name_ord=x_name_ord)

```
Per default, the output will be printed to the console and written into the output directory. More details follow in the [walkthrough](./part_i.md) and [tutorial](./tutorial_1.md).
