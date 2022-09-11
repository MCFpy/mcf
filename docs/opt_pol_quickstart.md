# Optimal Policy Tree

## Quick Start

The ``optpoltree`` is as quickly set up as follows:

```python
from mcf import optpoltree

datpath = 'your/data/is/here'
outpath = '/your/output/goes/here'
indata = 'dgp_mcfN1000S5PredpredXXX'

x_ord_name = ['CONT0', 'CONT1']
x_unord_name = ['CAT0PR', 'CAT1PR']
polscore_name = ['YLC0_pot', 'YLC1_pot', 'YLC2_Pot', 'YLC3_Pot']
effect_vs_0 = ['YLC1vs0_iate', 'YLC2vs0_iate', 'YLC3vs0_iate']
effect_vs_0_se = ['YLC1vs0_iate_se', 'YLC2vs0_iate_se',
                      'YLC3vs0_iate_se']
_with_output=True
```

Now, to run the programme simply type

```python
optpoltree(polscore_name=polscore_name, x_ord_name=x_ord_name, x_unord_name=x_unord_name, effect_vs_0=effect_vs_0, 
           effect_vs_0_se=effect_vs_0_se,  indata=indata, datpath=datpath, outpath=outpath, _with_output=_with_output)
```

By default, the output is printed to the console and written into the output directory.
