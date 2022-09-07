"""Created on Wed Apr  1 15:58:30 2021.

Modified Causal Forest - Python implementation

Can be used under Creative Commons Licence CC BY-SA
Michael Lechner, SEW, University of St. Gallen, Switzerland

Version: 0.0.1

-*- coding: utf-8 -*- .

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

Change-log: PLEASE check 'user_mcf_full.py' for information.

"""
from mcf import mcf_functions as mcf

# If paths are not specified, the current directory will be used
OUTPFAD = 'D:/mlechner/mcftest/out'
DATPFAD = 'D:/mlechner/mcftest/testdata'

INDATA = 'dgp_mcfN1000S5'           # csv for estimation

D_NAME = ['d']          # Treatment: Must be discrete (not needed for pred)
Y_NAME = ['y']          # List of outcome variables (not needed for pred)
X_NAME_ORD = ['cont0']

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    mcf.ModifiedCausalForest(
        outpfad=OUTPFAD, datpfad=DATPFAD, indata=INDATA,
        d_name=D_NAME, y_name=Y_NAME, x_name_ord=X_NAME_ORD)
