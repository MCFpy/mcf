import numpy as np
import pandas as pd

def simulate_data(n):
    """
    Simulate data with treatment 'd', outcome 'y', an unordered control
    variable 'female' and two ordered controls 'x1', 'x2'.

    Parameters:
    - n (int): Number of observations in the simulated data.

    Returns:
    pd.DataFrame: Simulated data in a Pandas DataFrame.

    """
    d = np.random.choice([0, 1, 2], n, replace=True)
    female = np.random.choice([0, 1], n, replace=True)
    x_ordered = np.random.normal(size=(n, 2))
    y = (x_ordered[:, 0] +
        x_ordered[:, 1] * (d == 1) +
        x_ordered[:, 1] * (d == 2) +
        0.5 * female +
        np.random.normal(size=n))

    data = {"y": y, "d": d, "female": female}

    for i in range(x_ordered.shape[1]):
        data["x" + str(i + 1)] = x_ordered[:, i]

    return pd.DataFrame(data)

df = simulate_data(1000)

indices = np.array_split(df.index, 3)
train_mcf_df, pred_mcf_train_pt_df, evaluate_pt_df = (df.iloc[ind] for ind in indices)

from mcf import ModifiedCausalForest

my_mcf = ModifiedCausalForest(
    var_y_name="y",
    var_d_name="d",
    var_x_name_ord=["x1", "x2"],
    var_x_name_unord=["female"],
    _int_show_plots=False # Suppress the display of diagnostic plots during estimation
)

my_mcf.train(train_mcf_df)
results = my_mcf.predict(pred_mcf_train_pt_df)
print(results.keys())
results["ate effect_list"]
results["ate"]
results["ate_se"]
results["iate_data_df"]
my_mcf.analyse(results)
oos_results = my_mcf.predict(evaluate_pt_df)

# Out of sample
results_oos = my_mcf.predict(evaluate_pt_df)
oos_df = results_oos['iate_data_df']

# TO-DO:
from mcf import OptimalPolicy 

my_opt_policy_tree = OptimalPolicy(var_d_name="d", 
                                   var_polscore_name=["Y_LC0_un_lc_pot", "Y_LC1_un_lc_pot", "Y_LC2_un_lc_pot"],
                                   var_x_ord_name=["x1", "x2"],
                                   var_x_unord_name=["female"],
                                   gen_method='policy tree', 
                                   pt_depth=2 # Depth of the policy tree
    )

data_train_pt = results['iate_data_df']
alloc_train_df = my_opt_policy_tree.solve(data_train_pt)

eval = my_opt_policy_tree.evaluate(allocation_df=alloc_train_df,
                data_df=data_train_pt,
                data_title='Training PT data')

# Augments this (comparison of tree, observerd allocation, random allocation)
alloc_train_df.head()

alloc_eva_df = my_opt_policy_tree.allocate(oos_df)
results_eva_train = myoptp.evaluate(alloc_eva_df, oos_df)