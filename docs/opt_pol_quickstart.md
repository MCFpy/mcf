# Optimal Policy Tree

## Quick Start

To set up an object of the class ``OptimalPolicy``, type

```python
from mcf import OptimalPolicy
myoptp = OptimalPolicy(var_d_name="d",
                       var_polscore_name=["y0", "y1"],
                       var_x_ord_name=["x1"],
                       var_x_unord_name=["x2"],
                       gen_method="policy tree"
                        )
```

Admissible methods of the class ``OptimalPolicy`` are ``solve``, ``allocate``, and ``evaluate``. Use ``solve`` to learn the policy rule. Use ``allocate`` to map observations to treatments. Use ``evaluate`` to evaluate the derived allocation.

In line with the updated mcf workflow, the data is passed over in form of a pandas DataFrame. Hence, in a first step we read in the training and prediction data.

```python
training_data = pd.read_csv("here/are/my/train_data.csv")
prediction_data = pd.read_csv("here/are/my/pred_data.csv")
```
Now, we are ready for the normative analysis. First, to derive the policy rule, type

```python
alloc_train_df = myoptp.solve(train_df)
```

Second and third to allocate and evaluate the allocation, specify

```python
alloc_pred_df = myoptp.allocate(pred_df)
results_eva_pred = myoptp.evaluate(alloc_pred_df, pred_df)
```
Should you wish to deploy the results from the ``mcf`` estimation, follow the routine

```python
training_data = results["iate_pred_df"]
prediction_data = results["iate_pred_df"]
```
Of course, you may specify a different dataset for the prediction task. Note that the results object is the results object from calling the ``predict`` method.

Note that the program automatically generates pickles in the application path, which contain information on the instantiated object of the class.

In addition to the output mentioned above, the program generates two text files (a long and short version) with further details.
