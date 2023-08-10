# Optimal Policy Tree

## Quick Start

An object of the class ``OptimalPolicy`` is quickly set up as follows:

```python
from mcf import OptimalPolicy
myoptp = OptimalPolicy(var_d_name="d",
                       var_polscore_name=["y0", "y1"],
                       var_x_ord_name=["x1"],
                       var_x_unord_name=["x2"]
                        )
```

Admissible methods of the class ``OptimalPolicy`` are ``train``, ``allocate``, and ``evaluate``. Use ``train`` to learn the policy rule. Use ``allocate`` to map observations to treatments. Use ``evaluate`` to evaluate the derived allocation.

In line with the mcf workflow, the data is passed over in form of a pandas dataframe. Hence, in a first step we read in the training and prediction data.

```python
training_data = pd.read_csv("here/are/my/train_data.csv")
prediction_data = pd.read_csv("here/are/my/pred_data.csv")
```
Now, we are ready for the normative analysis. First, to derive the policy rule, type

```python
alloc_train_df = myoptp.train(train_df)
```

Second and third to allocate and evaluate the allocation, specify

```python
alloc_pred_df = myoptp.allocate(pred_df)
results_eva_pred = myoptp.evaluate(alloc_pred_df, pred_df)
```
