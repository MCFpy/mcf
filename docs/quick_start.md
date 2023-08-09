# Modified Causal Forest

## Quick Start

Sticking to most default values, you generate an instance of the model class `ModifiedCausalForest` as follows:

```python
import pandas as pd
from mcf import ModifiedCausalForest

my_mcf = ModifiedCausalForest(var_d_name="d", var_x_name_ord="x_ord",
                              var_x_unord="x_unord", var_y_name="y")
```
Before calling the train and predict method, load the data, i.e.

```python
training_data = pd.read_csv("here/are/my/train_data.csv")
prediction_data = pd.read_csv("here/are/my/pred_data.csv")
```
Note the difference to earlier versions where the data was passed over in form of a path.

Now, you are ready to train and predict:

```python
my_mcf.train(training_data)
results = my_mcf.predict(prediction_data)
```

The ``results`` object is a dictionary, storing all estimated effects and their standard errors. Feel free to explore the scope of the object yourself via

```python
print(results.keys())
```
Variables without the *_df* suffix are lists or numpy arrays. Variables with the *_df* suffix are pandas DataFrames. The *iate_pred_df* contains the outcome variables, the IATEs, and the corresponding outcomes, which you can use later on for an optimal policy analysis.

To receive information on cluster membership as implied by the k-means analysis, type

```python
results_with_cluster = my_mcf.analyse(results)
```

The resulting object *results_with_cluster* differs from the predict method only through the ``iate_pred_df`` object, which contains the cluster indicator for each observation.
