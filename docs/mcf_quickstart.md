# Modified Causal Forest

## Quick Start

Sticking to most default values, you generate an instance of the model class `ModifiedCausalForest` as follows:

```python
import pandas as pd
from mcf import ModifiedCausalForest

my_mcf = ModifiedCausalForest(var_d_name="d", var_x_name_ord=["x_ord"],
                              var_x_name_unord=["x_unord"], var_y_name=["y"])
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
Note that training and prediction data need not be the same. The prediction data needs to have the same features and treatment. Outcome information is not necessary for predicting treatment effects. This may be convenient if you wish to deploy an existing forest structure for estimating treatment effects on similar but distinct data. This may speed up your research workflow considerably. 

The ``results`` object is a dictionary, storing all estimated effects and their standard errors. Feel free to explore the scope of the object yourself via

```python
print(results.keys())
```
Variables without the *_df* suffix are lists or numpy arrays. Variables with the *_df* suffix are pandas DataFrames. The *iate_pred_df* contains the IATEs, and the estimated potential outcomes, which you can use later on for an optimal policy analysis.

For the analysis of the IATEs from prediction, such as information on cluster membership as implied by the k-means analysis, type

```python
results_with_cluster = my_mcf.analyse(results)
```

The resulting object *results_with_cluster* differs from the predict method only through the ``iate_pred_df`` object, which contains the cluster indicator for each observation.  Note that potential outcomes are named ``OUTCOME_(LC)TREATMENT_pot``. The ``LC`` is optional and only included should you have enabled local centering. The IATEs are referred to as ``OUTCOME_LC(TREATMENT)vs(TREATMENT)_iate``. Clusters are assigned to individuals based on their features.

Note that the program generates two text files (a long and short version) with details on the estimation.
