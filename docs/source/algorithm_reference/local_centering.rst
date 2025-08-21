Local centering
===============

Method
--------

Local centering is a form of residualization and can improve the performance of forest estimators by regressing out the impact of the features on the outcome.
Let us define the conditionally centered outcome :math:`\tilde{Y}_i` as:

.. math::

   \tilde{Y}_i = Y_i - \hat{y}_{-i}(X_i)

where:

- :math:`Y_i` is the outcome for observation :math:`i`.
- :math:`\hat{y}_{-i}(X_i)` is an estimate of the conditional outcome expectation :math:`E[Y_i | X_i = x]`, given the realised :math:`x` of the feature vector :math:`X_i`, and computed without using the observation :math:`i`.

Implementation
---------------
Centered outcomes are obtained by subtracting the predicted from the observed outcomes.
The local centering procedure in the **mcf** applies the method from the sklearn.ensemble module `RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=A%20random%20forest%20regressor.,accuracy%20and%20control%20over%2Dfitting.>`_ to compute the predicted outcomes :math:`\hat{y}_{-i}(X_i)` for each observation :math:`i` non-parametrically. The predicted outcomes are computed in distinct subsets by cross-validation with the number of folds specified by ``lc_cs_cv_k``. 

By default ``lc_yes`` is set to ``True`` and runs the described local centering procedure. To overrule it, set ``lc_yes`` to ``False``. 

As an alternative, two separate data sets can be generated for running the local centering procedure with ``lc_cs_cv``. In this case, the first data set is used for training a Random Forest, again by applying the RandomForestRegressor method. The the size of this first dataset can be defined in ``lc_cs_share``. The second dataset is used to compute the predicted and centered outcomes :math:`\hat{y}_{-i}(X_i)` and :math:`\tilde{Y}_i`. Furthermore, this second data set is divided into mutually exclusive data sets for feature selection (optionally), tree building, and effect estimation.

Below, the table below provides a brief description of the relevant keyword arguments for local centering:

+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Argument          | Description                                                                                                                                                        |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lc_yes``        | Activates local centering. Default is True                                                                                                                         |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lc_cs_cv``      | Data for local centering & common support adjustment. True: Crossvalidation. False: Random sample not used for forest building. Default is True.                   |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lc_cs_share``   | Data for local centering & common support adjustment. Share of trainig data (if lc_cs_cv is False). Default is 0.25.                                               |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lc_cs_cv_k``    | Number of folds in cross-validation (if lc_cs_cv is True). This is dependent on the size of the training sample and ranges from 2 to 5.                            |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Example
~~~~~~~

.. code-block:: python

   from mcf.example_data_functions import example_data
   from mcf.mcf_main import ModifiedCausalForest
   
   # Generate example data using the built-in function `example_data()`
   training_df, prediction_df, name_dict = example_data()
   
   
   my_mcf = ModifiedCausalForest(
       var_y_name="outcome",
       var_d_name="treat",
       var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
       # Activates local centering
       lc_yes = True,
       # Data for local centering & common support adjustment by crossvalidation
       lc_cs_cv = True,
       # Number of folds in cross-validation
       lc_cs_cv_k = 5
   )
   
   my_mcf.train(training_df)
   results = my_mcf.predict(prediction_df)
