Local centering
===============

Method
--------

Local centering is a form of residualization, which can improve the performance of forest estimators. 
This performance improvement is achieved by regressing out the impact of the features on the outcome.

Formally, the conditionally centered outcome :math:`\tilde{Y}_i` can be defined as:

.. math::

   \tilde{Y}_i = Y_i - \hat{y}_{-i}(X_i)


where:

- :math:`\tilde{Y}_i` is the conditionally centered outcome.
- :math:`Y_i` indicates the outcome for observation :math:`\textrm{i}`.
- :math:`\hat{y}_{-i}(X_i)` is an estimate of the conditional outcome expectation :math:`E[Y_i | X_i = x_i]`, given the observed values :math:`x_i` of the feature vector :math:`X_i`, computed without using the observation :math:`\textrm{i}`.


Implementation
---------------

The local centering procedure applies the  method of the sklearn.ensemble module `RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=A%20random%20forest%20regressor.,accuracy%20and%20control%20over%2Dfitting.>`_ to compute the predicted outcomes :math:`\hat{y}_{-i}(X_i)` for each observation :math:`\textrm{i}` non-parametrically. 
To turn the procedure off, overrule the default ``lc_yes`` and set it to ``False``. The predicted outcomes are computed in distinct subsets by cross-validation, where the number of folds can be specified in ``lc_cs_cv_k``. Finally, the centered outcomes are obtained by subtracting the predicted from the observed outcomes.


Alternatively, two separate data sets can be generated for running the local centering procedure with ``lc_cs_cv``. In this case, the size of the first data set can be defined in ``lc_cs_share`` and it is used for training a Random Forest, again by applying the RandomForestRegressor method. The predicted and centered outcomes :math:`\hat{y}_{-i}(X_i)` and :math:`\tilde{Y}_i`, respectively, are computed in the second data set. Finally, this second data set is divided into mutually exclusive data sets for feature selection (optionally), tree building, and effect estimation.

Below, you find a table with a brief description of the relevant keyword arguments for local centering:

+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Argument          | Description                                                                                                                                                        |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lc_yes``        | Activates local centering. Default is True                                                                                                                         |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lc_cs_cv``      | Data for local centering & common support adjustment. True: Crossvalidation. False: Random sample not used for forest building. Default is True.                   |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lc_cs_share``   | Data for local centering & common support adjustment. Share of trainig data (if lc_cs_cv is False). Default is 0.25.                                               |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lc_cs_cv_k``    | Number of folds in cross-validation (if lc_cs_cv is True). Default is 5.                                                                                           |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Example
~~~~~~~

.. code-block:: python

   from mcf.example_data_functions import example_data
   from mcf.mcf_functions import ModifiedCausalForest
   
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
   results, _ = my_mcf.predict(prediction_df)
