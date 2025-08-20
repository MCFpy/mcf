Feature selection
======================================

The estimation quality of a random forest deteriorates with the number of irrelevant features, because the probability of picking a split based on an irrelevant feature increases. For this reason, it makes sense to remove such features prior to estimation. A bonus of feature selection is that the computational speed increases as a result of a smaller feature space.

The class :py:class:`~mcf_main.ModifiedCausalForest` provides you with the option to perform feature selection through the parameter ``fs_yes``. If set to True, feature selection is performed. Loosely speaking, the program estimates reduced forms for the treatment and the outcome using random forests and then drops features that have little power to predict the treatment **and** the outcome. 

Note that, an irrelevant feature is never dropped if

- the variable is required for the estimation of :math:`\textrm{GATE's}`, :math:`\textrm{BGATE's}` or :math:`\textrm{CBGATE's}`
- the variable is specified in the parameters ``var_x_name_remain_ord`` or ``var_x_name_remain_unord`` of your :py:class:`~mcf_main.ModifiedCausalForest`
- the correlation between two variables to be deleted is bigger than 0.5. In this case, one of the two variables is kept.

Parameter overview
------------------

The following table summarizes the parameters related to feature selection in the class :py:class:`~mcf_main.ModifiedCausalForest`:

+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameter                 | Description                                                                                                                                                                                                                                                                                                                       |
+===========================+===================================================================================================================================================================================================================================================================================================================================+
| ``fs_yes``                | If True, feature selection is performed. Default: False.                                                                                                                                                                                                                                                                          |
+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``fs_other_sample``       | If True, a random sample from the training data is used to perform feature selection. This sample will subsequently not be used to train the Modified Causal Forest. If False, the same data is used for feature selection and to estimate the Modified Causal Forest. Default: True. Only relevant if ``fs_yes`` is set to True. |
+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``fs_other_sample_share`` | If ``fs_other_sample`` is set to True, this determines the sample share used for feature selection. Default: 0.33. Only relevant if ``fs_yes`` is set to True.                                                                                                                                                                    |
+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``fs_rf_threshold``       | Defines the threshold for a feature to be considered "irrelevant". This is measured as the percentage increase of the loss function when the feature is randomly permuted. Default: 1. Only relevant if ``fs_yes`` is set to True.                                                                                                |
+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Please consult the :py:class:`API <mcf_main.ModifiedCausalForest>` for more details.

Example 
-------

.. code-block:: python

    from mcf.example_data_functions import example_data
    from mcf.mcf_main import ModifiedCausalForest
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        # Parameters for feature selection:
        fs_yes=True,
        fs_other_sample=True,
        fs_other_sample_share=0.1,
        fs_rf_threshold=0.5
    )
       
