Data cleaning
=============

TO-DO Link to API

The class :py:class:`~mcf_mini.ModifiedCausalForest` has several data cleaning options to improve the estimation quality of your Modified Causal Forest. Below, you find a table with the relevant parameters and a brief description: 

+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+
| Parameter                | Description                                                                                                                                | 
+==========================+============================================================================================================================================+
| ``dc_clean_data``        | If True, all observations with missing values are dropped. Variables not required in the analysis are also removed. Default: True.         | 
+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+
| ``dc_screen_covariates`` | If True, covariates are screened and cleaned. Specifically, features without variation are dropped. Default: True.                         |
+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+
| ``dc_check_perfectcorr`` | If ``dc_screen_covariates`` is True, covariates perfectly correlated with others are removed. Default: True.                               |
+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+
| ``dc_min_dummy_obs``     | If ``dc_screen_covariates`` is True, binary (dummy) covariates with less than ``dc_min_dummy_obs`` zeroes or ones are removed. Default: 10.|
+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------+

Please consult the :py:class:`API ~mcf_mini.ModifiedCausalForest` for more details.

Example 
-------

.. code-block:: python

    from mcf import ModifiedCausalForest

    my_mcf = ModifiedCausalForest(
        var_y_name="y",
        var_d_name="d",
        var_x_name_ord=["x1", "x2"],
        # Parameters for data cleaning:
        dc_clean_data=True,
        dc_screen_covariates=True,
        dc_check_perfectcorr=False,
        dc_min_dummy_obs=100
    )
