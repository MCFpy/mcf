Inference
=========

The **mcf** offers three ways of conducting inference. 

- **Weights-based Inference Procedure**: This is the default method in the program. It is particularly useful to gain information on the precision of estimators that have a representation as weighted averages of the outcomes. The variance of the treatment effect estimator is estimated based on a variance decomposition made up of two components:

    - Expectation of the conditional variance
    - Variance of the conditional expectation, given the weights

This decomposition accounts for heteroscedasticity in the weights. The conditional means and variances are estimated non-parametrically, either by the Nadaraya-Watson kernel estimator or by the k-Nearest Neighbor (k-NN) estimator (default). See `Lechner (2018) <https://doi.org/10.48550/arXiv.1812.09487>`_ for more details.

- **Variance of Treatment Effect Estimates**: This method estimates the variance of treatment effect estimates as the sum of the variance of weighted outcomes in the respective treatment states. A drawback of this inference method is that it implicitly assumes homoscedasticity in the weights for each treatment state.


- **Bootstrap Algorithm**: This method uses a bootstrap algorithm to obtain inference by computing standard errors. Our algorithm bootstraps the equally weighted weights and then renormalizes them.


**Note**: because of the weighting representation, inference can also readily be used to account for clustering, which is a common feature in economics data.


Parameters 
------------------------

Below you find a list of the main parameters which are related to the inference procedure of the **mcf**. Please consult the :py:class:`API <mcf_functions.ModifiedCausalForest>` for more details or additional parameters. 

.. list-table:: 
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Description
   * - ``p_se_boot_ate``
     - Bootstrap of standard errors for ATE. Accepts an integer or Boolean (or None). If True, the number of bootstrap replications is set to 199. Default is None, which sets the number of replications to 199 if p_cluster_std is True, and False otherwise.
   * - ``p_se_boot_gate``
     - Bootstrap of standard errors for GATE. Specify either a Boolean or an integer. If True, the number of bootstrap replications is set to 199. Default is None, which sets the number of replications to 199 if p_cluster_std is True, and False otherwise.
   * - ``p_se_boot_iate``
     - Bootstrap of standard errors for IATE. Accepts an integer or Boolean (or None). If True, the number of bootstrap replications is set to 199. Default is None, which sets the number of replications to 199 if p_cluster_std is True, and False otherwise.
   * - ``p_cond_var``
     - Determines if conditional mean and variances are used. Accepts True or False. If True, conditional mean and variances are used; if False, variance estimation is direct. Default (or None) is True.
   * - ``p_knn``
     - Specifies the k-NN method. If True, k-NN estimation is used; if False, Nadaraya-Watson estimation is employed. Nadaraya-Watson estimation provides a better approximation of the variance, while k-NN is faster, especially for larger datasets. Default (or None) is True.


Example
~~~~~~~~~

.. code-block:: python

   from mcf.example_data_functions import example_data
   from mcf.mcf_functions import ModifiedCausalForest
   
   # Generate example data using the built-in function `example_data()`
   training_df, prediction_df, name_dict = example_data()
   
   my_mcf = ModifiedCausalForest(
       var_y_name="outcome",
       var_d_name="treat",
       var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
       # Bootstrap of standard errors for ATE
       p_se_boot_ate=None,
       # Conditional mean & variances are used
       p_cond_var=True,
       # Specifies k-NN method
       p_knn=True
   )
   
   my_mcf.train(training_df)
   results, _ = my_mcf.predict(prediction_df)

