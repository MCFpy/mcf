Training a Modified Causal Forest
=================================

Regression tree splits the data into non-overlapping strata thanks to its non-parametric nature. The average of the dependent variable within each strata serve as the prediction for observations with similar covariate values. An issue with this approach is that using discrete, non-overlapping data splits can be inefficient, since valuable information from neighboring data points is not used. Furthermore, the curse of dimensionality makes it challenging to consistently fit stable splits with good performance. Lastly, the number of potential data splits grows significantly as the number of covariates increases, which can lead to exponential increases in computing time if all possible splits are evaluated at each node of the tree.

Random forest (to some extent) solves these problems by building many decorrelated trees and averaging their predictions. To do so, different random samples of the data, generated by bootstrapping or subsampling, and random subsets of covariates for each splitting decision in an individual leaf are used to build each tree. 

The **mcf** may be seen as a regression forest with a different splitting criterion. However, the underlying prediction task is fundamentally different. 
The **mcf** aims to predict causal effects, for which there is no data. It also provides (asymptotically) valid inference. 
To impute the missing data, the **mcf** requires a causal model. 
To provide valid inference, the **mcf** borrows the concept of honesty introduced by `Athey & Imbens (2016) <https://doi.org/10.1073/pnas.1510489113>`_. 
For a textbook-like discussion refer to `Athey & Imbens (2016) <https://www.pnas.org/doi/10.1073/pnas.1510489113>`_. **Note (1)**, the **mcf** differs from the causal forest of `Wager & Athey (2018) <https://doi.org/10.1080/01621459.2017.1319839>`_ with respect to the splitting criterion when growing the forest. Setting ``cf_mce_vart`` to ``2``, you may switch to the splitting rule of  `Wager & Athey (2018) <https://doi.org/10.1080/01621459.2017.1319839>`_. 

Forest Growing
------------------------------------

As a tree is grown, the algorithm greedily chooses the split which yields the best possible reduction of the objective function specified in ``cf_mce_vart``. The following objective criteria are implemented:

- Outcome Mean Squared Error (MSE)

- Outcome MSE and Mean Correlated Errors (MCE) 

- Variance of the Effect

- Random Switching: criterion randomly switches between outcome MSE and MCE and penalty functions which are defined under ``cf_p_diff_penalty``.

The outcome MSE is estimated as the sum of mean squared errors of the outcome regression in each treatment. 
The MCE depends on correlations between treatment states. For this reason, before building the trees, for each observation in each treatment state, the program finds a close ‘neighbor’ in every other treatment state and saves its outcome to then estimate the MCE. 

How the program matches is governed by the argument ``cf_match_nn_prog_score``. 
The program matches either by outcome scores (one per treatment) or on all covariates by Mahalanobis matching. If there are many covariates, it is advisable to match on outcome scores due to the curse of dimensionality. When performing Mahalanobis matching, a practical issue may be that the required inverse of the covariance matrix is unstable. For this reason the program allows to only use the main diagonal to invert the covariance matrix. This is regulated via the argument ``cf_nn_main_diag_only``. 

The program also allows for modification of the splitting rule by adding a penalty to the objective function, as specified by the ``cf_mce_vart`` argument. The purpose of using a penalty based on the propensity score is to reduce selection bias by promoting greater treatment homogeneity within newly formed splits.

The type of penalty can be controlled using the ``cf_penalty_type`` keyword, which supports two options:

1. 'mse_d' (default): Computes the MSE of the propensity scores in the daughter leaves.
2. 'diff_d': Calculates the penalty as the squared differences between treatment propensities in the daughter leaves.
    
For both options, you can define a multiplier using the ``cf_p_diff_penalty`` keyword to adjust the penalty's impact. An advantage of the 'mse_d' option is that it can be computed using the out-of-bag observations, making it useful when tuning the forest. 

**Note (2)**, the random switching option (3) in ``cf_mce_vart`` requires a penalty to function properly, as it does not work without one.

Once the forest is ready for training, the splits obtained in the training dataset are transferred to all data subsamples (by treatment state) in the held-out data set. Finally, the mean of the outcomes in each leaf is the prediction.

Below you find a list of the discussed parameters that are relevant for forest growing. Please consult the :py:class:`API <mcf_functions.ModifiedCausalForest>` for more details or additional parameters. 

+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameter                 | Description                                                                                                                                                                                                     |
+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``cf_mce_vart``           | Splitting rule for tree building, 0 for MSE, 1 for MSE+MCE, 2 for heterogeneity maximization, or 3 for random switching between MSE, MCE and penalty function defined in ``cf_p_diff_penalty`` . Default is 1.  |
+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``cf_p_diff_penalty``     | Penalty function used during tree building, dependent on ``cf_mce_vart``. Default is None.                                                                                                                      |
+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``cf_match_nn_prog_score``| Choice of method of nearest neighbour matching. True: Prognostic scores. False: Inverse of covariance matrix of features. Default (or None) is True.                                                            |
+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``cf_nn_main_diag_only``  | Nearest neighbour matching: Use main diagonal of covariance matrix only. Only relevant if match_nn_prog_score == False. Default (or None) is False.                                                             |
+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+



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
        # Determine splitting rule when growing trees
        cf_mce_vart = 3,
        # Determine penalty function
        cf_p_diff_penalty = 3,
        # Determine method of nearest neighbour matching
        cf_match_nn_prog_score = True,
        # Type of penalty function
        cf_penalty_type='mse_d'
    )

