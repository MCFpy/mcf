Training a Modified Causal Forest
=================================

Random forests are a group of decorrelated regression trees. Due to its non-parametric nature, the regression tree splits the data into non-overlapping strata. Subsequently, it computes the average of the dependent variable within each strata. These averages serve as the prediction for observations with similar covariate values. An issue with this approach is that using discrete, non-overlapping data splits can be inefficient, as it doesn't use information from neighboring data points. Additionally, the curse of dimensionality makes it difficult to fit stable splits (or 'trees') with consistently good performance. Moreover, as the number of covariates increases, the number of potential data splits grows significantly. This can lead to exponential increases in computing time if all possible splits are evaluated at each node of the tree.

However, random forests solve these problems (to some extent) by building many decorrelated trees and averaging their predictions. This is achieved by using different random samples of the data to build each tree, generated by bootstrapping or subsampling, as well as random subsets of covariates for each splitting decision in an individual leaf within a developing tree. 

**Note**, the **mcf** differs from the causal forest of `Wager & Athey (2018) <https://doi.org/10.1080/01621459.2017.1319839>`_ with respect to the splitting criterion when growing the forest. 
Setting ``cf_mce_vart`` to ``2``, you may switch to the splitting rule of  `Wager & Athey (2018) <https://doi.org/10.1080/01621459.2017.1319839>`_. 

Apart from the difference in the splitting criterion, the regression forest may seem much related to the **mcf**. 
However, note that the underlying prediction tasks are fundamentally different. 
The **mcf** aims to predict causal effects, for which there is no data, and provides (asymptotically) valid inference. 
To impute the missing data, the **mcf** requires a causal model. 
To provide valid inference, the **mcf** borrows the concept of honesty introduced by `Athey & Imbens (2016) <https://doi.org/10.1073/pnas.1510489113>`_. 
For a textbook-like discussion refer to `Athey & Imbens (2016) <https://www.pnas.org/doi/10.1073/pnas.1510489113>`_.

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

Note that the random switching option (3) in ``cf_mce_vart`` requires a penalty to function properly, as it does not work without one.

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

.. _computational-speed:

Parameter tuning and Computational Speed
-------------------------------------------

The **mcf** allows for a grid search mainly over tree types of parameters: 

- Number of variables drawn at each split

- Alpha-Regularity

- Minimum leaf size

In practical terms, a forest is estimated for all possible combinations of these parameters, with a fixed random seed. Below are the main parameters that you can adjust to either tune your forest or increase computational speed.


- **Forest Growing and Subsampling**: 

  - ``cf_boot`` defines the number of trees forming the forest. The larger number will increase processing time. 

  - ``cf_m_share_min`` determines the minimum share of variables used at each new split of tree. 

  - ``cf_m_share_max`` sets the maximum share of variables used at each new split of tree. 

  - ``cf_m_grid`` this parameter determines the number of variables used at each new split of the tree. If grid is used, optimal value is determined by out-of-bag estimation of objective function. The default value is 1. **Note**: The finer the grid-search, the more forests are estimated, which slows down computation time. To identify the best values from the grid-search, the program implements the out-of-bag estimation of the chosen objective. The best performing forest based on its out-of-bag value of its objective function is taken for further computations.

  - ``cf_n_min_min`` smallest minimum leaf size. Decreasing this minimum leaf size prolongs computation time as it prompts the tree to grow deeper. This augmentation in computation time can be significant with extensive datasets.

  - ``cf_n_min_max`` largest minimum leaf size.  Similar to its counterpart, adjusting this parameter influences computation time.

  - ``cf_chunks_maxsize`` this parameter randomly splits training data in chunks and takes the average of the estimated parameters to improve scalability. This can increase speed and reduce memory demand, but may slightly increase finite sample bias. If ``cf_chunks_maxsize`` is larger than sample size, there is no random splitting. 

  - ``cf_subsample_factor_eval`` this parameter determines the fraction of the data to be used for evaluation.  When it's set to False, no subsampling is performed in the evaluation subsample. If it's set to True or None, the subsample size used for tree building is employed, which helps to avoid too many empty leaves. If a float value greater than 0 is provided, it's used as a multiplier of the subsample size for tree building.  This parameter is particularly useful for larger samples, as using subsampling during evaluation can speed up computations and reduce memory demand. It also increases the speed at which asymptotic bias disappears, albeit at the expense of a slower reduction in variance. However, simulations so far show no significant impact from this trade-off. 

  - ``cf_random_thresholds`` this option can be used to enable the use of random thresholds in the decision trees, which can speed up the tree generation process. If this parameter is set to a value greater than 0, the program doesn't examine all possible split values of ordered variables. Instead, it only checks a number of random thresholds, with a new randomization for each split. A value of 0 for this parameter means no random thresholds are used. A value greater than 0 specifies the number of random thresholds used for ordered variables. Using fewer thresholds can speed up the program, but it might lead to less accurate results.

  - ``p_choice_based_sampling`` this option allows choice-based sampling to speed up programme if treatment groups have very different sizes.

  - ``cf_tune_all``: Tune all parameters. If True, all *_grid keywords will be set to 3. User specified values are respected if larger than 3. Default (or None) is False.

- **Parallel Processing**: 

  - ``gen_mp_parallel`` defines the number of parallel processes. A smaller value will slow down the program and reduce its demand on RAM. The default value is None, which means 80% of logical cores. If you run into memory problems, reduce the number of parallel processes.


Please refer to the :py:class:`API <mcf_functions.ModifiedCausalForest>` for a detailed description of these and other options. 

Adjusting these options can help to significantly reduce the computational time, but it may also affect the accuracy of the results. Therefore, it is recommended to understand the implications of each option before adjusting them. Below you find a list of the discussed parameters that are relevant for parameter tuning and computational speed.

**Note:** The **mcf** achieves faster performance when binary features, such as gender, are defined as ordered, using ``var_x_name_ord`` instead of ``var_x_name_unord``.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1

   * - Argument
     - Description
   * - ``cf_boot``
     - Number of trees forming the forest. Default is 1000.
   * - ``cf_m_share_min``
     - Minimum share of variables used at each new split of tree. Default is 0.1.
   * - ``cf_m_share_max``
     - Maximum share of variables used at each new split of tree. Default is 0.6.
   * - ``cf_m_grid``
     - Number of variables used at each new split of tree. Default is 1.
   * - ``cf_n_min_min``
     - Smallest minimum leaf size. Default is None.
   * - ``cf_n_min_max``
     - Largest minimum leaf size. Default is None.
   * - ``cf_chunks_maxsize``
     - Randomly splits training data in chunks and averages the estimated parameters (improved scalability). Default is None. 
   * - ``cf_subsample_factor_eval``
     - Subsampling to reduce the size of the dataset to process. Default is None. 
   * - ``cf_random_thresholds``
     - Enable the use of random thresholds in the decision trees. Default is None. 
   * - ``p_choice_based_sampling``
     -  Choice based sampling to speed up programme if treatment groups have different sizes. Default is False. 
   * - ``cf_tune_all``
     - Tune all parameters. If True, all *_grid keywords will be set to 3. User specified values are respected if larger than 3. Default (or None) is False.
   * - ``gen_mp_parallel``
     -  Number of parallel processes. Default is 80%.




Example
~~~~~~~

.. code-block:: python

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        # Number of trees
        cf_boot = 500,
        # Maximum share of variables used at each new split of tree
        cf_m_share_max = 0.6,
        # Minimum share of variables used at each new split of tree
        cf_m_share_min = 0.15,
        # Number of variables used at each new split of tree
        cf_m_grid = 2,
        # Smallest minimum leaf size
        cf_n_min_min = 5,
        # Largest minimum leaf size
        cf_n_min_max=None,
        # Number of parallel processes
        gen_mp_parallel=None,
        # Tune all parameters
        cf_tune_all=True
    )


