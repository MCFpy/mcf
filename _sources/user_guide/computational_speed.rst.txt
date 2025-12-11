.. _computational-speed:

Computational Speed and Ressources for Effect Estimation
========================================================

This section provides key considerations regarding computation and resource management. It includes speed- and resource-related information necessary for tuning the forest via grid search, setting parameter values to optimize runtime, and reducing RAM consumption.


Forest Tuning via Grid Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **mcf** allows for a grid search mainly over three types of parameters: 

- Number of variables drawn at each split

- Alpha-Regularity

- Minimum leaf size

In practical terms, a forest is estimated for all possible combinations of these parameters, with a fixed random seed. Below are the main parameters that you can adjust to either tune your forest or increase computational speed.

**Forest Growing and Subsampling**

- ``cf_boot`` defines the number of trees forming the forest. The larger number will increase processing time. 

- ``cf_m_share_min`` determines the minimum share of variables used at each new split of tree. 

- ``cf_m_share_max`` sets the maximum share of variables used at each new split of tree. 

- ``cf_m_grid`` this parameter determines the number of variables used at each new split of the tree. If grid is used, optimal value is determined by out-of-bag estimation of objective function. The default value is 1. The finer the grid-search, the more forests are estimated, which slows down computation time. To identify the best values from the grid-search, the program implements the out-of-bag estimation of the chosen objective. The best performing forest based on its out-of-bag value of its objective function is taken for further computations.

- ``cf_n_min_min`` smallest minimum leaf size. Decreasing this minimum leaf size prolongs computation time as it prompts the tree to grow deeper. This augmentation in computation time can be significant with extensive datasets.

- ``cf_n_min_max`` largest minimum leaf size.  Similar to its counterpart, adjusting this parameter influences computation time.

- ``cf_chunks_maxsize`` this parameter randomly splits training data in chunks and takes the average of the estimated parameters to improve scalability. This can increase speed and reduce memory demand, but may slightly increase finite sample bias. If ``cf_chunks_maxsize`` is larger than sample size, there is no random splitting. 

- ``cf_subsample_factor_eval`` this parameter determines the fraction of the data to be used for evaluation.  When it's set to False, no subsampling is performed in the evaluation subsample. If it's set to True or None, the subsample size used for tree building is employed, which helps to avoid too many empty leaves. If a float value greater than 0 is provided, it's used as a multiplier of the subsample size for tree building.  This parameter is particularly useful for larger samples, as using subsampling during evaluation can speed up computations and reduce memory demand. It also increases the speed at which asymptotic bias disappears, albeit at the expense of a slower reduction in variance. However, simulations so far show no significant impact from this trade-off. 

- ``cf_random_thresholds`` this option can be used to enable the use of random thresholds in the decision trees, which can speed up the tree generation process. If this parameter is set to a value greater than 0, the program doesn't examine all possible split values of ordered variables. Instead, it only checks a number of random thresholds, with a new randomization for each split. A value of 0 for this parameter means no random thresholds are used. A value greater than 0 specifies the number of random thresholds used for ordered variables. Using fewer thresholds can speed up the program, but it might lead to less accurate results.

- ``p_choice_based_sampling`` this option allows choice-based sampling to speed up programme if treatment groups have very different sizes.

- ``cf_tune_all``: Tune all parameters. If True, all *_grid keywords will be set to 3. User specified values are respected if larger than 3. Default (or None) is False.

**Parallel Processing** 
  
- ``gen_mp_parallel`` defines the number of parallel processes. A smaller value will slow down the program and reduce its demand on RAM. The default value is None, which means 80% of logical cores. If you run into memory problems, reduce the number of parallel processes.

Minimization of RAM usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

When datasets are large, the computational burden (incl. demands on RAM) may increase rapidly. First of all, it is important to remember that the mcf estimation consists of two steps:

1. Train the forest with the training data (outcome, treatment, features);
2. Predict the effects with the prediction data (needs features only, or treatment and features if, e.g., treatment effects on the treated are estimated). 

The precision of the results is (almost) entirely determined by the training data, while the prediction data (mainly) defines the population which the ATE and other effects are computed for.

The **mcf** deals as follows with large training data: When the training data becomes larger than ``cf_chunks_maxsize``, the data is randomly split and for each split a new forest is estimated. In the prediction part, effects are estimated for each forest and subsequently averaged.
       
The **mcf** deals as follows with large prediction data: The critical part when computing the effects is the weight matrix. Its size is :math:`N_{Tf}` x :math:`N_{P}`, where :math:`N_{P}` is number of observations in the prediction data and :math:`N_{Tf}` is the number of observations used for forest estimation. The weight matrix is estimated for each forest (to save memory it is deleted from memory and stored on disk). Although the weight matrix uses (by default) a sparse data format, it may still be very large and it can be very time consuming to compute.

Reducing computation and demand on memory without much performance loss: Tests for very large data (1 million and more) have shown that indeed the prediction part becomes the bottleneck,while the training part computes reasonably fast. Therefore, one way to speed up the mcf and reduce the demand on RAM is to reduce the size of the prediction data (e.g. take a x% random sample). Tests have shown that, for this approach, effect estimates and standard errors remain very similar whether 1 million or only 100,000 prediction observations are used, even with 1 million training observations.
       
The keywords ``_int_max_obs_training``, ``_int_max_obs_prediction``, ``_int_max_obs_kmeans``, and ``_int_max_obs_post_rel_graphs`` allow one to set these parameters accordingly.


Please refer to the :py:class:`API <mcf_main.ModifiedCausalForest>` for a detailed description of these and other options. 

Adjusting these options can help to significantly reduce the computational time, but it may also affect the accuracy of the results. Therefore, it is recommended to understand the implications of each option before adjusting them. Below you find a list and a coding example indicating the discussed parameters that are relevant for parameter tuning and computational speed.

**Note**, the **mcf** achieves faster performance when binary features, such as gender, are defined as ordered, using ``var_x_name_ord`` instead of ``var_x_name_unord``.

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
     - ``_int_iate_chunk_size``
        Number of IATEs that are estimated in a single ray worker. Default is number of prediction observations / workers. If programme crashes in second part of IATE because of excess memory consumption, reduce _int_iate_chunk_size.
    - ``_int_weight_as_sparse_splits`` 
        Compute sparse weight matrix in several chuncks. Default:(Rows of prediction data * rows of Fill_y data) / (number of training splits * 25'000 * 25'000).
    - ``_int_max_obs_training``
        Upper limit for sample size. Reducing observations for training increases MSE and thus should be avoided. Default is infinity.
    - ``_int_max_obs_prediction`` 
        Upper limit for sample size. Reducing observations for prediction does not much affect MSE. It may reduce detectable heterogeneity, but may also dramatically reduce computation time. Default is 250'000.
    - ``_int_max_obs_kmeans`` 
        Upper limit for sample size. Reducing observations may reduce detectable heterogeneity, but also reduces computation time. Default is 200'000.
    - ``_int_max_obs_post_rel_graphs`` 
        Upper limit for sample size. Figures show the relation of IATEs and features (note that the built-in non-parametric regression is computationally intensive).Default is 50'000.

Example
~~~~~~~

.. code-block:: python


    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        # Number of trees
        cf_boot=500,
        # Maximum share of variables used at each new split of tree
        cf_m_share_max=0.6,
        # Minimum share of variables used at each new split of tree
        cf_m_share_min=0.15,
        # Number of variables used at each new split of tree
        cf_m_grid=2,
        # Smallest minimum leaf size
        cf_n_min_min=5,
        # Largest minimum leaf size
        cf_n_min_max=None,
        # Number of parallel processes
        gen_mp_parallel=None,
        # Tune all parameters
        cf_tune_all=True,
        # Smallest minimum leaf size
        _int_iate_chunk_size=None,  # Corrected here
        # Largest minimum leaf size
        _int_weight_as_sparse_splits=None,
        # Number of parallel processes
        _int_max_obs_training=None,
        # Tune all parameters
        _int_max_obs_prediction=None,
        # Number of parallel processes
        _int_max_obs_kmeans=None,
        # Tune all parameters
        _int_max_obs_post_rel_graphs=None,
    )
