=========================
Optimal Policy Allocation
=========================

The evaluation of the IATEs makes it possible to detect potential heterogeneous effects across sub-groups of the population.
If heterogeneity is observed, certain individuals may either benefit or not from a particular treatment. 
To address this, the mcf introduces the :py:class:`~optpolicy_functions.OptimalPolicy` class.

To determine an optimal policy allocation, you can choose between three methods:

- **Policy Tree**: This method bases on a tree-search algorithm, designed to construct an optimal policy tree. 
- **Blackbox Rule**: This method conducts Black Box allocations obtained by using the policy scores in a direct manner.
- **Best Score**: This method is experimental; soon to be discussed further.

Policy allocation algorithms
============================

Algorithm 1: Policy Tree
------------------------
To opt for this method, set ``gen_method`` to ``policy tree``.

This method is a tree-search algorithm designed to construct a policy tree.
The implemented policy tree is the optimal tree among all possible trees and is found by looking for the tree that leads to a best performance.
The optimal tree maximises the value function (or welfare), computed as the sum of the individual policy scores, such as potential outcomes or :math:`IATEs`, by assigning all observations in a terminal leaf node to a single treatment.
If restrictions are specified, then they are incorporated into treatment specific cost parameters.

While the basic logic follows `Zhou, Athey, and Wager (2022) <https://doi.org/10.1287/opre.2022.2271>`_ , the details of the programmatic implementation differ. 
For instance, in contrast to policytree, the optpoltree allows you to consider constraints regarding the maximal shares of treated observations, treatment costs, and different policy scores.

Let us look into this method further:

Inputs
~~~~~~
- :math:`({(X_i, \hat{\Theta}_i(j))}_{i=1}^{n})` : A set of observations where :math:`X_i` represents the features of the :math:`i`-th observation and :math:`\hat{\Theta}_i(j)` represents the potential outcome for each observation :math:`i` for each treatment :math:`j`.
- :math:`(L)` : An integer indicating the depth of the tree plus one.
- :math:`(p_1)`: The number of ordered features.
- :math:`(p_2)` : The number of unordered features.

Outputs
~~~~~~~

- :math:`(\mathcal{R})` : The reward, which is the maximum cumulative potential outcome.
- :math:`(\mathcal{T})` : The policy tree constructed through the algorithm.

Purpose
~~~~~~~

The algorithm aims at constructing a policy tree that maximizes the cumulative potential outcome by selecting the best treatments at each node and splitting the data in a way that optimally partitions it based on the features.
This process is akin to building a decision tree, where each node represents a decision (a split based on a feature) and the leaves represent the final decision (the best treatment).
The goal is to maximize the overall outcome, making the best possible decisions at each step.

Steps
~~~~~

Here is a step-by-step explanation on how the Policy Tree works:

1. Case :math:`(L = 1)` : no further splits are possible and the algorithm returns the maximum sum of potential outcomes across all treatments alongside the treatment that achieves this maximum.
   Essentially, the algorithm computes the best possible treatment without further splitting. It does so by summing the potential outcomes for each treatment across all observations and selecting the treatment that maximizes this sum.

2. Case :math:`(L > 1)` : initialize the reward :math:`(\mathcal{R})` to negative infinity and the tree :math:`(\mathcal{T})` to empty. Loop over all features :math:`(m = 1, 2, \ldots, p_1 + p_2)`:

   For each feature, consider all possible split points:
     - Split the data into two sets: left and right, based on the split value.
     - Recursively apply the tree search algorithm to both sets, reducing the depth :math:`(L)` by 1.
     - Compute the rewards for the left and right splits.
     - If the sum of the rewards from the left and right splits exceeds the current maximum reward :math:`(\mathcal{R})`, update :math:`(\mathcal{R})` and :math:`(\mathcal{T})` to reflect the new best split.

   After considering all features and all possible splits, return the best reward and the corresponding policy tree.
    Essentially, the algorithm explores potential splits of the data by looping over all features. 
    For each feature, it considers sorted values of ordered features or unique categories of categorical features as potential split points.
    For each split point, it divides the data into left and right subsets and applies the tree search recursively on these subsets with depth :math:`(L - 1)`.
    The rewards from the left and right recursive calls are summed to determine the effectiveness of the split.
    If a new split yields a higher reward than the current best, the algorithm updates the reward and the structure of the policy tree.

Example
~~~~~~~

.. code-block:: python
        
    from mcf.example_data_functions import example_data
    from mcf.optpolicy_functions import OptimalPolicy
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
    my_policy_tree = OptimalPolicy(
        var_d_name='treat',
        var_polscore_name=['y_pot0', 'y_pot1', 'y_pot2'],
        var_x_name_ord=['x_cont0', 'x_cont1', 'x_ord1'],
        # Select the Policy Tree method
        gen_method='policy tree'
        )


Algorithm 2: Best Policy Score
------------------------------

To use this method, set ``gen_method`` to ``best_policy_score``. **Note** that this is the default method.

Very simply, this method assigns units to the treatment with the highest estimated potential outcome. 
This algorithm is computationally cheap, yet it lacks clear interpretability for the allocation rules, which makes it challenging for policymakers to adopt.

Example
~~~~~~~
       
.. code-block:: python
        
    from mcf.example_data_functions import example_data
    from mcf.optpolicy_functions import OptimalPolicy

    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()

    # Create an instance of the OptimalPolicy class:
    my_optimal_policy = OptimalPolicy(
        var_d_name='treat',
        var_polscore_name=['y_pot0', 'y_pot1', 'y_pot2'],
        var_x_name_ord=['x_cont0', 'x_cont1', 'x_ord1'],
        var_x_name_unord=['x_unord0'],
        # Select the Best Policy Score method
        gen_method='best_policy_score',
        pt_depth_tree_1=2
        )


Algorithm 3: bps Classifier
---------------------------

To use this method, set ``gen_method`` to ``bps_classifier``.

For the moment, this is an experimental feature and will soon be further discussed.

On a high level, this method uses the allocations obtained by ``best_policy_score`` and trains classifiers. 
The output is a decision rule that depends on features only and does not require knowledge of the policy scores.


Options for Optimal Policy Tree
===============================

You can personalize various parameters defined in the :py:class:`~optpolicy_functions.OptimalPolicy` class. 

You can use the ``var_effect_vs_0_se`` parameter to specify the standard errors of variables of effects of treatment relative to first treatment. Dimension is equal to the number of treatments minus 1. 

To control how many observations are required at minimum in a partition, you can define such number by using ``pt_min_leaf_size``. Leaves that are smaller than ``pt_min_leaf_size`` in the training data will not be considered. A larger number reduces computation time and avoids overfitting. Default is :math:`0.1 \times \frac{\text{{number of training observations}}}{\text{{number of leaves}}}`. 

If the number of individuals who receive a specific treatment is constrained, you may specify admissible treatment shares via the keyword argument ``other_max_shares``. Note that the information must come as a tuple with as many entries as there are treatments.

When considering treatment costs, input them via ``other_costs_of_treat``.  When evaluating the reward, the aggregate costs (costs per unit times units) of the policy allocation are subtracted. If left as default (None), the program determines a cost vector that imply an optimal reward (policy score minus costs) for each individual, while guaranteeing that the restrictions as specified in ``other_max_shares`` are satisfied. This is only relevant when ``other_max_shares`` is specified.

Alternatively, if restrictions are present and ``other_costs_of_treat`` is left to its default, you can specify ``other_costs_of_treat_mult``. Admissible values for this parameter are either a scalar greater zero or a tuple with values greater zero. The tuple needs as many entries as there are treatments. The imputed cost vector is then multiplied by this factor.

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Keyword
     - Details
   * - ``var_effect_vs_0_se``
     - Standard errors of effects relative to treatment zero. Dimension is equal to the number of treatments minus 1. Default is None.
   * - ``pt_min_leaf_size``
     - Minimum leaf size. Leaves that are smaller will not be considered. A larger number reduces computation time and avoids some overfitting. Only relevant if ``gen_method`` is ``policy tree``. Default is None.
   * - ``other_max_shares``
     - Maximum share allowed for each treatment. Note that the information must come as a tuple with as many entries as there are treatments. Default is None.
   * - ``other_costs_of_treat``
     - Treatment specific costs. Subtracted from policy scores. None (when there are no constraints): 0 None (when there are constraints): Costs will be automatically determined such as to enforce constraints in the training data by finding cost values that lead to an allocation (``best_policy_score``) that fulfils restrictions ``other_max_shares``. Default is None.
   * - ``other_costs_of_treat_mult``
     - Multiplier of automatically determined cost values. Use only when automatic costs violate the constraints given by ``other_max_shares``. This allows to increase :math:`(>1)` or decrease :math:`(<1)` the share of treated in particular treatment. Default is None.

Please consult the :py:class:`API <mcf_functions.ModifiedCausalForest>` for more details or additional parameters. 

Example
-------

.. code-block:: python

   from mcf.example_data_functions import example_data
   from mcf.optpolicy_functions import OptimalPolicy
   
   # Generate example data using the built-in function `example_data()`
   training_df, prediction_df, name_dict = example_data()
   
   my_policy_tree = OptimalPolicy(
       var_d_name='treat',
       var_polscore_name=['y_pot0', 'y_pot1', 'y_pot2'],
       var_x_name_ord=['x_cont0', 'x_cont1', 'x_ord1'],
       gen_method='policy tree',
       #  Effects of treatment relative to treatment zero
       var_effect_vs_0 = ['iate1vs0', 'iate2vs0'], 
       # Minimum leaf size
       pt_min_leaf_size = None,
       # Maximum share allowed for each treatment (as many elements as treatment (d))
       other_max_shares = (1,1,1),
       # Treatment specific costs
       other_costs_of_treat = None,
       # Multiplier of automatically determined cost values
       other_costs_of_treat_mult = None
       )

Computational speed
===================

Additionally, you can control certain aspects of the algorithm which impact running time:

- **Tree Depth**: You can specify the depth of the trees via the keyword arguments ``pt_depth_tree_1`` and ``pt_depth_tree_2``. 

  - ``pt_depth_tree_1`` defines the depth of the first optimal tree. The default is 3. Tree depth is defined such that a depth of 1 implies 2 leaves, a depth of 2 implies 4 leaves, a depth of 3 implies 8 leaves, etc.

  - ``pt_depth_tree_2`` defines the depth of the second optimal tree, which builds upon the strata obtained from the leaves of the first tree. If ``pt_depth_tree_2`` is set to 0, the second tree is not built. The default is 1. Together with the default for ``pt_depth_tree_1``, this leads to a total tree of depth 4 (which is not optimal). Tree depth is defined in the same way as for ``pt_depth_tree_1``.

- **Number of Evaluation Points**: ``pt_no_of_evalupoints`` parameter specifies the number of evaluation points for continuous variables during the tree search. It determines how many of the possible splits in the feature space are considered. If the value of ``pt_no_of_evalupoints`` is smaller than the number of distinct values of a certain feature, the algorithm visits fewer splits, thus increasing computational efficiency. However, a lower value may also deviate more from the optimal splitting rule. This parameter is closely related to the approximation parameter of `Zhou, Athey, and Wager (2022) <https://doi.org/10.1287/opre.2022.2271>`_ . This parameter is only relevant if ``gen_method`` is ``policy tree``. The default value (or None) is 100.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1

   * - Keyword
     - Details
   * - ``pt_depth_tree_1``
     -   Depth of 1st optimal tree. Default is 3. 
   * - ``pt_depth_tree_2``
     -   Depth of 2nd optimal tree. Default is 1. 
   * - ``pt_no_of_evalupoints``
     -   Number of evaluation points for continous variables. Default is 100.
