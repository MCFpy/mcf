=========================
Optimal Policy Allocation
=========================

The evaluation of the IATEs makes it possible to detect potential heterogeneous effects across sub-groups of the population.
If heterogeneity is observed, certain individuals may either benefit or not from a particular treatment.
To address this, the mcf introduces the :py:class:`~optpolicy_main.OptimalPolicy` class.

To determine an optimal policy allocation, you can choose between three methods:

- **Policy Tree**: This method bases on a tree-search algorithm, designed to construct an optimal policy tree.
- **Best Policy Score**: This method conducts allocations by directly using the policy scores.
- **Best Policy Score Classifier**: Learns to predict the treatment assignments
  obtained by the best-policy-score method from the decision variables.
  It can then assign treatments to new observations without requiring
  their policy scores.

Policy allocation algorithms
============================

Algorithm 1: Policy Tree
------------------------
For using this method, set ``gen_method`` to ``policy_tree``.

This method is a tree-search algorithm designed to construct a policy tree.
The search optimizes over the implemented candidate splits under the selected depth and leaf restrictions. Ordered grids and approximations for unordered variables can exclude possible splits.
The optimal tree maximises the value function (or welfare), computed as the sum of the individual policy scores, such as potential outcomes or treatment-aligned gains relative to a common baseline, by assigning all observations in a terminal leaf node to a single treatment.
If restrictions are specified, then they are incorporated into treatment specific cost parameters.

While the basic logic follows `Zhou, Athey, and Wager (2022) <https://doi.org/10.1287/opre.2022.2271>`_, the details in the programmatic implementation differ.
Treatment costs and requested treatment shares enter the implemented search as described below.

Inputs
~~~~~~
- :math:`{(X_i, \hat{\Theta}_i(j))}_{i=1}^{n}` : A set of observations where :math:`X_i` represents the features of the :math:`i`-th observation and :math:`\hat{\Theta}_i(j)` represents the potential outcome for each observation :math:`i` for each treatment :math:`j`.
- :math:`L`: Remaining split depth plus one in the recursion below. The public depth parameters count splits from root to leaf.
- :math:`p_1`: The number of ordered features.
- :math:`p_2` : The number of unordered features.

Outputs
~~~~~~~

- :math:`\mathcal{R}` : The reward, which is the maximum cumulative potential outcome.
- :math:`\mathcal{T}` : The policy tree constructed through the algorithm.

Purpose
~~~~~~~

The algorithm aims at constructing a policy tree that maximizes the cumulative potential outcome by selecting the best treatments at each node and splitting the data in a way that optimally partitions it based on the features.
This process is akin to building a decision tree, where each node represents a decision (a split based on a feature) and the leaves represent the final decision (the best treatment).
The goal is to maximize the overall outcome, making the best possible decisions at each step.

Steps
~~~~~

Here is a step-by-step explanation on how ``policy_tree`` works:

1. Case :math:`L = 1`, i.e., no further splits are possible. The algorithm defines the best treatment by summing the potential outcomes for each treatment across all observations and selecting the treatment that maximizes this sum. The algorithm returns this maximum sum of potential outcomes across all treatments and the corresponding treatment.

2. Case :math:`L > 1`, i.e., further splits are possible. The algorithm initializes the reward :math:`\mathcal{R}` to negative infinity and the policy tree :math:`\mathcal{T}` to empty. Loop over all features :math:`X_{m,i}` where :math:`(m = 1, 2, \ldots, p_1 + p_2)`:

   For each feature, consider supported candidate split points:
     - Split the data into two sets: left and right, based on the split value.
     - Recursively apply the tree search algorithm to both sets, reducing the depth :math:`L` by 1.
     - Compute the rewards for the left and right splits.
     - If the sum of the rewards from the left and right splits exceeds the current maximum reward :math:`\mathcal{R}`, update :math:`\mathcal{R}` and :math:`\mathcal{T}` to reflect the new best split.

   After considering all features and all candidate splits, return the best reward and the corresponding policy tree.

Example
~~~~~~~

This configuration uses simulated potential outcomes as demonstration scores.
See :doc:`../user_guide/optimal-policy_example` for fitting and evaluation.

.. code-block:: python

    from mcf.example_data import example_data
    from mcf.optpolicy_main import OptimalPolicy

    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()

    my_policy_tree = OptimalPolicy(
        var_d_name='treat',
        var_polscore_name=['y_pot0', 'y_pot1', 'y_pot2'],
        var_x_name_ord=['x_cont0', 'x_cont1', 'x_ord1'],
        # Select the Policy Tree method
        gen_method='policy_tree'
        )


Algorithm 2: Best Policy Score
------------------------------

To opt for this method, set ``gen_method`` to ``best_policy_score``.
Note that this is the **default method**.

Without share restrictions, this method assigns units to the treatment with the highest supplied policy score net of costs. 
This algorithm is computationally cheap, but comes with the downside of a low interpretability for the allocation rules.

Example
~~~~~~~

.. code-block:: python

    from mcf.example_data import example_data
    from mcf.optpolicy_main import OptimalPolicy

    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()

    # Create an instance of the OptimalPolicy class:
    my_optimal_policy = OptimalPolicy(
        var_d_name='treat',
        var_polscore_name=['y_pot0', 'y_pot1', 'y_pot2'],
        var_x_name_ord=['x_cont0', 'x_cont1', 'x_ord1'],
        var_x_name_unord=['x_unord0'],
        # Select the Best Policy Score method
        gen_method='best_policy_score'
        )


Algorithm 3: Best Policy Score Classifier
-----------------------------------------

To opt for this method, set ``gen_method`` to ``bps_classifier``.

This method learns a feature-based allocation rule from best-score training labels. Classifier predictions need not satisfy the shares imposed on those labels.

On a high level, this method uses the allocations obtained by the previous Best Policy Score method and trains classifiers.
The output is a decision rule that depends on features only and does not require knowledge about the policy scores.


Parameters for the Optimal Policy Tree
============================================

You can adjust different parameters defined in the :py:class:`~optpolicy_main.OptimalPolicy` class.

General parameters
------------------

``pt_min_leaf_size`` sets the minimum number of training observations
allowed in a policy-tree leaf. If not specified, it is determined
automatically from the training sample size, the tree depths and any
treatment-share restrictions. Larger values can reduce computation
time and limit overfitting.

``other_max_shares`` specifies a maximum allocation share for each
treatment. Provide one value between zero and one per treatment, in
the order of ``var_polscore_name``. The shares must sum to at least one
and allow a feasible allocation of whole observations.

How these restrictions are applied depends on the selected method:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Method
     - Treatment-share restrictions
   * - ``best_policy_score``
     - Returns allocations that respect the treatment limits.
   * - ``policy_tree``
     - Adjusts treatment costs to encourage the requested shares, but the resulting tree may exceed them. ``pt_enforce_restriction`` does not guarantee compliance and is disabled when ``pt_depth_tree_2`` is greater than zero.
   * - ``bps_classifier``
     - Learns from assignments that reflect the restrictions, but its predicted assignments may exceed the requested shares.

``other_costs_of_treat=None`` sets base costs to zero. Restricted tree fitting
can calibrate additional costs. Costs must be in the same units as policy scores.
``other_costs_of_treat_mult`` requires a sequence of finite positive values,
one per treatment; a scalar is not accepted. A multiplier above one increases a
positive calibrated cost increment and discourages assignment to that treatment. See the :py:class:`API <optpolicy_main.OptimalPolicy>` for details.

Example
~~~~~~~

.. code-block:: python

   from mcf.example_data import example_data
   from mcf.optpolicy_main import OptimalPolicy

   # Generate example data using the built-in function `example_data()`
   training_df, prediction_df, name_dict = example_data()

   my_policy_tree = OptimalPolicy(
       var_d_name='treat',
       var_polscore_name=['y_pot0', 'y_pot1', 'y_pot2'],
       var_x_name_ord=['x_cont0', 'x_cont1', 'x_ord1'],
       gen_method='policy_tree',
       # Minimum leaf size
       pt_min_leaf_size = None,
       # Maximum share allowed for each treatment (as many elements as treatment (d))
       other_max_shares = (1,1,1),
       # Treatment specific costs
       other_costs_of_treat = None,
       # Multiplier of automatically determined cost values
       other_costs_of_treat_mult = None
       )

Parameters for computational speed
----------------------------------

Additionally, you can control certain aspects of the algorithm which impact running time:

- **Tree Depth**: You can specify the depth of the trees via the keyword arguments ``pt_depth_tree_1`` and ``pt_depth_tree_2``.

  - ``pt_depth_tree_1`` defines the depth of the first optimal tree. The default is 3. Tree depth is defined such that depth d permits at most 2**d leaves; fitted trees may be smaller.

  - ``pt_depth_tree_2`` defines the depth of the second optimal tree, which builds upon the strata obtained from the leaves of the first tree. If ``pt_depth_tree_2`` is set to 0, the second tree is not built. The default is 1. Together with the default for ``pt_depth_tree_1``, this permits at most four split levels and 16 leaves. The sequential search need not match a single search at the combined depth. Tree depth is defined in the same way as for ``pt_depth_tree_1``.

- **Number of Evaluation Points**: ``pt_no_of_evalupoints`` parameter specifies the number of evaluation points for continuous variables during the tree search. It determines how many of the possible splits in the feature space are considered. If the value of ``pt_no_of_evalupoints`` is smaller than the number of distinct values of a certain feature, the algorithm visits fewer splits, thus increasing computational efficiency. However, a lower value may also deviate more from the optimal splitting rule. This parameter is closely related to the approximation parameter of `Zhou, Athey, and Wager (2022) <https://doi.org/10.1287/opre.2022.2271>`_ . This parameter is only relevant if ``gen_method`` is ``policy_tree``. The default value (or None) is 100.

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
