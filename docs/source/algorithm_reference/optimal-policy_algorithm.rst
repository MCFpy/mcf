Policy Tree algorithm
=====================

To determine the policy allocation, you may choose between two methods:

- **Policy Tree**: This method follows `Zhou, Athey, and Wager (2022) <https://doi.org/10.1287/opre.2022.2271>`_ . To opt for this method, set ``gen_method`` to 'policy tree'. The implemented `policy tree` are optimal trees where all possible trees are checked if they lead to a better performance. If restrictions are specified, then they are incorporated into treatment specific cost parameters. 

- **Blackbox Rule**: To use this method, set ``gen_method`` to `best_policy_score`. which conducts Black Box allocations which are obtained by using the scores directly (potentially subject to restrictions). **Note** this is the default method. 


Optimal Policy Tree
-------------------

The :py:class:`~optpolicy_functions.OptimalPolicy` class is designed to discover the optimal policy tree in a computationally efficient and tractable way. While the basic logic follows `Zhou, Athey, and Wager (2022) <https://doi.org/10.1287/opre.2022.2271>`_ , the details of the programmatic implementation differ. 
For instance, in contrast to policytree, the optpoltree allows you to consider constraints regarding the maximal shares of treated observations, treatment costs and different policy scores.


Implementation
-----------------------------

The :py:class:`~optpolicy_functions.OptimalPolicy` class explores the space of all viable policy trees and picks the optimal one. This optimal tree maximizes the value function, computed as the sum of individual-specific policy scores, by assigning treatments to observations within terminal nodes.

Given a fixed choice of previous partitions, the problem of finding an optimal solution simplifies to solving two subproblems: 

- Finding optimal left and right subtrees. 

Once we have reached a terminal node, we are no longer allowed to perform splits of the feature space and the treatment which maximises the score of all observations in the respective leaf is chosen. 

This recursive approach breaks down the problem into smaller, more manageable subproblems, easing the overall solution.


Notation
----------------------------

Before we delve into the solution method to find the optimal policy tree (Tree-search Exact Algorithm), let's introduce some notation:

- :math:`i=1, \ldots, n`: are :math:`n` observations
- :math:`p_1`: number of ordered features 
- :math:`p_2`: number of unordered features
- :math:`M`: number of treatments
- :math:`\hat{\Theta}_i`: vector of estimated policy scores, the potential outcomes, for the :math:`M+1` distinct potential outcomes are stacked for each observation :math:`i`.
- :math:`\hat{\Theta}_i(d)`: potential outcome for observation :math:`i` for treatment :math:`d`.
- :math:`L`: depth of the tree, which equals the number of splitting nodes plus one.

With this notation, we can now describe the Tree-Search Exact algorithm.


Tree-search Exact Algorithm
-----------------------------

The Tree-search Exact algorithm can be described as follows:

1. If :math:`L = 1`:

   - Choose :math:`j^* \in \{0, 1, \ldots, M\}`, which maximizes :math:`\sum_i \hat{\Theta}_i(j)` and return the corresponding reward = :math:`\sum_{\forall i} \hat{\Theta}_i(j^*)`.

2. Else:

   - Initialize reward = :math:`-\infty`, and an empty tree = :math:`\emptyset` for all :math:`m = 1, \ldots, p_1 + p_2`.

   - Pick the m-th feature; for ordered features return the unique values observed and sorted; if unordered return the unique categories to derive all possible splits.

      a. Then, for all possible splitting values of the m-th feature split the sample accordingly into a sample_left and sample_right.
   
      b. :math:`(\text{reward left}, \text{tree left}) = \text{Tree-search}(\text{sample left}, L-1)`.
   
      c. :math:`(\text{reward right}, \text{tree right}) = \text{Tree-search}(\text{sample right}, L-1)`.

   - If :math:`\text{reward left} + \text{reward right} > \text{reward}`:

        a. :math:`\text{reward} = \text{reward left} + \text{reward right}`.
   
        b. :math:`\text{tree} = \text{Tree-search}(m, \text{splitting value}, \text{tree left}, \text{tree right})`.


Options for Optimal Policy Tree
-----------------------------------

You can personalize various parameters defined in the :py:class:`~optpolicy_functions.OptimalPolicy` class. 

You can use the ``var_effect_vs_0_se`` parameter to specify the standard errors of variables of effects of treatment relative to first treatment. Dimension is equal to the number of treatments minus 1. 

To control how many observations are required at minimum in a partition, you can define such number by using ``pt_min_leaf_size``. Leaves that are smaller than ``pt_min_leaf_size`` in the training data will not be considered. A larger number reduces computation time and avoids overfitting. Default is :math:`0.1 \times \frac{\text{{number of training observations}}}{\text{{number of leaves}}}`. 

If the number of individuals who receive a specific treatment is constrained, you may specify admissible treatment shares via the keyword argument ``other_max_shares``. Note that the information must come as a tuple with as many entries as there are treatments.

When considering treatment costs, input them via ``other_costs_of_treat``.  When evaluating the reward, the aggregate costs (costs per unit times units) of the policy allocation are subtracted. If left as default (None), the program determines a cost vector that imply an optimal reward (policy score minus costs) for each individual, while guaranteeing that the restrictions as specified in ``other_max_shares`` are satisfied. This is only relevant when ``other_max_shares`` is specified.

Alternatively, if restrictions are present and `other_costs_of_treat` is left to its default, you can specify `other_costs_of_treat_mult`. Admissible values for this parameter are either a scalar greater zero or a tuple with values greater zero. The tuple needs as many entries as there are treatments. The imputed cost vector is then multiplied by this factor.


.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Keyword
     - Details
   * - ``var_effect_vs_0_se``
     - Standard errors of effects relative to treatment zero. Dimension is equal to the number of treatments minus 1. Default is None.
   * - ``pt_min_leaf_size``
     - Minimum leaf size. Leaves that are smaller will not be considered. A larger number reduces computation time and avoids some overfitting. Only relevant if ``gen_method`` is 'policy tree' or 'policy tree old'. Default is None.
   * - ``other_max_shares``
     - Maximum share allowed for each treatment. Note that the information must come as a tuple with as many entries as there are treatments. Default is None.
   * - ``other_costs_of_treat``
     - Treatment specific costs. Subtracted from policy scores. None (when there are no constraints): 0 None (when are constraints): Costs will be automatically determined such as to enforce constraints in the training data by finding cost values that lead to an allocation (``best_policy_score``) that fulfils restrictions ``other_max_shares``. Default is None.
   * - ``other_costs_of_treat_mult``
     - Multiplier of automatically determined cost values. Use only when automatic costs violate the constraints given by ``other_max_shares``. This allows to increase (>1) or decrease (<1) the share of treated in particular treatment. Default is None.

Please consult the :py:class:`API <mcf_functions.ModifiedCausalForest>` for more details or additional parameters. 


Example
---------

.. code-block:: python

   my_policy_tree = OptimalPolicy(
       var_d_name="d",
       var_polscore_name=["Y_LC0_un_lc_pot", "Y_LC1_un_lc_pot", "Y_LC2_un_lc_pot"],
       var_x_name_ord=["x1", "x2"],
       gen_method="policy tree", 
       # Standard errors of effects relative to treatment zero
       var_effect_vs_0_se = ('YLC1vs0_iate_se', 'YLC2vs0_iate_se', 'YLC3vs0_iate_se'), 
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
----------------------------------

Additionally, you can control certain aspects of the algorithm which impact running time:

- **Tree Depth**: You can specify the depth of the trees via the keyword arguments ``pt_depth_tree_1`` and ``pt_depth_tree_2``. 

  - ``pt_depth_tree_1`` defines the depth of the first optimal tree. The default is 3. Note that tree depth is defined such that a depth of 1 implies 2 leaves, a depth of 2 implies 4 leaves, a depth of 3 implies 8 leaves, etc.

  - ``pt_depth_tree_2`` defines the depth of the second optimal tree, which builds upon the strata obtained from the leaves of the first tree. **Note**: If ``pt_depth_tree_2`` is set to 0, the second tree is not built. The default is 1. Together with the default for ``pt_depth_tree_1``, this leads to a total tree of depth 4 (which is not optimal). Note that tree depth is defined in the same way as for ``pt_depth_tree_1``.

- **Number of Evaluation Points**: ``pt_no_of_evalupoints`` parameter specifies the number of evaluation points for continuous variables during the tree search. It determines how many of the possible splits in the feature space are considered. If the value of ``pt_no_of_evalupoints`` is smaller than the number of distinct values of a certain feature, the algorithm visits fewer splits, thus increasing computational efficiency. However, a lower value may also deviate more from the optimal splitting rule. This parameter is closely related to the approximation parameter of `Zhou, Athey, and Wager (2022) <https://doi.org/10.1287/opre.2022.2271>`_ . Lastly, note that this parameter is only relevant if ``gen_method`` is 'policy tree' or 'policy tree old'. The default value (or `None`) is 100.


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


Example
---------

.. code-block:: python

   my_policy_tree = OptimalPolicy(
       var_d_name="d",
       var_polscore_name=["Y_LC0_un_lc_pot", "Y_LC1_un_lc_pot", "Y_LC2_un_lc_pot"],
       var_x_name_ord=["x1", "x2"],
       gen_method="policy tree",
       # Depth of 1st optimal tree (Default is 3)
       pt_depth_tree_1 = 2, 
       # Depth of 2nd optimal tree (Default is 1)
       pt_depth_tree_2 = 0, 
       # Number of evaluation points for continuous variables
       pt_no_of_evalupoints = 100
       )

