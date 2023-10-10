# ``OptimalPolicy`` Core Walkthrough

## Getting started


### Variables

The class ``OptimalPolicy``requires information on the column names for the policy scores and the features which shall be deployed to build the policy tree if the policy tree shall be built. The policy scores are injected via [var_polscore_name](./opt-pol_api.md#var_polscore_name), the ordered features via [var_x_ord](./opt-pol_api.md#var_x_ord) and the unordered via [var_x_unord](./opt-pol_api.md#var_x_unord). Note for the blackbox allocation the policy scores are sufficient.  

### Compulsory variable arguments

|**Variable name** |**Description**|
| -- | -- |
|[var_x_ord](./opt-pol_api.md#var_x_ord)|Specifies names of ordered variables used to build the policy tree.|
|[var_x_unord](./opt-pol_api.md#var_x_unord)|Specifies names of unordered variables used to build the policy tree.|
|[var_polscore_name](./opt-pol_api.md#var_polscore_name)|Specifies the policy score. |


 The optional variables are [var_effect_vs_0](./opt-pol_api.md#var_effect_vs_0), [var_effect_vs_0_se](./opt-pol_api.md#var_effect_vs_0_se).

## Data cleaning

The `optpoltree` function offers several useful data wrangling routines. First, by default the program **drops observations with a missing** (coded as `NaN`) and variables, which are not needed by the program. If desired, you can change these properties by the keyword argument [dc_clean_data](./opt-pol_api.md#dc_clean_data). Second, **features without variation will be dropped**. Third, if  [dc_check_perfectcorr](./opt-pol_api.md#dc_check_perfectcorr) is left to its default value True, as many features as necessary will be dropped to eliminate any perfect correlation. Finally, the keyword argument [dc_min_dummy_obs](./opt-pol_api.md#dc_min_dummy_obs) controls how many observations in each category are minimally required. If the target is not hit, the corresponding dummy variable is dropped.

### Keyword argument for data cleaning

|**Keyword** |**Details**|
| -- | -- |
|[dc_clean_data](./opt-pol_api.md#dc_clean_data)|If True, all missing and unnecessary variables are removed from the data set; the default is True.|  
|[dc_screen_covariates](./opt-pol_api.md#dc_screen_covariates)|If [dc_screen_covariates](./opt-pol_api.md#dc_screen_covariates) is True and if there are perfectly correlated variables, as many variables as necessary are excluded to remove the perfect correlation.|
|[dc_min_dummy_obs](./opt-pol_api.md#dc_min_dummy_obs)|If the program also screens covariates, i.e. when [dc_screen_covariates](./opt-pol_api.md#dc_screen_covariates) is True, the [dc_min_dummy_obs](./opt-pol_api.md#dc_min_dummy_obs) regulates the minimal number of observations in one category of a dummy variable for the dummy variable not to be removed from the data set; the default is set to 10.|

## The Quest for the Optimal Policy Tree

You may choose between two methods in determining policy allocation: a policy tree following [Zhou, Athey, and Wager (2022)](https://pubsonline.informs.org/doi/10.1287/opre.2022.2271) and a blackbox rule. Use [gen_method](./opt-pol_api.md#gen_method) to opt for the 'best_policy_score' or 'policy tree'. The blackbox rule follows the logic of allocating the treatment, which implies the best potential outcome (potentially taking estimation uncertainty into account if passed over to the program via [var_effect_vs_0_se](./opt-pol_api.md#var_effect_vs_0_se)).

In what follows, we briefly explain the solution method for finding the optimal policy tree.

**A Primer**

The ``optpoltree`` function is designed to discover the optimal policy tree in a computationally cheap and tractable manner. While the basic logic follows [Zhou, Athey, and Wager (2022)](https://pubsonline.informs.org/doi/10.1287/opre.2022.2271), the details of the programmatic implementation differ. For instance, in contrast to  [policytree](https://grf-labs.github.io/policytree/), the ``optpoltree``  allows you to consider constraints in terms of the maximal shares of treated and to detail treatment costs as well as using different policy scores.

**Algorithmic Implementation**

The ``optpoltree`` function explores the space of all viable policy trees and picks the optimal one, i.e. the one which maximizes the value function, i.e. the sum of all individual-specific policy scores, by assigning one treatment to all observations in a specific terminal node. The algorithmic idea is immediate. Given a fixed choice of previous partitions, the problem of  finding an optimal solution simplifies to solving two subproblems: find an optimal left and right subtree. Once, we have reached a terminal node, i.e. we are no longer permitted to perform splits of the covariate space, the treatment is chosen, which maximises the score of all observations in the respective leaf. This train-of-thought motivates a recursive algorithm as the overall problem naturally disaggregates into smaller and easier subproblems, which feed into the overall solution. The tree-search procedure is outlined in the subsequent pseudocode **Algorithm Tree-search Exact**.

But first things first! To begin with, we need to introduce some notation. Suppose there are $i = 1, \cdots, n$ observations, for which $p_1$ ordered and $p_2$ unordered features are observed. Further, suppose there are $M$ distinct treatments. Estimated policy scores, the potential outcomes, for the $M + 1$ distinct potential outcomes are stacked for each $i$ in a vector $\hat{\Theta}_{i}$; where $\hat{\Theta}_i(d)$ is the potential outcome for observation $i$ for treatment $d$.  Finally, let $L$ denote the depth of the tree, which equals the number of splitting nodes plus one. Then, the **Tree-Search Exact** algorithm reads as follows:

**Algorithm: Tree-search Exact**
1. If L = 1:
	2. Choose $j^* \in \{0, 1, \cdots, M\}$, which maximizes $\sum \hat{\Theta}_i(j)$ and return the corresponding reward = $\sum_{\forall i} \hat{\Theta}_{i}(j^*)$
2. Else:
	- Initialize reward = $-\infty$, and an empty tree = $\varnothing$
	  For all $m = 1, ..., p_1 + p_2$
	- Pick the m-th feature;  for ordered features return the unique values observed and sorted; if unordered return the unique categories to derive all possible splits.  
        - Then, for all possible splitting values of the m-th feature split the sample accordingly into a sample_left and sample_right
		* (reward_left, tree_left) = Tree-search(sample_left, L-1)
		* (reward_right, tree_right) = Tree-search(sample_right, L-1)
	- If reward_left + reward_right > reward:
		 - reward = reward_left + reward_right
		 - tree = Tree-search(m, splitting value, tree_left, tree_right)

The ``optpoltree``comes with options:

1. To control how many observations are required at minimum in a partition, inject a number into [pt_min_leaf_size](./opt-pol_api.md#pt_min_leaf_size).
2. If the number of individuals who receive a specific treatment is constrained, you may specify admissible treatment shares via the keyword argument [other_max_shares](./opt-pol_api.md#other_max_shares). Note that the information must come as a tuple with as many entries as there are treatments.
2. If costs of the respective treatment(s) are relevant, you may input [other_costs_of_treat](./opt-pol_api.md#other_costs_of_treat). When evaluating the reward, the aggregate costs (costs per unit times units) of the policy allocation are subtracted. If you leave the costs to their default, None, the program determines a cost vector that imply an optimal reward (policy score minus costs) for each individual, while guaranteeing that the restrictions as specified in [other_max_shares](./opt-pol_api.md#other_max_shares) are satisfied. This is of course only relevant when [other_max_shares](./opt-pol_api.md#other_max_shares) is specified.
3. If there are restrictions, and  [other_costs_of_treat](./opt-pol_api.md#other_costs_of_treat) is left to its default, the [other_costs_of_treat_mult](./opt-pol_api.md#other_costs_of_treat_mult) can be specified. Admissible values are either a scalar greater zero or a tuple with values greater zero. The tuple needs as many entries as there are treatments.  The imputed cost vector is then multiplied by this factor.  



|**Keyword** |**Details**|
| -- | -- |
|[var_effect_vs_0](./opt-pol_api.md#var_effect_vs_0)|Specifies effects relative to the default treatment zero.|
|[var_effect_vs_0_se](./opt-pol_api.md#var_effect_vs_0_se) |Specifies standard errors of the effects given in [var_effect_vs_0](./opt-pol_api.md#var_effect_vs_0). |
|[other_max_shares](./opt-pol_api.md#other_max_shares)|Specifies maximum shares of treated for each policy.|
|[other_costs_of_treat_mult](./opt-pol_api.md#other_costs_of_treat_mult)|Specifies a multiplier to costs; valid values range from 0 to 1; the default is 1. Note that parameter is only relevant if [other_costs_of_treat](./opt-pol_api.md#other_costs_of_treat) is set to its default None.|
|[other_costs_of_treat](./opt-pol_api.md#other_costs_of_treat)|Specifies costs per  unit of treatment. Costs will be subtracted from policy scores; 0 is no costs; the default is None, which implies 0 costs if there are no constraints. Accordingly, the program determines individually best treatments that fulfils the restrictions in [other_max_shares](./opt-pol_api.md#other_max_shares) and imply the smallest possible costs.|
|[pt_min_leaf_size](./opt-pol_api.md#pt_min_leaf_size)|Specifies minimum leaf size; the default is the integer part of 10% of the sample size divided by the number of leaves.|
|[pt_depth](./opt-pol_api.md#pt_depth)|Regulates depth of the policy tree; the default is 3; the programme accepts any number strictly greater 0.|
|[pt_no_of_evalupoints](./opt-pol_api.md#pt_no_of_evalupoints)|Implicitly set the approximation parameter of [Zhou, Athey, and Wager (2022)](https://pubsonline.informs.org/doi/10.1287/opre.2022.2271) - $A$. Accordingly, $A = N/n_{\text{evalupoints}}$, where $N$ is the number of observations and $n_{\text{evalupoints}}$ the number of evaluation points; default value is 100.|

 ### Speed Considerations

 You can control aspects of the algorithm, which impact running time:


1. Specify the number of evaluation points via [pt_no_of_evalupoints](./opt-pol_api.md#pt_no_of_evalupoints). This regulates when performing the tree search how many of the possible splits in the covariate space are considered. If the [pt_no_of_evalupoints](./opt-pol_api.md#pt_no_of_evalupoints)  is smaller than the number of distinct values of a certain feature, the algorithm visits fewer splits, thus increasing computational efficiency.
2. Specify the admissible depth of the tree via the keyword argument [pt_depth](./opt-pol_api.md#pt_depth).
2. Run the program in parallel. You can set the the number of processes via the keyword argument [_int_how_many_parallel](./opt-pol_api.md#_int_how_many_parallel). By default, the number is set equal to the 80 percent of the number of logical cores on your machine.
 2. A further speed up is accomplished through Numba. Numba is a Python library, which translates Python functions to optimized machine code at runtime. By default, the program uses Numba. To disable Numba, set [_int_with_numba](./opt-pol_api.md#_int_with_numba) to False.


|**Keyword** |**Details**|
| -- | -- |
| [_int_parallel_processing](./opt-pol_api.md#_int_parallel_processing) |If True, the program is run in parallel with the number of processes equal to [_int_how_many_parallel](./opt-pol_api.md#_int_how_many_parallel). If False, the program is run on one core; the default is True. |  
| [_int_how_many_parallel](./opt-pol_api.md#_int_how_many_parallel) |Specifies the number of parallel processes; the default number of processes is set equal to the logical number of cores of the machine.|
| [_int_with_numba](./opt-pol_api.md#_int_with_numba) |Specifies if Numba is deployed to speed up computation time; the default is True.|
