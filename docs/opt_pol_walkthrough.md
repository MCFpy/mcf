# ``optpoltree`` Core Walkthrough

## Getting started

### Structure of directory

Analogous to the ``ModifiedCausalForest`` function,
you can specify the working directory via [outpath](./opt_pol_1.md#outpath) and the data directory via [datapath](./opt_pol_1.md#datapath). Please note that the data file, which you define via [indata](./opt_pol_1.md#indata), must be a *csv*.

The program creates the folder *out*, where results are stored. If a folder *out* already exists, folder *out1* is created, if *out1* exists *out2* and so on. The folder goes to your default working directory or to the directory you have passed over via the keyword argument [outpath](./opt_pol_1.md#outpath).  By default, the output is printed to the console and saved in a file. To change this behaviour, set [_with_output](./opt_pol_1.md#_with_output) to False. This can be useful for Monte Carlo studies.

|**Keyword** |**Details**|
| -- | -- |
|[output_type](./opt_pol_1.md#output_type)|Determines where the output goes; for a value of 0 the output goes to the terminal; for a value of 1, the output is sent to a file; for a value of 2, the output is sent to both terminal and file; the default is 2.|  
|[_with_output](./opt_pol_1.md#_with_output)|If True, print statements are executed; the default is True.|
|[outpath](./opt_pol_1.md#outpath)|Creates a folder in the application directory, in which the output files are written.|
|[datapath](./opt_pol_1.md#datapath)|Specifies the directory, in which the data is saved for estimation and/or prediction.|
|[indata](./opt_pol_1.md#indata)|Specifies the file name for the data, which is used for estimation; the file needs to be in csv format and contain the policy scores and the covariates.|

### Variables

The ``optpoltree``requires at least three distinct lists, containing the column names for the policy scores and the features which shall be deployed to build the policy tree. The policy scores are injected via [polscore_name](./opt_pol_1.md#polscore_name), the ordered features via [x_ord](./opt_pol_1.md#x_ord) and the unordered via [x_unord](./opt_pol_1.md#x_unord).  

### Compulsory variable arguments

|**Variable name** |**Description**|
| -- | -- |
|[x_ord_name](./opt_pol_1.md#x_ord_name)|Specifies names of ordered variables used to build the policy tree.|
|[x_unord_name](./opt_pol_1.md#x_unord_name)|Specifies names of unordered variables used to build the policy tree.|
|[polscore_name](./opt_pol_1.md#polscore_name)|Specifies the policy score. |


 The optional variables are [effect_vs_0‌](./opt_pol_1.md#effect_vs_0‌), [effect_vs_0‌_se](./opt_pol_1.md#effect_vs_0‌_se
), and [only_if_sig_better_vs_0](./opt_pol_1.md#only_if_sig_better_vs_0) and are detailed [here](###Tree-Search):

## Data cleaning

The `optpoltree` function offers several useful data wrangling routines. First, by default the program **drops observations with a missing** (coded as `NaN`) and variables, which are not needed by the program. If desired, you can change these properties by the keyword argument [clean_data_flag](./opt_pol_1.md#clean_data_flag). Second, **features without variation will be dropped**. Set [screen_covariates](./opt_pol_1.md#screen_covariates) to False, to change this behaviour. Third, if  [check_perfectcorr](./opt_pol_1.md#check_perfectcorr) is left to its default value True, as many features as necessary will be dropped to eliminate any perfect correlation. Finally, the keyword argument [min_dummy_obs](./opt_pol_1.md#min_dummy_obs) controls how many observations in each category are minimally required. If the target is not hit, the corresponding dummy variable is dropped.

### Keyword argument for data cleaning

|**Keyword** |**Details**|
| -- | -- |
|[clean_data_flag](./opt_pol_1.md#clean_data_flag)|If True, all missing and unnecessary variables are removed from the data set; the default is True.|  
|[screen_covariates](./opt_pol_1.md#screen_covariates)|Determines whether the covariates are screened; the default is True; to omit screening stage specify False.|
|[check_perfectcorr](./opt_pol_1.md#check_perfectcorr)|If [screen_covariates](./opt_pol_1.md#screen_covariates) is True and if there are perfectly correlated variables, as many variables as necessary are excluded to remove the perfect correlation.|
|[min_dummy_obs](./opt_pol_1.md#min_dummy_obs)|If the program also screens covariates, i.e. when [screen_covariates](./opt_pol_1.md#screen_covariates) is True, the [min_dummy_obs](./opt_pol_1.md#min_dummy_obs) regulates the minimal number of observations in one category of a dummy variable for the dummy variable not to be removed from the data set; the default is set to 10.|

## The Quest for the Optimal Policy Tree

**A Primer**

The ``optpoltree`` function is designed to discover the optimal policy tree in a computationally cheap and tractable manner. While the basic logic follows [Zhou, Athey, and Wager (2018)](https://arxiv.org/abs/1810.04778), the details of the programmatic implementation differ. For instance, in contrast to  [policytree](https://grf-labs.github.io/policytree/), the ``optpoltree``  allows you to consider constraints in terms of the maximal shares of treated and to detail treatment costs as well as using different policy scores.

**Algorithmic Implementation**

If [ft_yes](./opt_pol_1.md#ft_yes) is set to True, the algorithm proceeds recursively rather than sequentially, which is the case if [st_yes](./opt_pol_1.md#st_yes). The ``optpoltree`` function explores the space of all viable policy trees and picks the optimal one, i.e. the one which maximizes the value function, i.e. the sum of all individual-specific policy scores, by assigning one treatment to all observations in a specific terminal node. The algorithmic idea is immediate. Given a fixed choice of previous partitions, the problem of  finding an optimal solution simplifies to solving two subproblems: find an optimal left and right subtree. Once, we have reached a terminal node, i.e. we are no longer permitted to perform splits of the covariate space, the treatment is chosen, which maximises the score of all observations in the respective leaf. This train-of-thought motivates a recursive algorithm as the overall problem naturally disaggregates into smaller and easier subproblems, which feed into the overall solution. The tree-search procedure is outlined in the subsequent pseudocode **Algorithm Tree-search Exact**.

But first things first! To begin with, we need to introduce some notation. Suppose there are $i = 1, \cdots, n$ observations, for which $p_1$ ordered and $p_2$ unordered features are observed. Further, suppose there are $M$ distinct treatments. Estimated policy scores, the potential outcomes, for the $M + 1$ distinct potential outcomes are stacked for each $i$ in a vector $\hat{\Theta}_{i}$; where $\hat{\Theta}_i(d)$ is the potential outcome for observation $i$ for treatment $d$.  Finally, let $L$ denote the depth of the tree, which equals the number of splitting nodes plus one. Then, the **Tree-Search Exact** algorithm reads as follows:

**Algorithm: Tree-search Exact**
1. If L = 1:
	2. Choose $j^* \in \{0, 1, \cdots, M\}$, which maximizes $\sum \hat{\Theta}_i(j)$ and return the corresponding reward = $\sum_{\forall i} \hat{\Theta}_{i}(j^*)$
2. Else:
	3. Initialize reward = $-\infty$, and an empty tree = $\varnothing$
	 2. For all $m = 1, ..., p_1 + p_2$
		 3. Pick the m-th feature;  for ordered features return the unique values observed and sorted; if unordered return the unique categories to derive all possible splits.  
			 4. Then, for all possible splitting values of the m-th feature split the sample accordingly into a sample_left and sample_right
				 5. (reward_left, tree_left) = Tree-search(sample_left, L-1)
				 6. (reward_right, tree_right) = Tree-search(sample_right, L-1)
				 7. If reward_left + reward_right > reward:
					 8. reward = reward_left + reward_right
					 9. tree = Tree-search(m, splitting value, tree_left, tree_right)

The ``optpoltree``comes with options:

1. To control how many observations are required at minimum in a partition, inject a number into [ft_min_leaf_size](./opt_pol_1.md#ft_min_leaf_size).
2. If the number of individuals who receive a specific treatment is constrained, you may specify admissible treatment shares via the keyword argument [max_shares](./opt_pol_1.md#max_shares). Note that the information must come as a tuple with as many entries as there are treatments.
2. If costs of the respective treatment(s) are relevant, you may input [costs_of_treat](./opt_pol_1.md#costs_of_treat). When evaluating the reward, the aggregate costs (costs per unit times units) of the policy allocation are subtracted. If you leave the costs to their default, None, the program determines a cost vector that imply an optimal reward (policy score minus costs) for each individual, while guaranteeing that the restrictions as specified in [max_shares](./opt_pol_1.md#max_shares) are satisfied.
3. If  [costs_of_treat](./opt_pol_1.md#costs_of_treat) is left to its default, the [costs_mult](./opt_pol_1.md#costs_mult) can be specified. Admissible values are either a scalar greater zero or a tuple with values greater zero. The tuple needs as many entries as there are treatments.  The imputed cost vector is then multiplied by this factor.  
3. If you set [only_if_sig_better](./opt_pol_1.md#only_if_sig_better) to True, the  ``optpoltree`` programme examines whether any given treatment significantly outperforms the null treatment; in practice, this is checked by a one-sided test of statistical significance. Note that due to the nature of our optimization problem, we only take interest in positive deviations from the null treatment. Hence, the one-sided test. If the null that the difference of the two treatments is greater zero, cannot be rejected, the program recodes the policy score for the corresponding individual and treatment to the score implied by the null treatment minus an arbitrarily small float (1e-8).  To execute those tests, you have to specify a list of strings, which correspond to the column names where the [effect_vs_0](./opt_pol_1.md#effect_vs_0) and the corresponding standard errors are stored [effect_vs_0_se](./opt_pol_1.md#effect_vs_0_se). The relevant significance level for the statistical testing can be passed over via  [sig_level_vs_0](./opt_pol_1.md#sig_level_vs_0).


|**Keyword** |**Details**|
| -- | -- |
|[only_if_sig_better](./opt_pol_1.md#only_if_sig_better)|If True, the assignment is based on policy scores, which are  significantly better than the first score in [polscore_name](./opt_pol_1.md#polscore_name); the default is False.|
|[effect_vs_0](./opt_pol_1.md#effect_vs_0)|Specifies effects relative to the default treatment zero.|
|[effect_vs_0_se](./opt_pol_1.md#effect_vs_0_se) |Specifies standard errors of the effects given in [effect_vs_0](./opt_pol_1.md#effect_vs_0). |
|[sig_level_vs_0](./opt_pol_1.md#sig_level_vs_0)|Specifies relevant significance level for statistical testing; the default is 0.05.|
|[max_shares](./opt_pol_1.md#max_shares)|Specifies maximum shares of treated for each policy.|
|[costs_mult](./opt_pol_1.md#costs_mult)|Specifies a multiplier to costs; valid values range from 0 to 1; the default is 1. Note that parameter is only relevant if [costs_of_treat](./opt_pol_1.md#costs_of_treat) is set to its default None.|
|[costs_of_treat](./opt_pol_1.md#costs_of_treat)|Specifies costs per distributed unit of treatment. Costs will be subtracted from policy scores; 0 is no costs; the default is None. Accordingly, the program determines individually best treatments that fulfils the restrictions in [max_shares](./opt_pol_1.md#max_shares) and imply the smallest possible costs.|
|[ft_min_leaf_size](./opt_pol_1.md#ft_min_leaf_size)|Specifies minimum leaf size; the default is the integer part of 10% of the sample size divided by the number of leaves.|
|[ft_depth](./opt_pol_1.md#ft_depth)|Regulates depth of the policy tree; the default is 3; the programme accepts any number strictly greater 0.|
|[ft_no_of_evalupoints](./opt_pol_1.md#ft_no_of_evalupoints)|Implicitly set the approximation parameter of [Zhou, Athey, and Wager (2018)](https://arxiv.org/abs/1810.04778) - $A$. Accordingly, $A = N/n_{\text{evalupoints}}$, where $N$ is the number of observations and $n_{\text{evalupoints}}$ the number of evaluation points; default value is 100.|

 ### Speed Considerations

 You can control aspects of the algorithm, which impact running time:


1. Specify the number of evaluation points via [no_of_evalupoints](./opt_pol_1.md#no_of_evalupoints). This regulates when performing the tree search how many of the possible splits in the covariate space are considered. If the [no_of_evalupoints](./opt_pol_1.md#no_of_evalupoints)  is smaller than the number of distinct values of a certain feature, the algorithm visits fewer splits, thus increasing computational efficiency.
2. Specify the admissible depth of the tree via the keyword argument [ft_depth](./opt_pol_1.md#ft_depth).
2. Run the program in parallel. You can set the the number of processes via the keyword argument [how_many_parallel](./opt_pol_1.md#how_many_parallel). By default, the number is set equal to the number of logical cores on your machine.
 2. A further speed up is accomplished through Numba. Numba is a Python library, which translates Python functions to optimized machine code at runtime. By default, the program uses Numba. To disable Numba, set [with_numba](./opt_pol_1.md#with_numba) to False.
 3. Finally, run the program not on the full data, which naturally speeds up things. Set [_smaller_sample](./opt_pol_1.md#_smaller_sample) a strict positive float smaller one to run the program on a smaller sample.

|**Keyword** |**Details**|
| -- | -- |
| [parallel_processing](./opt_pol_1.md#parallel_processing) |If True, the program is run in parallel with the number of processes equal to [how_many_parallel](./opt_pol_1.md#how_many_parallel). If False, the program is run on one core; the default is True. |  
| [how_many_parallel](./opt_pol_1.md#how_many_parallel) |Specifies the number of parallel processes; the default number of processes is set equal to the logical number of cores of the machine.|
| [with_numba](./opt_pol_1.md#with_numba) |Specifies if Numba is deployed to speed up computation time; the default is True.|
| [_smaller_sample](./opt_pol_1.md#_smaller_sample)|Specifies share of original data used to test the program.|
