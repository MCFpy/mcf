# Python API


```python
OptimalPolicy(dc_check_perfectcorr=DC_CHECK_PERFECTCORR
              dc_clean_data=DC_CLEAN_DATA,
              dc_min_dummy_obs=DC_MIN_DUMMY_OBS,
              dc_screen_covariates=DC_SCREEN_COVARIATES,
	      gen_method=GEN_METHOD,
              gen_outfiletext=GEN_OUTFILETEXT,
              gen_outpath=GEN_OUTPATH,
              gen_output_type=GEN_OUTPUT_TYPE,
              int_how_many_parallel=INT_HOW_MANY_PARALLEL,
              int_parallel_processing=INT_PARALLEL_PROCESSING,
              int_with_numba=INT_WITH_NUMBA,
              int_with_output=INT_WITH_OUTPUT,
              other_costs_of_treat=OTHER_COSTS_OF_TREAT,
              other_costs_of_treat_mult=OTHER_COSTS_OF_TREAT_MULT,
              other_max_shares=OTHER_MAX_SHARES,
              pt_depth=PT_DEPTH,
              pt_min_leaf_size=PT_MIN_LEAF_SIZE,
              pt_no_of_evalupoints=PT_NO_OF_EVALUPOINTS,
              rnd_shares=RND_SHARES,
              var_bb_restrict_name=VAR_BB_RESTRICT_NAME,
              var_d_name=VAR_D_NAME,
              var_effect_vs_0=VAR_EFFECT_VS_0,
              var_effect_vs_0_se=VAR_EFFECT_VS_0_SE,
              var_id_name=VAR_ID_NAME,
              var_polscore_desc_name=VAR_POLSCORE_DESC_NAME,
              var_polscore_name=VAR_POLSCORE_NAME,
              var_x_ord_name=VAR_X_ORD_NAME,
              var_x_unord_name=VAR_X_UNORD_NAME)
```

## Variable Names

- <a id="var_bb_restrict_name"><strong>var_bb_restrict_name</strong></a>
	* Specifies variables related to a restriction. If there is a capacity constraint, priority will be given to observations with
the highest value. Default is None.

- <a id="var_d_name"><strong>var_d_name</strong></a>
	* Specifies name of discrete treatment. Only required if changes are analyzed.

- <a id="var_effect_vs_0"><strong>var_effect_vs_0</strong></a>
	* Specifies effects relative to the default treatment zero.

- <a id="var_effect_vs_0_se"><strong>var_effect_vs_0_se</strong></a>
	* Specifies standard errors of the effects given in **effect_vs_0**.

- <a id="var_id_name"><strong>var_id_name</strong></a>
	* Specifies an identifier; if there is no identifier, an identifier will be added to the data.

- <a id="var_polscore_desc_name"><strong>var_polscore_desc_name</strong></a>
	* Tuple of tuples. Each tuple contains treatment specific variables that are used to evaluate the effect of the allocation with respect to those variables. Default is None.

- <a id="var_polscore_name"><strong>var_polscore_name</strong></a>
	* Specifies the policy score, the potential outcomes.

- <a id="var_x_ord_name"><strong>var_x_ord_name</strong></a>
	* Specifies names of ordered variables used to build the policy tree.

- <a id="var_x_unord_name"><strong>var_x_unord_name</strong></a>
	* Specifies names of unordered variables used to build the policy tree.



## Parameters

- <a id="dc_clean_data"><strong>dc_clean_data</strong></a>
	* If True, observations with any missing as well as variables that are not needed by the program are removed from the data set; the default is True.

- <a id="dc_min_dummy_obs"><strong>dc_min_dummy_obs</strong></a>
	* If the program also screens covariates, i.e. when **screen_covariates** is True, the **min_dummy_obs** regulates the minimal number of observations in one category of a dummy variable for the dummy variable not to be removed from the data set; the default is set to 10.

- <a id="dc_screen_covariates"><strong>dc_screen_covariates</strong></a>
	* If True, the program screens the covariates.

- <a id="gen_method"><strong>gen_method</strong></a>
	* Tuple used to specify the solver algorithms. Choose between 'best_policy_score' and 'policy tree'.

- <a id="gen_outfiletext"><strong>gen_outfiletext</strong></a>
	* Specifies the name for the file, in which the text output from the program is written; if the name is not specified the default is set to the name of the data with the extension *txt* is used.

- <a id="gen_outpath"><strong>gen_outpath</strong></a>
	* Creates a folder *out* in the application directory, in which the output files are written.

- <a id="gen_output_type"><strong>gen_output_type</strong></a>
	* Determines where the output goes; for a value of 0 the output goes to the terminal; for a value of 1, the output is sent to a file; for a value of 2, the output is sent to both terminal and file; the default is None.

- <a id="int_how_many_parallel"><strong>int_how_many_parallel</strong></a>
	* Specifies the number of parallel processes; the default number of processes is set equal to the number of logical cores.

- <a id="int_parallel_processing"><strong>int_parallel_processing</strong></a>
	* If True, the program is run in parallel with the number of processes equal to **int_how_many_parallel**. If False, the program is run on one core; the default is True.

- <a id="int_with_numba"><strong>int_with_numba</strong></a>
	* Specifies if Numba is deployed to speed up computation time; the default is True.

- <a id="int_with_output"><strong>int_with_output</strong></a>
	* If True, output is printed on file and screen. The default is True.

- <a id="other_costs_of_treat"><strong>other_costs_of_treat</strong></a>
	* Specifies costs per distributed unit of treatment. Costs will be subtracted from policy scores; 0 is no costs; the default is 0. Accordingly, the program determines individually best treatments that fulfils the restrictions in **max_shares** and imply small costs.

- <a id="other_costs_of_treat_mult"><strong>other_costs_of_treat_mult</strong></a>
	* Specifies a multiplier to costs; valid values range from 0 to 1; the default is 1. Note that parameter is only relevant if **other_costs_of_treat** is set to its default None.

- <a id="other_max_shares"><strong>other_max_shares</strong></a>
	* Specifies maximum shares of treated for each policy.

- <a id="pt_depth"><strong>pt_depth</strong></a>
	* Regulates depth of the policy tree; the default is 3; the programme accepts any number strictly greater 0.

- <a id="pt_min_leaf_size"><strong>pt_min_leaf_size</strong></a>
	* Specifies minimum leaf size; the default is the integer part of 10% of the sample size divided by the number of leaves.

- <a id="pt_no_of_evalupoints"><strong>pt_no_of_evalupoints</strong></a>
	* Implicitly set the approximation parameter of [Zhou, Athey, and Wager (2012)](https://pubsonline.informs.org/doi/10.1287/opre.2022.2271) - $A$. Accordingly, $A = N/n_{\text{evalupoints}}$, where $N$ is the number of observations and $n_{\text{evalupoints}}$ the number of evaluation points; default value is 100.

- <a id="rnd_shares"><strong>rnd_shares</strong></a>
	* Create a stochastic assignment of the data passed. Tuple of size of number treatments. Sum of all elements must add to 1. This used only used as a comparison in the evaluation of other allocations. Default is shares of treatments in allocation under investigation.
