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
              gen_variable_importance=GEN_VARIABLE_IMPORTANCE,
              _int_how_many_parallel=_INT_HOW_MANY_PARALLEL, _int_output_no_new_dir=_INT_OUTPUT_NO_NEW_DIR,
              _int_parallel_processing=_INT_PARALLEL_PROCESSING,
              _int_with_numba=_INT_WITH_NUMBA,
              _int_with_output=_INT_WITH_OUTPUT,
              other_costs_of_treat=OTHER_COSTS_OF_TREAT,
              other_costs_of_treat_mult=OTHER_COSTS_OF_TREAT_MULT,
              other_max_shares=OTHER_MAX_SHARES,
              pt_depth=PT_DEPTH,
              pt_enforce_restriction=PT_ENFORCE_RESTRICTION,
              pt_eva_cat_mult=PT_EVA_CAT_MULT,
              pt_min_leaf_size=PT_MIN_LEAF_SIZE,
              pt_no_of_evalupoints=PT_NO_OF_EVALUPOINTS,
              pt_select_values_cat=PT_SELECT_VALUES_CAT,
              rnd_shares=RND_SHARES,
              var_bb_restrict_name=VAR_BB_RESTRICT_NAME,
              var_d_name=VAR_D_NAME,
              var_effect_vs_0=VAR_EFFECT_VS_0,
              var_effect_vs_0_se=VAR_EFFECT_VS_0_SE,
              var_id_name=VAR_ID_NAME,
              var_polscore_desc_name=VAR_POLSCORE_DESC_NAME,
              var_polscore_name=VAR_POLSCORE_NAME,
              var_vi_to_dummy_name=VAR_VI_TO_DUMMY_NAME,
              var_vi_x_name=VAR_VI_X_NAME,
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

- <a id="var_vi_to_dummy_name"><strong>var_vi_to_dummy_name</strong></a>
  * List of strings or None, optional. Names of variables for which variable importance is computed. These variables will be broken up into dummies. Default is None.

- <a id="var_vi_x_name"><strong>var_vi_x_name</strong></a>
  * List of strings or None, optional. Names of variables for which variable importance is computed. Default is None.

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

- <a id="gen_variable_importance"><strong>gen_variable_importance</strong></a>
  Boolean. Compute variable importance statistics based on random forest classifiers. Default is False.

- <a id="_int_how_many_parallel"><strong>_int_how_many_parallel</strong></a>
	* Specifies the number of parallel processes; the default number of processes is set equal to the number of logical cores.

- <a id="_int_output_no_new_dir"><strong>_int_output_no_new_dir</strong></a> 
  * Boolean. Do not create a new directory for outputs when the path already exists. Default is False.

- <a id="_int_parallel_processing"><strong>_int_parallel_processing</strong></a>
	* If True, the program is run in parallel with the number of processes equal to **_int_how_many_parallel**. If False, the program is run on one core; the default is True.

- <a id="_int_with_numba"><strong>_int_with_numba</strong></a>
	* Specifies if Numba is deployed to speed up computation time; the default is True.

- <a id="_int_with_output"><strong>_int_with_output</strong></a>
	* If True, output is printed on file and screen. The default is True.

- <a id="other_costs_of_treat"><strong>other_costs_of_treat</strong></a>
	* Specifies costs per distributed unit of treatment. Costs will be subtracted from policy scores; 0 is no costs; the default is 0. Accordingly, the program determines individually best treatments that fulfils the restrictions in **max_shares** and imply small costs.

- <a id="other_costs_of_treat_mult"><strong>other_costs_of_treat_mult</strong></a>
	* Specifies a multiplier to costs; valid values range from 0 to 1; the default is 1. Note that parameter is only relevant if **other_costs_of_treat** is set to its default None.

- <a id="other_max_shares"><strong>other_max_shares</strong></a>
	* Specifies maximum shares of treated for each policy.

- <a id="pt_depth"><strong>pt_depth</strong></a>
	* Regulates depth of the policy tree; the default is 3; the programme accepts any number strictly greater 0.

- <a id="pt_enforce_restriction"><strong>pt_enforce_restriction</strong></a>
  * Boolean (or None). Enforces the imposed restriction (to some extent) during the computation of the policy tree. This can be very time consuming. Default is True.

- <a id="pt_eva_cat_mult"><strong>pt_eva_cat_mult</strong></a>
  * Integer (or None). Changes the number of the evaluation points (pt_no_of_evalupoints) for the unordered (categorical) variables to: pt_eva_cat_mult * pt_no_of_evalupoints (available only for the method 'policy tree eff'). Default is 1.

- <a id="pt_min_leaf_size"><strong>pt_min_leaf_size</strong></a>
	* Specifies minimum leaf size; the default is the integer part of 10% of the sample size divided by the number of leaves.

- <a id="pt_no_of_evalupoints"><strong>pt_no_of_evalupoints</strong></a>
	* Implicitly set the approximation parameter of [Zhou, Athey, and Wager (2022)](https://pubsonline.informs.org/doi/10.1287/opre.2022.2271) - $A$. Accordingly, $A = N/n_{\text{evalupoints}}$, where $N$ is the number of observations and $n_{\text{evalupoints}}$ the number of evaluation points; default value is 100.

- <a id="pt_select_values_cat"><strong>pt_select_values_cat</strong></a>
  * Approximation method for larger categorical variables. Since we search among optimal trees, for categorical variables variables we need to check for all possible combinations of the different values that lead to binary splits. This number could indeed be huge. Therefore, we compare only pt_no_of_evalupoints * 2 different combinations. Method 1 (pt_select_values_cat = True) does this by randomly drawing values from the particular categorical variable and forming groups only using those values. Method 2 (pt_select_values_cat = False) sorts the values of the categorical variables according to a values of the policy score as one would do for a standard random forest. If this set is still too large, a random sample of the entailed combinations is drawn.  Method 1 is only available for the method 'policy tree eff'. The default is False.

- <a id="rnd_shares"><strong>rnd_shares</strong></a>
	* Create a stochastic assignment. Tuple of size of number treatments. Sum of all elements must add to 1. This used only used as a comparison in the evaluation of other allocations. Default is shares of treatments in allocation under investigation.
