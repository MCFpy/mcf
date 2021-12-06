# Python API


```python
optpoltree(x_ord_name=x_ord_name, x_unord_name=x_unord_name, check_perfectcorr=check_perfectcorr, clean_data_flag=clean_data_flag, costs_of_treat=costs_of_treat, costs_of_treat=costs_of_treat, costs_of_treat_mult=costs_of_treat_mult, datpath=datpath, depth_of_tree=depth_of_tree, effect_vs_0‌=effect_vs_0‌, effect_vs_0_se=effect_vs_0_se, how_many_parallel=how_many_parallel, id_name=id_name, indata=indata, only_if_sig_better_vs_0=only_if_sig_better_vs_0, outfiletext=outfiletext, outpath=outpath, output_type=output_type, parallel_processing=parallel_processing, polscore_name=polscore_name, min_dummy_obs=min_dummy_obs, min_leaf_size=min_leaf_size, max_shares=max_shares, mp_with_ray = mp_with_ray, no_of_evalupoints=no_of_evalupoints, with_numba=with_numba, screen_covariates=screen_covariates, sig_level_vs_0=sig_level_vs_0, _smaller_sample=_smaller_sample, _with_output=_with_output)
```

## Variable Names

**x**
* <a id="x_ord_name">**x_ord_name**</a> - list with **String**
	* Specifies names of ordered variables used to build the policy tree.  
* <a id="x_unord_name">**x_unord_name**</a> - list with **String**
	* Specifies names of unordered variables used to build the policy tree.

## Parameters

**c**
* <a id="check_perfectcorr">**check_perfectcorr**</a> - **Boolean**
	* If **screen_covariates** is True and if there are perfectly correlated variables, as many variables as necessary are excluded to remove the perfect correlation.
* <a id="clean_data_flag">**clean_data_flag**</a> - **Boolean**
	* If True, observations with any missing as well as variables that are not needed by the program are removed from the data set; the default is True.
* <a id="costs_of_treat">**costs_of_treat**</a> - **None** or **Floats**
	* Specifies costs per distributed unit of treatment. Costs will be subtracted from policy scores; 0 is no costs; the default is 0. Accordingly, the program determines individually best treatments that fulfils the restrictions in **max_shares** and imply small costs.
* <a id="costs_of_treat_mult">**costs_of_treat_mult**</a> - **Float** between **0, 1**
	* Specifies a multiplier to costs; valid values range from 0 to 1; the default is 1. Note that parameter is only relevant if **costs_of_treat** is set to its default None.

**d**
* <a id="datpath">**datpath**</a> - **String**
	* Specifies the directory, where the data for estimation is located.
* <a id="depth_of_tree">**depth_of_tree**</a> - positive **Integer**
	* Regulates depth of the policy tree; the default is 3; the programme accepts any number strictly greater 0.

**e**
* <a id="effect_vs_0‌">**effect_vs_0‌**</a>
	* Specifies effects relative to the default treatment zero.
* <a id="effect_vs_0_se">**effect_vs_0_se**</a>
	* Specifies standard errors of the effects given in **effect_vs_0**.

**h**
* <a id="how_many_parallel">**how_many_parallel**</a> - positive **Integer**
	* Specifies the number of parallel processes; the default number of processes is set equal to the number of logical cores.

**i**
* <a id="id_name">**id_name**</a> - **String**
	* Specifies an identifier; if there is no identifier, an identifier will be added to the data.
* <a id="indata">**indata**</a> - **String**
	* Specifies the file name for the data, which is used for estimation; the file needs to be in *csv* format.

**o**
* <a id="only_if_sig_better_vs_0">**only_if_sig_better_vs_0**</a> - **Boolean**
	* If True, the assignment is based on policy scores, which are  significantly better than the first score in **polscore_name**; the default is False.
* <a id="outfiletext">**outfiletext**</a> - **String**
	* Specifies the name for the file, in which the text output from the program is written; if the name is not specified the default is set to the name of the data, deployed for estimation - **indata** - with the extension *txt* is used.
* <a id="outpath">**outpath**</a> - **String**
	* Creates a folder *out* in the application directory, in which the output files are written.
* <a id="output_type">**output_type**</a> - **Integer** taking value **0**, **1**, **2** or **None**
	* Determines where the output goes; for a value of 0 the output goes to the terminal; for a value of 1, the output is sent to a file; for a value of 2, the output is sent to both terminal and file; the default is None.

**p**
* <a id="parallel_processing">**parallel_processing**</a> - **Boolean**
	* If True, the program is run in parallel with the number of processes equal to **how_many_parallel**. If False, the program is run on one core; the default is True.
* <a id="polscore_name">**polscore_name**</a> - list of **Strings**
	* Specifies the policy score, the potential outcomes.

**m**
* <a id="max_shares">**max_shares**</a> - **tuple** with **Floats** between **0, 1**
	* Specifies maximum shares of treated for each policy.
* <a id="min_dummy_obs">**min_dummy_obs**</a> - positive **Integer**
	* If the program also screens covariates, i.e. when **screen_covariates** is True, the **min_dummy_obs** regulates the minimal number of observations in one category of a dummy variable for the dummy variable not to be removed from the data set; the default is set to 10.
* <a id="min_leaf_size">**min_leaf_size**</a> - positive **Integer**
	* Specifies minimum leaf size; the default is the integer part of 10% of the sample size divided by the number of leaves.
* <a id="mp_with_ray">**mp_with_ray**</a> - **Boolean**
	* If True, Ray is used for multiprocessing. If False, concurrent futures is used; the default ist True.


**n**
* <a id="no_of_evalupoints">**no_of_evalupoints**</a> - positive **Integer**
	* Implicitly set the approximation parameter of [Zhou, Athey, and Wager (2018)](https://arxiv.org/abs/1810.04778) - $A$. Accordingly, $A = N/n_{\text{evalupoints}}$, where $N$ is the number of observations and $n_{\text{evalupoints}}$ the number of evaluation points; default value is 100.

**w**
* <a id="with_numba">**with_numba**</a> - **Boolean**
	* Specifies if Numba is deployed to speed up computation time; the default is True.

**s**
* <a id="screen_covariates">**screen_covariates**</a> - **Boolean**
	* If True, the program screens the covariates.
* <a id="sig_level_vs_0">**sig_level_vs_0**</a> **Float** between **0, 1**
	* Specifies relevant significance level for statistical testing; the default is 0.05.

**_**
* <a id="_smaller_sample"> **_smaller_sample**</a> - **Float** between **0, 1**
	* Specifies share of original data used to test the program.
* <a id="_with_output"> **_with_output**</a> - **Boolean**
	* If True, output is displayed; the default is True.
