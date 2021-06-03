# Python API


```python
mcf.ModifiedCausalForest(cluster_name=cluster_name, d_name=d_name, id_name=id_name,
w_name=w_name, x_balance_name_ord=x_balance_name_ord,
x_balance_name_unord=x_balance_name_unord,
x_name_always_in_ord=x_name_always_in_ord,
x_name_always_in_unord=x_name_always_in_unord, x_name_ord=x_name_ord,
x_name_remain_ord=x_name_remain_ord, x_name_remain_unord=x_name_remain_unord,
x_name_unord=x_name_unord, y_name=y_name, y_tree_name=y_tree_name,
z_name_amgate=z_name_amgate, z_name_mgate=z_name_mgate,
z_name_list=z_name_list, z_name_split_ord=z_name_split_ord,
z_name_split_unord=z_name_split_unord, alpha_reg_grid=alpha_reg_grid,
alpha_reg_max=alpha_reg_max, alpha_reg_min=alpha_reg_min,
atet_flag=atet_flag, balancing_test=balancing_test,
bin_corr_threshold=bin_corr_threshold, bin_corr_yes=bin_corr_yes,
boot=boot, check_perfectcorr=check_perfectcorr,
choice_based_sampling=choice_based_sampling,
choice_based_weights=choice_based_weights, ci_level=ci_level,
clean_data_flag=clean_data_flag,
cluster_std=cluster_std, datpfad=datpfad, descriptive_stats=descriptive_stats,
dpi=dpi,  fontsize=fontsize, forest_files=forest_files, fs_other_sample=fs_other_sample,
fs_other_sample_share=fs_other_sample_share, fs_rf_threshold=fs_rf_threshold,
fs_yes=fs_yes, gatet_flag=gatet_flag,
gmate_no_evaluation_points=gmate_no_evaluation_points,
gmate_sample_share=gmate_sample_share, indata=indata,
knn_const=knn_const, knn_flag=knn_flag, knn_min_k=knn_min_k,
l_centering=l_centering, l_centering_cv_k=l_centering_cv_k,
l_centering_new_sample=l_centering_new_sample,
l_centering_share=l_centering_share,
match_nn_prog_score=match_nn_prog_score, max_cats_z_vars=max_cats_z_vars,
max_weight_share=max_weight_share, mce_vart=mce_vart,
min_dummy_obs=min_dummy_obs, mp_parallel=mp_parallel,
mp_ray_objstore_multiplier=mp_ray_objstore_multiplier,
mp_vim_type=mp_vim_type, mp_weights_tree_batch=mp_weights_tree_batch,
mp_weights_type=mp_weights_type, mp_with_ray=mp_with_ray,
m_grid=m_grid, m_max_share=m_max_share, m_min_share=m_min_share,
m_random_poisson=m_random_poisson, nn_main_diag_only=nn_main_diag_only,
no_filled_plot=no_filled_plot, nw_bandw=nw_bandw, nw_kern_flag=nw_kern_flag,
n_min_grid=n_min_grid, n_min_max=n_min_max, n_min_min=n_min_min,
outfiletext=outfiletext, outpfad=outpfad, output_type=output_type,
panel_data=panel_data, panel_in_rf=panel_in_rf, post_est_stats=post_est_stats,
post_kmeans_max_tries=post_kmeans_max_tries,
post_kmeans_no_of_groups=post_kmeans_no_of_groups,
post_kmeans_replications=post_kmeans_replications,
post_kmeans_yes=post_kmeans_yes, post_plots=post_plots,
post_random_forest_vi=post_random_forest_vi, preddata=preddata, predict_mcf=predict_mcf
p_diff_penalty=p_diff_penalty, random_thresholds=random_thresholds,
relative_to_first_group_only=relative_to_first_group_only,
save_forest=save_forest, screen_covariates=screen_covariates,
share_forest_sample=share_forest_sample, show_plots=show_plots,
smooth_gates=smooth_gates, smooth_gates_bandwidth=smooth_gates_bandwidth,
smooth_gates_no_evaluation_points=smooth_gates_no_evaluation_points,
stop_empty=stop_empty, subsample_factor=subsample_factor,
support_check=support_check, support_min_p=support_min_p,
support_quantil=support_quantil, weighted=weighted,
weight_as_sparse=weight_as_sparse, train_mcf=train_mcf,  
variable_importance_oob=variable_importance_oob, verbose=verbose,
_max_cats_cont_vars=_max_cats_cont_vars, _max_save_values=_max_save_values,
_seed_sample_split=_seed_sample_split, _smaller_sample=_smaller_sample,
_with_output=_with_output)
```


## Variable Names

**c**

* <a id="cluster_name">**cluster_name**</a> - list with **string**
	* Specifies variable defining the clusters.

**d**

* <a id="d_name">**d_name**</a> - list with **string**
	* Specifies name of treatment, which must be discrete.

**i**

* <a id="id_name">**id_name**</a> - list with **string**
	* Specifies an identifier; if there is no identifier, an identifier will be added to the data.

**w**

* <a id="w_name">**w_name**</a> - list with **string**
  	* Specifies the name of the weight, if the weighting option is used.

**x**

* <a id="x_balance_name_ord">**x_balance_name_ord**</a> - list with **string**
	* Specifies names of ordered variables for balancing tests (also excluded from feature selection).
* <a id="x_balance_name_unord">**x_balance_name_unord**</a> - list with **string**
	* Specifies names of unordered variables for balancing tests (also excluded from feature selection).
* <a id="x_name_always_in_ord">**x_name_always_in_ord**</a> - list with **string**
	* Specifies names of ordered features, which are always included when deciding upon the next split.
* <a id="x_name_always_in_unord">**x_name_always_in_unord**</a> - list with **string**
	* Specifies names of unordered variables, which are always included when deciding upon next split.
* <a id="x_name_ord">**x_name_ord**</a> - list with **string**
	* Specifies names of ordered variables.
* <a id="x_name_remain_ord">**x_name_remain_ord**</a> - list with **string**
	* Specifies names of ordered variables to be excluded from preliminary feature selection.
* <a id="x_name_remain_unord">**x_name_remain_unord**</a> - list with **string**
	* Specifies names of unordered variables to be excluded from preliminary feature selection.
* <a id="x_name_unord">**x_name_unord**</a> - list with **string**
	* Specifies names of unordered variables.

**y**

* <a id="y_name">**y_name**</a> - list with **string**
	* Specifies outcome variables.
* <a id="y_tree_name">**y_tree_name**</a> - list with **string**
	*  Specifies variable to build trees; if None or [], the first variable in **y_name** is used to build trees; it will be added to the list of outcome variables.


**z**

* <a id="z_name_amgate">**z_name_amgate**</a> - list with **string**
	* Specifies names of variables for which average marginal GATE will be computed; variable must be in included in **x_name_ord** or **x_name_unord**; otherwise, variables will be deleted from list.
* <a id="z_name_mgate">**z_name_mgate**</a> - list with **string**
	* Specifies names of variables for which marginal GATE (at median) will be computed; variable must be in included in **x_name_ord** or **x_name_unord**; otherwise, variables will be deleted from list.
* <a id="z_name_list">**z_name_list**</a> - list with **string**
	* Specifies names of ordered variables with many values; these variables are recoded to define the split, they will be added to the list of confounders. Since they are broken up in categories for all their observed values, it does not matter whether they are coded as ordered or unordered.
* <a id="z_name_split_ord">**z_name_split_ord**</a> - list with **string**
	* Specifies names of ordered variables that are discrete and define a unique sample split for each value.
* <a id="z_name_split_unord">**z_name_split_unord**</a> - list with **string**
	* Specifies names of unordered variables that are discrete and define a unique sample split for each value.

## Parameters

**a**

* <a id="alpha_reg_grid">**alpha_reg_grid**</a> - positive **integer**
	* Sets the number of grid values; the default is 2.
* <a id="alpha_reg_max">**alpha_reg_max**</a> - **float** between **0, 0.5**
	* Determines the maximal $\alpha$ for $0 < \alpha < 0.5$; the default is 0.2.
* <a id="alpha_reg_min">**alpha_reg_min**</a> - **float** between **0, 0.4**
	* Determines smallest $\alpha$ for $0 < \alpha < 0.4$; the default is 0.1.
* <a id="atet_flag">**atet_flag**</a> - **Boolean**
	* If  True, average effects for subpopulations are computed by treatments (if available); this works only if at least one $z$ variable is specified; the default is False.

**b**
* <a id="balancing_test">**balancing_test**</a> - **Boolean**
	* If  True, the ATE-based balancing test predicates on weights; requires **weight_based_inference**; the default is True.
* <a id="bin_corr_threshold">**bin_corr_threshold**</a> - **float** between **0, 1**
	* Determines the minimum threshold of absolute correlation to be displayed; the default is 0.1.
* <a id="bin_corr_yes">**bin_corr_yes**</a> - **Boolean**
	* If True, the program checks the binary predictions.
* <a id="boot">**boot**</a> - positive **integer**
	* Gives the number of trees in the forest to be estimated; the default is 1000.

**c**
* <a id="check_perfectcorr">**check_perfectcorr**</a> - **Boolean**
	* If **screen_covariates** is True and if there are perfectly correlated variables, as many variables as necessary are excluded to remove the perfect correlation.
* <a id="choice_based_sampling">**choice_based_sampling**</a> - **Boolean**
	* If True, the program uses choice based sampling; the default value is False.
* <a id="choice_based_weights">**choice_based_weights**</a> - list of **floats**, each between **0, 1**
	* Includes treatment specific sampling probabilities, which are used for (g)ates only and relate to 'pred_eff_data'.
* <a id="ci_level">**ci_level**</a> - **float** between **0-1**
	* Sets the confidence level used for the figures; must be between 0 and 1; the default is 0.9.
* <a id="clean_data_flag">**clean_data_flag**</a> - **Boolean**
	* If True, all missing and unnecessary variables are removed from the data set; the default is True.
* <a id="cluster_std">**cluster_std**</a> - **Boolean**
	* If True, program computes clustered standard errors; the value  will be automatically set to 1 if panel data option is activated.
* <a id="cond_var_flag">**cond_var_flag**</a> - **Boolean**
	* If True, variance estimation is based on a variance decomposition of weighted conditional means $\hat{\mathrm{w}}_{\mathrm{i}} \mu_{\mathrm{Y} \mid \hat{\mathrm{W}}}\left(\hat{\mathrm{w}}_{\mathrm{i}}\right)$ and variances $\hat{\mathrm{w}}_{\mathrm{i}} \sigma_{\mathrm{Y} \mid \hat{\mathrm{W}}}^{2}\left(\hat{\mathrm{w}}_{\mathrm{i}}\right)$. If False, variance estimation builds on the sum of variances of weighted outcomes; the default is True.


**d**
* <a id="datpfad">**datpfad**</a>  - **string**
	* Specifies the directory, in which the data is saved for estimation and/or prediction.
* <a id="descriptive_stats">**descriptive_stats**</a> - **Boolean**
	* If True, the descriptive statistics are printed to the input and output files; the default is True.
* <a id="dpi">**dpi**</a> - **integer** larger than **0**
	* Sets the resolution, i.e. dots per inch (dpi); a value larger than 0 must be specified; the default is 500.

**f**
* <a id="fontsize">**fontsize**</a> - **integer** from **0-7**
	* Sets the font size for the legend; ranges from 1 (very small) to 7 (very large); the default is 2.
* <a id="forest_files">**forest_files**</a> - **integer** from **0-7**
	* Specifies name of the forest files; if None, file names are governed by *indat* + extensions (_savepred.pickle, _savepred.csv, _savepredps.npy, _savepredd.npy); else file names are *name* + extensions.
* <a id="fs_other_sample">**fs_other_sample**</a> - **Boolean**
	* Determines whether the same sample as for the random forest estimation is used; the default is True, for which a random sample is taken from the overall sample. If False, the same sample as for the random forest estimation is deployed.
* <a id="fs_other_sample_share">**fs_other_sample_share**</a> - **float** between **0, 1**
	* Determines the share of the sample used for feature selection; the default is 0.2.
* <a id="fs_rf_threshold">**fs_rf_threshold**</a> - **float**
	* If feature selection is enabled, specifies a threshold in percent of loss of feature importance; features with values below the defined threshold are deleted; the default is 0.
* <a id="fs_yes">**fs_yes**</a> - **Boolean**
	* If True, feature selection is active; the default is False.

**g**

* <a id="gatet_flag">**gatet_flag**</a> - **Boolean**
	* If True, GATE(T)s are computed  for subpopulations by treatments; the default is False.
* <a id="gmate_no_evaluation_points">**gmate_no_evaluation_points**</a> - positive **integer**
	* Determines number of evaluation points for continuous variables; the default is 50.
* <a id="gmate_sample_share">**gmate_sample_share**</a> - **float**
	*  Specifies the share of the the prediction data used; the default is None. For the default, the share is computed as follows: if the number of observations in the prediction sample, $no^{\text{pred}}$, is smaller than $1,000$ the share is set to $1$; else the share is computed as $\frac{1000 + (no^{\text{pred}} - 1000)^{0.75}}{no^{\text{pred}}}$. If you set a number less or equal to $0$, the program sets the share according to the previous rule; else you may specify a valid share greater $0$, which the program uses.

**i**
* <a id="indata">**indata**</a> - **string**
	* Specifies the file name for the data, which is used for estimation; the file needs to be in *csv* format.

**k**
* <a id="knn_const">**knn_const**</a> - positive **integer**
	* Specifies the constant in number of neighbour in the asymptotic expansion formula of k-nn estimation; the default value is 1.
* <a id="knn_flag">**knn_flag**</a> - **Boolean**
	* If set False, the program uses Nadaraya-Watson estimation; if set True k-nn estimation is used; the default is True.
* <a id="knn_min_k">**knn_min_k**</a> - positive **integer**
	* Determines the minimum number of neighbours in k-nn estimation; the default value is 10.


**l**

* <a id="l_centering">**l_centering**</a>  - **Boolean**
	* Determines whether local centering is used; the default value is False.
* <a id="l_centering_cv_k">**l_centering_cv_k**</a> - **Boolean**
	* Specifies number of folds used in cross-validation; only valid if *l_centering_new_sample* is False;  the default is 5; note that the larger the value the better estimation quality but the longer computation time.
* <a id="l_centering_new_sample">**l_centering_new_sample**</a> - **Boolean**
	* If True, a random sample is used for computing predictions. This prediction is subtracted from the outcome variables. The data used to compute it, is not used anywhere else and thus the sample size is reduced. If False, cross-validation is used to compute the centering within the major samples (tree buildung, filling with $y$, and feature selection).  This version is computationally more intensive but statistically more efficient.
* <a id="l_centering_share">**l_centering_share**</a> - **float** between **0.1, 0.9**
	* Specifies the share of data used for estimating the conditional expectation $\mathbb{E}[Y_i|X = x]$; this data is not available for other estimations; the default value is 0.25.


**m**

* <a id="match_nn_prog_score">**match_nn_prog_score**</a> - **Boolean**
	* If True the program computes prognostic scores to find close neighbors for the mean correlated error; if False, Mahalanobis matching using all covariates is applied; the default is True.
* <a id="max_cats_z_vars">**max_cats_z_vars**</a> - positive **integer**
	* Specifies maximum number of categories for continuous $z$ variables; the default value is $no^{0.3}$, where $no$ is the number of observations in the training data.
* <a id="max_weight_share">**max_weight_share**</a> - **float** between **0, 1**
	* Regulates the maximum share of any weight; the default value 0.05; the program trims excess weights and renormalises the ATE, GATE, and IATE separately; due to renormalising, the final weights might be somewhat above the specified threshold.
* <a id="mce_vart">**mce_vart**</a> - **integer** taking values **0, 1, 2, 3**
	* Determines the rule, deployed for splitting when growing trees; if the value is 0, only the mean squared error of the regressions are considered; if the value is 1, the sum of the outcome MSE and MCE are considered; if the value is 2, the variance of the effect is chosen as the splitting criterion; with a value of 3, the criterion randomly switches between outcome MSE and MCE and penalty functions.
* <a id="min_dummy_obs">**min_dummy_obs**</a> - **Boolean**
	* If the program also screens covariates, i.e. when **screen_covariates** is True, the **min_dummy_obs** regulates the minimal number of observations in one category of a dummy variable for the dummy variable not to be removed from the data set; the default is set to 10.   
* <a id="mp_parallel">**mp_parallel**</a> - **None** or **integer number** larger **0**
	* Specifies the number of parallel processes; the default value is None; for a value of None the number of parallel processes is set to the integer part of 80 percent of the number of the cores; for values of 0 and 1, there are no parallel computations.
* <a id="mp_ray_objstore_multiplier">**mp_ray_objstore_multiplier**</a>  - **integer**
	* Increases internal default values for Ray object store to avoid crashes induced by full object stores; the default value is 1.
* <a id="mp_vim_type">**mp_vim_type**</a> - **integer** taking values **1, 2**
	* Decides how multiprocessing is implemented in the computation of feature importance; 1: multiprocessing over variables (fast but demanding in terms of memory), 2: multiprocessing over the  trees (slower but less demanding in terms of memory); the default is an automated rule; if the number of observations is less than 20,000 multiprocessing is done over variables else over trees.
* <a id="mp_weights_tree_batch">**mp_weights_tree_batch**</a> - **None** or **integer**
	* Determines how the forest is split into batches to compute the weights; in general, fewer batches demand more memory but less computing time; the number of batches is set by the program by default.
* <a id="mp_weights_type">**mp_weights_type**</a>  - **integer** taking values **1, 2**
	* Determines how multiprocessing is done in the computation of weights; if set to 1, the program parallelizes over groups of observations, which is more demanding in terms of memory; if set to 2, the program parallelizes over trees; latter option is less memory intensive but slower.
* <a id="mp_with_ray">**mp_with_ray**</a> - **Boolean**
	* If True, parallel processing is implemented via Ray; else the concurrent futures module is deployed; the concurrent futures module may be faster for small samples; Ray tends to more performant for larger samples the default is True.
* <a id="m_grid">**m_grid**</a> - positive **integer**
	* Sets the number of grid values which are logarithmically spaced between **m_min** and **m_max**; the default is 2.
* <a id="m_max_share">**m_max_share**</a> - float between **0, 1** or **integer** taking values **-1, -2**
	* Sets the maximum share of variables used for the next split; admissible values range from 0 to 1;  note that if **m_max_share** is set to 1, the algorithm corresponds to bagging trees; for a value of -1, the share obtains as $0.66*N$, where $N$ denotes the number of variables; for a value -2 the value is computed as $0.8*N$; the default is -1.
* <a id="m_min_share">**m_min_share**</a> - **float** between **0, 1** or **integer** taking values **-1, -2**
	* Sets the minimum share of variables used for  the next split; admissible values range from 0 to 1; if value is set to -1, the share is set to $0.1 * N$, where $N$ denotes the number of variables;  for a value of -2, the share obtains as $0.2 * N$; the default is -1.
* <a id="m_random_poisson">**m_random_poisson**</a> - **Boolean**
	* If True, the number of randomly selected variables is stochastic and obtains as $1+\text{Pois}(m-1)$, where $m$ denotes the number of variables used for splitting.

**n**


* <a id="nn_main_diag_only">**nn_main_diag_only**</a> - **Boolean**
	* If True, the program uses only main diagonal; if False, the program uses the inverse of the covariance matrix; the default is False.
* <a id="no_filled_plot">**no_filled_plot**</a> - **None** or **integer number** larger **0**
	* Determines that the plot is filled if there are more than a minimal number of specified points; the default is None; in this case  the plot is filled when there are more than 20 points.
* <a id="nw_bandw">**nw_bandw**</a> - positive **float** greater equal **1**
	* Sets the bandwidth for Nadaraya-Watson  estimation in form of the multiplier  of Silverman's optimal bandwidth; the default is None associated with a multiplier of 1.
* <a id="nw_kern_flag">**nw_kern_flag**</a> - **integer** taking values **1, 2**
	* Determines the kernel for Nadaraya-Watson estimation; a value of 1 sets the Epanechnikov kernel, a value of 2 sets a normal kernel; the default is 1.
* <a id="n_min_grid">**n_min_grid**</a> - **integer** larger **0**
	* Sets the number of grid values; the default is 1; if **n_min_grid** equals 1, **n_min_min** is used for leaf size.
* <a id="n_min_max">**n_min_max**</a> - **integer** taking values **-1, -2** or **integer** larger **0**
	* Determines the largest minimum leaf size; for a value of -2 the number is given by $\max(\sqrt(n)/10, 3)$; for a value of  -1, the number is given by $\max(\sqrt(n)/5, 5)$, where  $n$ is twice the number of observations in smallest treatment group; the default is -1.
* <a id="n_min_min">**n_min_min**</a> - **integer** taking values **-1, -2** or **integer** larger **0**
	* Determines smallest minimum leaf size; if a grid search is performed, an optimal value will be determined by evaluating the criterion function out-of-bag; for a value of -2: $\max(n^{0.4}/20, 3)$; -1: $\max(n^{0.4}/10, 5)$ where $n$ is twice the number of observations in smallest treatment group; the default is -1.

**o**

* <a id="outfiletext">**outfiletext**</a> - **string**
	* Specifies the name for the file, in which the text output from the program is written; if the name is not specified the default is set to the name of the data, deployed for estimation - **indata** - with the extension *txt* is used.
* <a id="outpfad">**outpfad**</a> - **string**
	* Creates a folder *out* in the application directory, in which the output files are written.
* <a id="output_type">**output_type**</a> - **integer** taking value **0, 1, 2** or **None**
	* Determines where the output goes; for a value of 0 the output goes to the terminal; for a value of 1, the output is sent to a file; for a value of 2, the output is sent to both terminal and file; the default is None.

**p**


* <a id="panel_data">**panel_data**</a> - **Boolean**
	* Indicates whether data are panel data; if True, the program computes clustered standard errors and performs only weight-based inference; the program uses **cluster_name** to infer panel unit; the default is None or False, implying no panel data.  
* <a id="panel_in_rf">**panel_in_rf**</a> - **Boolean**
	* If True, uses the panel structure also when building the random samples within the forest procedure if panel data are deployed; the default is True.
* <a id="post_est_stats">**post_est_stats**</a> - **Boolean**
	* If True, the program analyses the predictions by binary correlations or some regression type methods; the default is True.
* <a id="post_kmeans_max_tries">**post_kmeans_max_tries**</a> - positive **integer**
	* If **post_kmeans_yes** is True, gives maximum number of iterations in each replication to achieve convergence; default is 1000.
* <a id="post_kmeans_no_of_groups">**post_kmeans_no_of_groups**</a> - positive **integer**
	* If **post_kmeans_yes** is set True, sets the number of clusters to be formed; default values depend on the number of observation in the prediction sample $no^{\text{pred}}$. The default is None; the number of clusters to be formed is computed as follows: if $no^{\text{pred}} < 10,000$ the value is set to 5, else if $no^{\text{pred}} < 100,000$ the value is set to 10, for all other cases, the numbers is computed as $5 + \text{round}(no^{\text{pred}}/20,000)$. If a value other than None is specified, the number of clusters to be formed is computed as follows; if the specified number is smaller 2, the number is set according to the previous rule; otherwise, the number is obtained as the rounded number specified by the user.
* <a id="post_kmeans_replications">**post_kmeans_replications**</a> - **Boolean**
	* If **post_kmeans_yes** is set to True, sets number of replications with random start centers to avoid local extrema; the default value is 10.
* <a id="post_kmeans_yes">**post_kmeans_yes**</a> - **Boolean**
	* If True, program uses k-means clustering to analyse patterns in the estimated effects; the default is True.
* <a id="post_plots">**post_plots**</a> - **Boolean**
	* If True, the program delivers plots of estimated treatment effects in **pred_eff_data**; the default is True.  
* <a id="post_random_forest_vi">**post_random_forest_vi**</a> - **Boolean**
	* If True, program uses variable importance measure of predictive random forests to learn major factors influencing IATEs; the default is True.
* <a id="preddata">**preddata**</a> - **string**
	* Specifies the file name for the data for the effects estimation; the file needs to be in *csv* format.
* <a id="pred_mcf">**pred_mcf**</a> - **string**
		* If True, effects are estimated; default is True.
* <a id="p_diff_penalty">**p_diff_penalty**</a> - **None** or **integer** taking values **-1, 0** or float larger **0**
	* Sets value to further specify the utilized penalty function in combination with *mce_vart*; if *mce_vart* is 0, the *p_diff_penalty* is irrelevant; if  *mce_vart* is 1, for the default value of None or -1 the penalty is computed as follows $4\times((no^{\text{training}}\times \text{subsample share})^{0.8})/(no^{\text{training}} \times \text{subsample share})\times no^{\text{treatment}} \times (no^{\text{treatment}} -1)/2$; if the balancing tests indicate bad balance, you should increase the penalty above the default. If *mce_vart* is 2, the penalty is set by the program as follows: for the default value of None or -1, the penalty is $400\times((no^{\text{training}}\times \text{subsample share})^{0.8})/(no^{\text{training}} \times \text{subsample share})\times no^{\text{treatment}} \times (no^{\text{treatment}} -1)/2$. If the value is set to 0, there is no penalty; if *mce_vart* is equal to 3, by default (or if set to -1) the probability of setting the p-score is 0.5; if the specified probability is larger $1$, the program checks if the user-defined probability has been accidentally scaled in percent and rescales the number to obtain valid scores in the zero-one interval.    

**r**

* <a id="random_thresholds">**random_thresholds**</a> - **integer** taking value **0** or **integer** larger **0**
	* Regulates the random thresholds; if 0, no random thresholds are used; if set to values > 0, dictates number of random thresholds; the default is 20.
* <a id="relative_to_first_group_only">**relative_to_first_group_only**</a> - **Boolean**
	* If True, uses only effects relative to lowest treatment value; the default is True.

**s**

* <a id="save_forest">**save_forest**</a> - **Boolean**
	* If set True, the forest is saved for prediction.
* <a id="screen_covariates">**screen_covariates**</a> - **Boolean**
	* Determines whether the covariates are screened; the default is  True; to omit screening stage specify False.
* <a id="share_forest_sample">**share_forest_sample**</a> - **float** between **0, 1**
	* Determines the share used for predicting the outcome of interest, $y$; admissible values range from 0 to 1; the default is  0.5; the other share of the sample is used for building the forest.
* <a id="show_plots">**show_plots**</a> - **Boolean**
	* If True, the plots are shown, i.e. `plt.show()` is executed, else not; the default is True.  
* <a id="smooth_gates">**smooth_gates**</a> - **Boolean**
	* Specifies an alternative way to estimate GATEs for continuous variables; instead of discretizing variable, the GATE is evaluated at **smooth_gates_no_evaluation_points**. Since there are likely to be no  observations, a local neighbourhood around the evaluation points is considered; the default is True.
* <a id="smooth_gates_bandwidth">**smooth_gates_bandwidth**</a> - **float**
	* Defines the multiplier for SGATE aggregation; the default is 1.
* <a id="smooth_gates_no_evaluation_points">**smooth_gates_no_evaluation_points**</a> - positive **integer**
	* Sets the number of evaluation points for the GATE; the default is 50.
* <a id="stop_empty">**stop_empty**</a> - **integer** taking value **0** or integer larger **0**
	* Determines a stopping rule for splitting; the default is None. The program stops splitting the tree if the next $25$ randomly chosen variable did not lead to a new leaf; if  a value smaller is given, the program stops splitting after $25$ unsuccessful trials, else the stopping rule is governed by the user-defined number (rounded).
*  <a id="subsample_factor">**subsample_factor**</a> - **float** between **0, 1**
	* Sets the size of the subsampling sample; reduces the default subsample size by 1-subsample_factor; the default is $\min(0.67,(2*(n^{0.8})/n))$ ; $n$ is computed as twice the sample size in the smallest treatment group.
* <a id="support_check">**support_check**</a> - **integer** taking values **0, 1, 2**
	* Determines whether program checks for common support and sets rule how common support is enforced; for a value of 0 no common support is checked and enforced; for values of 1 and 2 common support is checked and enforced; for values 1 and 2, the support check is based on the estimated predictive random forests for each treatment probability but one; if the value is set to 1, the program uses min and max rules to enforce common support; for a value of 2, the program enforces minimum and maximum probabilities for all observations and all observations off support are removed. Out-of-bag predictions are used to avoid overfitting.
* <a id="support_min_p">**support_min_p**</a> - **float** between **0, 1**
	* Specifies minimal probability for common support if support check is set to 2; an observation is deleted if the conditional probability $p(d=m|x)$ is less or equal than **support_min_p** for at least one treatment $m$, the default is set to 0.01.
* <a id="support_quantil">**support_quantil**</a> -  **float** between **0, 1** and **integer** taking value **1**
	* Specifies how common support is enforced given that support check is set to 1; for a value of 1 the min-max rule is enforced, for values from 0 to and not including 1, the respective quantile is taken for the cut-offs; the default is 1.

**t**

* <a id="train_mcf">**train_mcf**</a> - **Boolean**
	* If True, a forest is estimated; the default is True


**w**


* <a id="weighted">**weighted**</a> - **Boolean**
	* If  True, the program uses sampling weights; if False no sampling weights are used. If  1, sampling weights specified in **w_name** will be used; the default is False.
* <a id="weight_as_sparse">**weight_as_sparse**</a> - **Boolean**
	* Determines whether the weight matrix is a sparse matrix; the default is True.

**v**

* <a id="variable_importance_oob">**variable_importance_oob**</a> - **Boolean**
	* If True, the program  computes the variable importance based on permuting every single characteristic, $x$, in the out-of-bag prediction; this  exercise is time consuming; the default is False.
*  <a id="verbose">**verbose**</a> - **Boolean**
	*  If True,  the output of the program is printed whilst running; if False the output is suppressed. The default is True.

**_**

* <a id="_max_cats_cont_vars">**_max_cats_cont_vars**</a> - positive **integer**
	* Determines how to discretise continuous variables, i.e. regulates the maximum number of categories for continuous variables.
* <a id="_max_save_values">**_max_save_values**</a> - positive **integer**
	* Is only relevant for continuous features; saves value of $x$ only if less than specified; default value is 50.
* <a id="_seed_sample_split">**_seed_sample_split**</a> - positive **integer**
	* Sets seed for building forest; the default is 67567885.
* <a id="_smaller_sample">**_smaller_sample**</a> - **float** between **0,1**
	*  Determines whether program shall be tested with smaller sample.
* <a id="_with_output">**_with_output**</a> - **Boolean**
	* If True, print statements are executed; the default is True.
