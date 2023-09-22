# Python API Extended


```python
ModifiedCausalForest(
cf_alpha_reg_grid=CF_ALPHA_REG_GRID,
cf_alpha_reg_max=CF_ALPHA_REG_MAX,
cf_alpha_reg_min=CF_ALPHA_REG_MIN,
cf_boot=CF_BOOT,
cf_chunks_maxsize=CF_CHUNKS_MAXSIZE,
cf_m_grid=CF_M_GRID,
cf_m_random_poisson=CF_M_RANDOM_POISSON,
cf_m_share_max=CF_M_SHARE_MAX,
cf_m_share_min=CF_M_SHARE_MIN,
cf_match_nn_prog_score=CF_MATCH_NN_PROG_SCORE,
cf_mce_vart=CF_MCE_VART,
cf_p_diff_penalty=CF_P_DIFF_PENALTY,
cf_n_min_grid=CF_N_MIN_GRID,
cf_n_min_max=CF_N_MIN_MAX,
cf_n_min_min=CF_N_MIN_MIN,
cf_n_min_treat=CF_N_MIN_TREAT,
cf_nn_main_diag_only=CF_NN_MAIN_DIAG_ONLY,
cf_random_thresholds=CF_RANDOM_THRESHOLDS,
cf_subsample_factor_eval=CF_SUBSAMPLE_FACTOR_EVAL,
cf_subsample_factor_forest=CF_SUBSAMPLE_FACTOR_FOREST,
cf_vi_oob_yes=CF_VI_OOB_YES,
cs_adjust_limits=CS_ADJUST_LIMITS,
cs_max_del_train=CS_MAX_DEL_TRAIN,
cs_min_p=CS_MIN_P,
cs_quantil=CS_QUANTIL,
cs_type=CS_TYPE,
ct_grid_dr=CT_GRID_DR,
ct_grid_nn=CT_GRID_NN,
ct_grid_w=CT_GRID_W,
dc_check_perfectcorr=DC_CHECK_PERFECTCORR,
dc_clean_data=DC_CLEAN_DATA,
dc_min_dummy_obs=DC_MIN_DUMMY_OBS,
dc_screen_covariates=DC_SCREEN_COVARIATES,
fs_other_sample_share=FS_OTHER_SAMPLE_SHARE,
fs_rf_threshold=FS_RF_THRESHOLD,
fs_yes=FS_YES,
gen_d_type=GEN_D_TYPE,
gen_iate_eff=GEN_IATE_EFF,
gen_mp_parallel=GEN_MP_PARALLEL,
gen_outfiletext=GEN_OUTFILETEXT,
gen_outpath=GEN_OUTPATH,
gen_output_type=GEN_OUTPUT_TYPE,
gen_panel_data=GEN_PANEL_DATA,
gen_panel_in_rf=GEN_PANEL_IN_RF,
gen_weighted=GEN_WEIGHTED,
lc_cs_cv=LC_CS_CV,
lc_cs_cv_k=LC_CS_CV_K,
lc_cs_share=LC_CS_SHARE,
lc_uncenter_po=LC_UNCENTER_PO,
lc_yes=LC_YES,
p_amgate=P_AMGATE,
p_atet=P_ATET,
p_bgate=P_BGATE,
p_choice_based_probs=P_CHOICE_BASED_PROBS,
p_ci_level=P_CI_LEVEL,
p_choice_based_sampling=P_CHOICE_BASED_SAMPLING,
p_cluster_std=P_CLUSTER_STD,
p_cond_var=P_COND_VAR,
p_gates_minus_previous=P_GATES_MINUS_PREVIOUS,
p_gates_smooth=P_GATES_SMOOTH,
p_gates_smooth_bandwidth=P_GATES_SMOOTH_BANDWIDTH,
p_gates_smooth_no_evalu_points=P_GATES_SMOOTH_NO_EVALU_POINTS,
p_gatet=P_GATET,
p_gmate_no_evalu_points=P_GMATE_NO_EVALU_POINTS,
p_gmate_sample_share=P_GMATE_SAMPLE_SHARE,
p_iate=P_IATE,
p_iate_m_ate=P_IATE_M_ATE,
p_iate_se=P_IATE_SE,
p_knn=P_KNN,
p_knn_const=P_KNN_CONST,
p_knn_min_k=P_KNN_MIN_K,
p_max_cats_z_vars=P_MAX_CATS_Z_VARS,
p_max_weight_share=P_MAX_WEIGHT_SHARE,
p_nw_bandw=P_NW_BANDW,
p_nw_kern=P_NW_KERN,
p_se_boot_ate=P_SE_BOOT_ATE,
p_se_boot_gate=P_SE_BOOT_GATE,
p_se_boot_iate=P_SE_BOOT_IATE,
post_bin_corr_threshold=POST_BIN_CORR_THRESHOLD,
post_bin_corr_yes=POST_BIN_CORR_YES,
post_est_stats=POST_EST_STATS,
post_kmeans_max_tries=POST_KMEANS_MAX_TRIES,
post_kmeans_no_of_groups=POST_KMEANS_NO_OF_GROUPS,
post_kmeans_replications=POST_KMEANS_REPLICATIONS,
post_kmeans_yes=POST_KMEANS_YES,
post_plots=POST_PLOTS,
post_random_forest_vi=POST_RANDOM_FOREST_VI,
post_relative_to_first_group_only=POST_RELATIVE_TO_FIRST_GROUP_ONLY,
var_bgate_name=VAR_BGATE_NAME,
var_cluster_name=VAR_CLUSTER_NAME,
var_d_name=VAR_D_NAME,
var_id_name=VAR_ID_NAME,
var_w_name=VAR_W_NAME,
var_x_balance_name_ord=VAR_X_BALANCE_NAME_ORD,
var_x_balance_name_unord=VAR_X_BALANCE_NAME_UNORD,
var_x_name_always_in_ord=VAR_X_NAME_ALWAYS_IN_ORD,
var_x_name_always_in_unord=VAR_X_NAME_ALWAYS_IN_UNORD,
var_x_name_ord=VAR_X_NAME_ORD,
var_x_name_unord=VAR_X_NAME_UNORD,
var_x_name_remain_ord=VAR_X_NAME_REMAIN_ORD,
var_x_name_remain_unord=VAR_X_NAME_REMAIN_UNORD,
var_y_name=VAR_Y_NAME,
var_y_tree_name=VAR_Y_TREE_NAME,
var_z_name_list=VAR_Z_NAME_LIST,
var_z_name_ord=VAR_Z_NAME_ORD,
var_z_name_unord=VAR_Z_NAME_UNORD,
_int_descriptive_stats=_INT_DESCRIPTIVE_STATS,
_int_dpi=_INT_DPI,
_int_fontsize=_INT_FONTSIZE,
_int_max_cats_cont_vars=_INT_MAX_CATS_CONT_VARS,
_int_max_save_values=_INT_MAX_SAVE_VALUES,
_int_mp_ray_del=_INT_MP_RAY_DEL,
_int_mp_ray_objstore_multiplier=_INT_MP_RAY_OBJSTORE_MULTIPLIER,
_int_mp_ray_shutdown=_INT_MP_RAY_SHUTDOWN,
_int_mp_vim_type=_INT_MP_VIM_TYPE,
_int_mp_weights_tree_batch=_INT_MP_WEIGHTS_TREE_BATCH,
_int_mp_weights_type=_INT_MP_WEIGHTS_TYPE,
_int_no_filled_plot=_INT_NO_FILLED_PLOT,
_int_return_iate_sp=_INT_RETURN_IATE_SP,
_int_seed_sample_split=_INT_SEED_SAMPLE_SPLIT,
_int_share_forest_sample=_INT_SHARE_FOREST_SAMPLE,
_int_show_plots=_INT_SHOW_PLOTS,
_int_verbose=_INT_VERBOSE,
_int_weight_as_sparse=_INT_WEIGHT_AS_SPARSE,
_int_weight_as_sparse_splits=_INT_WEIGHT_AS_SPARSE_SPLITS,
_int_with_output=_INT_WITH_OUTPUT)
```


## Variable Names

- <a id="var_bgate_name"><strong>var_bgate_name</strong></a>
	* Variable to balance the GATEs on. Only relevant if **p_bgate** is set to True.

- <a id="var_cluster_name"><strong>var_cluster_name</strong></a>
	* Name of cluster variable.

- <a id="var_d_name"><strong>var_d_name</strong></a>
	* Name of treatment.

- <a id="var_id_name"><strong>var_id_name</strong></a>
	* Identifier. If None or an empty list is specified, identifier will be added to the data. Default is None.

- <a id="var_w_name"><strong>var_w_name</strong></a>
	* Name of variable containing weights.

- <a id="var_x_balance_name_ord"><strong>var_x_balance_name_ord</strong></a>
	* Names of ordered variables for balancing tests.

- <a id="var_x_balance_name_unord"><strong>var_x_balance_name_unord</strong></a>
	* Names of unordered variables for balancing tests.

- <a id="var_x_name_always_in_ord"><strong>var_x_name_always_in_ord</strong></a>
	* Names of ordered variables, which should be always included when deciding on next split.

- <a id="var_x_name_always_in_unord"><strong>var_x_name_always_in_unord</strong></a>
	* Names of unordered variables, which should be always included when deciding on next split.

- <a id="var_x_name_ord"><strong>var_x_name_ord</strong></a>
	*	Names of ordered features.

- <a id="var_x_name_unord"><strong>var_x_name_unord</strong></a>
	* Names of unordered features.

- <a id="var_x_name_remain_ord"><strong>var_x_name_remain_ord</strong></a>
	*	Names of ordered features excluded from preliminary feature selection. Default is None.

- <a id="var_x_name_remain_unord"><strong>var_x_name_remain_unord</strong></a>
	* Names of unordered variables excluded from preliminary feature selection. Default is None.

- <a id="var_y_name"><strong>var_y_name</strong></a>
	* Name(s) of outcome variable(s).

- <a id="var_y_tree_name"><strong>var_y_tree_name</strong></a>
	* Name of variable to build trees. If None or empty list, the first string in **var_y_name** is used.

- <a id="var_z_name_list"><strong>var_z_name_list</strong></a>
	* Names of variables for heterogeneity analysis.

- <a id="var_z_name_ord"><strong>var_z_name_ord</strong></a>
	* Names of ordered variables to define policy relevant heterogeneity.

- <a id="var_z_name_unord"><strong>var_z_name_unord</strong></a>
 * Names of unordered variables to define policy relevant heterogeneity.


## Parameters

**cf** or **causal forest**

- <a id="cf_alpha_reg_grid"><strong>cf_alpha_reg_grid</strong></a>
	* Number of grid values. Default is 1.

- <a id="cf_alpha_reg_max"><strong>cf_alpha_reg_max</strong></a>
	* Maximum alpha. May take values between 0 and 0.5. Default is 0.15.

- <a id="cf_alpha_reg_min"><strong>cf_alpha_reg_min</strong></a>
	* Minimum alpha. May take values between 0 and 0.4. Default is 0.05.

- <a id="cf_boot"><strong>cf_boot</strong></a>
	* Number of bootstrap or subsampling replications. Default is 1000.

- <a id="cf_chunks_maxsize"><strong>cf_chunks_maxsize</strong></a>
	* Specifies maximum allowed number of observations per block. If the number is larger than the sample size, there is no random splitting. For the default, None, the number is set to $round(60000 + \sqrt{n - 60000)}$, where $n$ denotes the number of observations.

- <a id="cf_m_grid"><strong>cf_m_grid</strong></a>
	* Number of grid values logarithmically spaced between the lower and upper bounds. If **cf_m_grid** is 1, **cf_m_share** is equal to 0.5(**cf_m_share_min** + **cf_m_share_max**). The default is 2.

- <a id="cf_m_random_poisson"><strong>cf_m_random_poisson</strong></a>
	*  If True, number of randomly selected variables is drawn from a Poisson distribution with expectation m - 1. If m > 10, the default is set to True. Otherwise, the default is set to False.

- <a id="cf_m_share_max"><strong>cf_m_share_max</strong></a>
	* Maximum share of variables to be included in tree growing. Viable range is from 0 to 1 excluding the bounds. Default is 0.6.

- <a id="cf_m_share_min"><strong>cf_m_share_min</strong></a>
	* Minimum share of variables to be included in tree growing. Viable range is from 0 to 1 excluding the bounds. Default is 0.1.

- <a id="cf_match_nn_prog_score"><strong>cf_match_nn_prog_score</strong></a>
	* Specifies matching procedure in the MCE computation. If set to False, Mahalanobis matching is deployed. If set to True, prognostic scores are deployed. Default is True.

- <a id="cf_mce_vart"><strong>cf_mce_vart</strong></a>
	* Determines splitting rule in the tree-growing process. When set to 0, only the MSEs are considered. When set to 1, the sum of MSE and MCE are used (MSE-MCE criterion). When set to 2, the effect  heterogeneity maximizing rule of Wager and Athey (2018) is deployed. When set to 3, the rule randomly switches between outcome and MSE-MCE criterion in combination with the penalty function. Default is None, which implies 1.

- <a id="cf_n_min_grid"><strong>cf_n_min_grid</strong></a>
	* Determines number of grid values. Default is 1. For the default of 1, **cf_n_min**= 0.5(**cf_n_min_min**+**cf_n_min_max**).

- <a id="cf_n_min_max"><strong>cf_n_min_max</strong></a>
	* Determines largest minimum leaf size. The default is $\max(\sqrt{n_d} / 6,3)$, where $n_d$ denotes the number of observations in the smallest treatment arm. All values are multiplied by the number of treatments.

- <a id="cf_n_min_min"><strong>cf_n_min_min</strong></a>
	* Determines smallest minimum leaf size; specify an integer larger than 2. The default is $n_d^{0.4}/6$.

- <a id="cf_n_min_treat"><strong>cf_n_min_treat</strong></a>
	* Specifies minimum number of observations per treatment in leaf. The default is 0.5(**cf_n_min_min** + **cf_n_min_max**) / number of of treatments / 4. The minimum is 2.

- <a id="cf_nn_main_diag_only"><strong>cf_nn_main_diag_only</strong></a>
	* Relevant if **cf_match_nn_prog_score** is set to False. If set to True, only the main diagonal is used. If False, the inverse of the covariance matrix is used. Default is False.

- <a id="cf_p_diff_penalty"><strong>cf_p_diff_penalty</strong></a>
	* Sets value to further specify the utilized penalty function in combination with *mce_vart*; if *mce_vart* is 0, the *p_diff_penalty* is irrelevant; if  *mce_vart* is 1, for the default value of None or -1 the multiplier of the penalty is computed as follows $2((no^{train} \times \text{subsample share} )^{0.9})/(no^{train}\times \text{subsample share})\times(\text{no of treatments} \times(\text{no of treatments}-1)/2)^{0.5}$; if the balancing tests indicate bad balance, you should increase the penalty above the default. If *mce_vart* is 2, the penalty is set by the program as follows: for the default value of None or -1, the penalty is $200((no^{train} \times \text{subsample share} )^{0.9})/(no^{train}\times \text{subsample share})\times(\text{no of treatments} \times(\text{no of treatments}-1)/2)^{0.5}$; increase the penalty if balancing tests indicate bad balance. If the value is set to 0, there is no penalty; if *mce_vart* is equal to 3, by default the probability of setting the p-score is 0.5; if the specified probability is larger $1$, the program checks if the user-defined probability has been accidentally scaled in percent and rescales the number to obtain valid scores in the zero-one interval.

- <a id="cf_random_thresholds"><strong>cf_random_thresholds</strong></a>
	* For values larger 0, only **cf_random_thresholds** are considered for splitting. Randomisation is repeated for each split. If set to 0, all possible splits are considered. The default value is $4 + n_t^{0.2}$, where $n_t$ denotes the length of the training data.

- <a id="cf_subsample_factor_eval"><strong>cf_subsample_factor_eval</strong></a>
	* Determines the subsampling size. If set to False, there is no subsampling in the evaluation subsample. If True or None, the size is 2 times the subsampling size used for the tree building. For a float greater 0, the multiplier of the subsample size used for tree building is deployed. Subsampling in the evaluation may speed up computations and reduce memory demands. The default is True.

- <a id="cf_subsample_factor_forest"><strong>cf_subsample_factor_forest</strong></a>
	* Determines size of subsampling sample for the tree building. The default share is $\min(0.67,(2*(n^{0.8})/n))$, where $n$ is 2 times the sample size of the smallest treatment arm. The viable range is (0, 0.8]. The actual share of the subsample is equal to **cf_def_share** times **cf_subsample_factor_forest**.

- <a id="cf_vi_oob_yes"><strong>cf_vi_oob_yes</strong></a>
	* If set to True, the causal forest's variable importance is computed. The variable importance measure is based on permuting every single feature in the OOB prediction. The default is False.

**cs** or **common support**

- <a id="cs_adjust_limits"><strong>cs_adjust_limits</strong></a>
	* This parameter reduces the restrictiveness of the common support criterion, which increases in the number of treatments. The upper limit is multiplied by 1 + **cs_support_adjust_limits**, and the minimum by 1 - **cs_support_adjust_limits**. The default is None and renders **cs_support_adjust_limits** being equal to 0.05 (number of treatments - 2). If **cs_type** is 0 or None, there is no adjustment. Default is None.

- <a id="cs_max_del_train"><strong>cs_max_del_train</strong></a>
	* If share of observations in training data used for forest data that are off support is larger than **cs_support_max_del_train**, program terminates. Viable range is between 0 and 1. Default is 0.5.

- <a id="cs_min_p"><strong>cs_min_p</strong></a>
	* If **cs_min_p** equals 2, an observation is deleted if at least one of the estimated propensities is less or equal than **cs_support_min_p**. The default is 0.01.

- <a id="cs_quantil"><strong>cs_quantil</strong></a>
	* If **cs_type** is 1, the min-max rule is deployed. For values between 0 and 1 the respective quantiles are taken. The default is the mi-max rule.

- <a id="cs_type"><strong>cs_type</strong></a>
	* Specifies type of common support adjustment. If set to 0, there is no common support adjustment. If set to 1 or 2, the support check is based on the estimated classification regression forests. For 1, the min-max rules for the estimated probabilities in the treatment subsamples are deployed. For 2, the minimum and maximum probabilities for all observations are deployed. All observations off support are removed. Note that out-of-bag predictions are used to avoid overfitting (which leads to a too large reduction in observations). The default is 1.

**ct** or **continuous treatment**

- <a id="ct_grid_dr"><strong>ct_grid_dr</strong></a>
	* Specifies number of grid point for discretization of continuous treatment. Used to approximate the dose response function. The grid is defined in terms of the quantiles of the continuous treatment. The default is 100.

- <a id="ct_grid_nn"><strong>ct_grid_nn</strong></a>
	* Specifies number of grid point for the discretization of the continuous treatment. Grid is defined in terms of quantiles of the continuous treatment. Default is 10.

**dc** or **data cleaning**

- <a id="dc_check_perfectcorr"><strong>dc_check_perfectcorr</strong></a>
	* If **dc_screen_covariates** is True, variables that are perfectly correlated with others will be deleted if True. Default is True.

- <a id="dc_clean_data"><strong>dc_clean_data</strong></a>
	* If True, all missing and unnecessary variables are removed. The default is True.

- <a id="dc_min_dummy_obs"><strong>dc_min_dummy_obs</strong></a>
	* If **dc_screen_covariates** is True, dummy variables with less than **dc_min_dummy_obs** will be deleted. The default is 10.

- <a id="dc_screen_covariates"><strong>dc_screen_covariates</strong></a>
	* If True, covariates are screened. The default is True.

**fs** or **feature selection**

- <a id="fs_other_sample"><strong>fs_other_sample</strong></a>
	* If True, a random sample from training data is used, which will not be used for the causal forest. If False, the same data is used for feature selection and the causal forest. The default is True. For small datasets, we recommend changing the parameter to False.

- <a id="fs_other_sample_share"><strong>fs_other_sample_share</strong></a>
	* If **fs_other_sample** is set to True, **fs_other_sample_share** determines sample share for feature selection. Default is 0.33.


- <a id="fs_rf_threshold"><strong>fs_rf_threshold</strong></a>
	* Specifies threshold for feature selection as relative loss of variable importance (in percent). The default is 1.

- <a id="fs_yes"><strong>fs_yes</strong></a>
	* If True, feature selection is active. Default is False.

**gen** or **general**

- <a id="gen_d_type"><strong>gen_d_type</strong></a>
	* Specifies type of treatment. Choose between 'discrete' and 'continuous'. Default is 'discrete'.

- <a id="gen_iate_eff"><strong>gen_iate_eff</strong></a>
	* If True, the second round of IATEs are estimated based on switching training and estimation subsamples. If False, execution time is considerable faster. Default is False.

- <a id="gen_mp_parallel"><strong>gen_mp_parallel</strong></a>
	* Specifies the number of parallel processes in the parallelization. For values of 0 and 1, there are no parallel computations. By default, the number of parallel processes is set to 80 percent of the logical cores.

- <a id="gen_outfiletext"><strong>gen_outfiletext</strong></a>
	* Specifies filename, in which the text output is stored. If None, name of the training data with extension .out is used.

- <a id="gen_output_type"><strong>gen_output_type</strong></a>
	* Regulates where text output is rendered. When set to 0, output goes to the terminal. When set to 1, the output goes exclusively to the text file. When set to 2, the output goes to the file and terminal. The default is 2.

- <a id="gen_outpath"><strong>gen_outpath</strong></a>
	* Specifies path to  where the output is written too. If this is None, an ``out`` directory will be created in the current working directory.

- <a id="gen_panel_data"><strong>gen_panel_data</strong></a>
	* If set to True, clustered standard errors are computed. If None or False, data is assumed to have no panel structure. Default is False.

- <a id="gen_panel_in_rf"><strong>gen_panel_in_rf</strong></a>
	* If True, the panel structure is also used when drawing the random samples in the forest growing process. The default is True.

- <a id="gen_weighted"><strong>gen_weighted</strong></a>
	* If set to True, sampling weights are used. The sampling weights are specified via **var_w_name**. Sampling weights slow down the program. The default is False.

- <a id="lc_cs_cv"><strong>lc_cs_cv</strong></a>
	* Specifies which data to use for local centering and common support. If set to True, cross-validation is used. If False, a random sample is used, which is not used for the causal forest later. The default is True.

- <a id="lc_cs_cv_k"><strong>lc_cs_cv_k</strong></a>
	* Specifies number of folds for cross validation. The default is 5.

- <a id="lc_cs_share"><strong>lc_cs_share</strong></a>
	* Specifies share of data used for conditonal outcome estimation. Viable range is from 0.1 to 0.9. The default is 0.25.

- <a id="lc_uncenter_po"><strong>lc_uncenter_po</strong></a>
	* If True, predicted potential outcomes are added to the data output. Default is True.

- <a id="lc_yes"><strong>lc_yes</strong></a>
	* If True, local centering is deployed. The default is True.

- <a id="p_amgate"><strong>p_amgate</strong></a>
	* If set to True, the program computes AMGATEs.  If no variables are specified for GATE estimation, **p_amgate** is set to False. The default is False.

- <a id="p_atet"><strong>p_atet</strong></a>
	* If True, the average effects are estimated by treatment group. This works only if at least one heterogeneity variable is defined. The default is False.

- <a id="p_bt_yes"><strong>p_bt_yes</strong></a>
	* If True, executes balancing test based on wights. Requires weight based inference. Relevance of this test not fully clear. Default is False.

- <a id="p_choice_based_probs"><strong>p_choice_based_probs</strong></a>
	* Specifies sampling probabilities. These weights are used for (G)ATEs only. Treatment information must be available in the prediction file.

- <a id="p_ci_level"><strong>p_ci_level</strong></a>
	* Specifies confidence level for plots. Default is 0.9.

- <a id="p_choice_based_sampling"><strong>p_choice_based_sampling</strong></a>
	* If True, implements choice based sampling. The default is False.

- <a id="p_cluster_std"><strong>p_cluster_std</strong></a>
	* If True, clustered standard errors are computed. If False, standard errors are not clustered. Default is False. Option will be automatically turned on, if panel data option is activated.

- <a id="p_cond_var"><strong>p_cond_var</strong></a>
	* If False, variance estimation uses $var(wy)$. If True, conditonal mean and variances are used. The default is True.

- <a id="p_gates_minus_previous"><strong>p_gates_minus_previous</strong></a>
	* If set to True, GATES will be compared to GATEs computed at the previous evaluation point. GATE estimation is a bit slower as it is not optimized for multiprocessing. No plots are shown. Default is False.

- <a id="p_gates_smooth"><strong>p_gates_smooth</strong></a>
	* Alternative way to estimate GATEs for continuous variables. Instead of discretizing the heterogeneity variable, the GATE is evaluated at a local neighbourhood around the **p_gates_smooth_no_evalu_points**. Default is True.

- <a id="p_gates_smooth_bandwidth"><strong>p_gates_smooth_bandwidth</strong></a>
	* Specifies multiplier for smoothed GATE aggregation. Default is 1.

- <a id="p_gates_smooth_no_evalu_points"><strong>p_gates_smooth_no_evalu_points</strong></a>
	* Specifies number of evaluation points. Default is 50.

- <a id="p_gatet"><strong>p_gatet</strong></a>
	* If set to True, GATEs and GATETs are computed. If no variables are specified for GATE estimation, **p_bgate** is set to False. The default is False.

- <a id="p_gmate_no_evalu_points"><strong>p_gmate_no_evalu_points</strong></a>
	* Specifies number of evaluation points for continuous variables for the AMGATE. The default is 50.

- <a id="p_iate"><strong>p_iate</strong></a>
	* IF True, IATEs will be estimated. The default is True.

- <a id="p_iate_m_ate"><strong>p_iate_m_ate</strong></a>
	* If True, IATE(x) - ATE will be estimated. The default is False.

- <a id="p_iate_se"><strong>p_iate_se</strong></a>
	* If True, standard errors for the IATEs will be computed. The default is False.

- <a id="p_knn"><strong>p_knn</strong></a>
	* If set to False, the program uses Nadaraya-Watson for variance estimation. If True, k-nearest neighbor estimation is used. Nadaraya-Watson estimation gives a better approximation of the variance, but k-nearest neighbor estimation is faster. Default is True.

- <a id="p_knn_const"><strong>p_knn_const</strong></a>
	* If True, considers constant in number of neighbour asymptotic expansion formula of k-nearest neighbor. The default is True.

- <a id="p_knn_min_k"><strong>p_knn_min_k</strong></a>
	* Specifies minimum number of neighbors in k-nearest neighbor estimation. The default is 10.

- <a id="p_max_cats_z_vars"><strong>p_max_cats_z_vars</strong></a>
	* Specifies the maximum number of categories for discretizing heterogeneity variable in the GATE estimation. Default is $n^{0.3}$.

- <a id="p_max_weight_share"><strong>p_max_weight_share</strong></a>
	* Specifies maximum share of any weight. Viable range is from 0 to 1. The rule is enforced by trimming excess weights and renormalising. This is done for each ATE, GATE, and IATE separately. The default is 0.05.


- <a id="p_nw_bandw"><strong>p_nw_bandw</strong></a>
	* Specifies bandwidth for Nadaraya-Watson estimation in terms of a multiplier to Silverman's optimal bandwidth. The default is 1.


- <a id="p_nw_kern"><strong>p_nw_kern</strong></a>
	* Specifies kernel for Nadaraya-Watson estimation. If set to 1, Epanechikov. If set to 2, the  normal. The default is 1.


- <a id="p_se_boot_ate"><strong>p_se_boot_ate</strong></a>
	* If True, $(w_{ji} y_i)$ are bootstrapped.


- <a id="p_se_boot_gate"><strong>p_se_boot_gate</strong></a>
	* If True, use bootstrap for GATE standard errors. The default is False.

- <a id="p_se_boot_iate"><strong>p_se_boot_iate</strong></a>
	* If True, use 199 bootstraps (block-bootstrap). If **p_cluster_std** is False, **p_se_boot_iate** is by default False. If **p_cluster_std** is False, **p_se_boot_iate** is by default False. If **p_cluster_std** is True, the default is True.

- <a id="post_bin_corr_threshold"><strong>post_bin_corr_threshold</strong></a>
	* Specifies minimum threshhold of absolute correlation. Default is 0.1.

- <a id="post_bin_corr_yes"><strong>post_bin_corr_yes</strong></a>
	* If True, binary predictions are checked. The default is True.

- <a id="post_est_stats"><strong>post_est_stats</strong></a>
	* Analyses predictions by binary correlations or some regression type methods. Default is True. The default is overwritten to be False if **p_iate** is False.

- <a id="post_kmeans_max_tries"><strong>post_kmeans_max_tries</strong></a>
	* If **post_kmeans_yes** is True, **post_kmeans_max_tries** sets the maximum number of iterations in each replication to archive convergence. Default is 1000.

- <a id="post_kmeans_no_of_groups"><strong>post_kmeans_no_of_groups</strong></a>
	* If **post_kmeans_yes** is True, **post_kmeans_no_of_groups** determines number of clusters. Information is passed over in the form  of an integer list or tuple. If not otherwise specified, the default is a list of 5 values: $[a, b, c, d, e]$, where depending on $n$, c takes values from 5 to 10. If c is smaller than 7, $a=c-2$, $b=c-1$, $d=c+1$, $e=c+2$ else $a=c-4$, $b=c-2$, $d=c+2$, $e=c+4$.

- <a id="post_kmeans_replications"><strong>post_kmeans_replications</strong></a>
	* If **post_kmeans_yes** is True, **post_kmeans_replications** regulates the number of replications for the k-means clustering algorithm. The default is 10.

- <a id="post_kmeans_yes"><strong>post_kmeans_yes</strong></a>
	* If True, the program uses k-means clustering to analyse patterns in the estimated effects. The default is True.

- <a id="post_plots"><strong>post_plots</strong></a>
	* If True, plots of estimated treatment effects are generated. The default is True.

- <a id="post_random_forest_vi"><strong>post_random_forest_vi</strong></a>
	* If True, predictive random forests and the associated variable importance measures are deployed to learn factors influencing the IATEs. The default is True.

- <a id="post_relative_to_first_group_only"><strong>post_relative_to_first_group_only</strong></a>
	* If True, only the effects of the lowest treatment value are used. The default is True.

- <a id="_int_descriptive_stats"><strong>_int_descriptive_stats</strong></a>
	* If True, descriptive stats of input and output files are printed. The default is True.

- <a id="_int_dpi"><strong>_int_dpi</strong></a>
	* Specifies dpi for plots. Default is 500.


- <a id="_int_fontsize"><strong>_int_fontsize</strong></a>
	* Regulates font size for legends varying from 1 (very small) to 7 (very large). The default is 2.

- <a id="_int_max_cats_cont_vars"><strong>_int_max_cats_cont_vars</strong></a>
	* Regulates if continuous variables are discretized. The smaller the admissible cardenality, the faster the computation. By default the functionality is not used. The default is 50.

- <a id="_int_max_save_values"><strong>_int_max_save_values</strong></a>
	* Save value of x only if it is smaller than 50 (continuous variables).

- <a id="_int_mp_ray_del"><strong>_int_mp_ray_del</strong></a>
	* Determines Ray objects to be deleted. Tuple with any of the following is admissible 'refs', 'rest', and 'None'. 'refs' is the default and deletes references to the Ray object store, 'rest' deletes all other objects of the Ray task. 'refs' and 'rest' can be combined. If nothing shall be deleted, use ('None',). None leads to the default.

- <a id="_int_mp_ray_objstore_multiplier"><strong>_int_mp_ray_objstore_multiplier</strong></a>
	* Relevant if **_mp_ray_shutdown** is set to True. By setting this parameter, the internal default values for Ray object store is set above 1. This may help to avoid program abortions due to the object store being full. The default is True.


- <a id="_int_mp_vim_type"><strong>_int_mp_vim_type</strong></a>
	* Regulates type of multiprocessing when computing the variable importance. When set to 1, multiprocessing is variable-based (fast, memory-intensive). When set to 2, it is bootstrap-based (slow, less memory). If n < 20000, the default is 1, 2 otherwise.

- <a id="_int_mp_weights_tree_batch"><strong>_int_mp_weights_tree_batch</strong></a>
	* Determines details of the weight computation. If the forest is split into few batches, computations tend to be speedier but more memory-intensive. The default is automatically determined.

- <a id="_int_mp_weights_type"><strong>_int_mp_weights_type</strong></a>
	* Regulates type of multiprocessing for weights computation. If set to 1, multiprocessing is based on groups of observations (fast, memory-intensive). If set to 2, multiprocessing is tree based (slow, less memory-intensive). Default is 1.

- <a id="_int_no_filled_plot"><strong>_int_no_filled_plot</strong></a>
	* Use filled plots for for more than **_int_no_filled_plot** points. The default is 20.

- <a id="_int_return_iate_sp"><strong>_int_return_iate_sp</strong></a>
	* If True, data with predictions is returned despite *with_output* being set False. Default is False.

- <a id="_int_seed_sample_split"><strong>_int_seed_sample_split</strong></a>
	* Seeding is redone when building forest. The default is 67567885. 

- <a id="_int_share_forest_sample"><strong>_int_share_forest_sample</strong></a>
	*  Determines share of sample used for predicting $y$ given  forests. The default is 0.5.

- <a id="_int_show_plots"><strong>_int_show_plots</strong></a>
	* If True, ``plt.show()`` is executed. Default is True.

- <a id="_int_verbose"><strong>_int_verbose</strong></a>
	* If True, the program provides information on the status quo. The default is True.

- <a id="_int_weight_as_sparse"><strong>_int_weight_as_sparse</strong></a>
	* Specifies if the weights matrix is presented as a sparse matrix. The default is True.

- <a id="_int_weight_as_sparse_splits"><strong>_int_weight_as_sparse_splits</strong></a>
	* Determines in how many pieces the sparse weight matrix is computed. The default is int(Rows of prediction data * rows of Fill_y data / (20'000 * 20'000)).

- <a id="_int_with_output"><strong>_int_with_output</strong></a>
	* If True, print statements are used. The default is True.
