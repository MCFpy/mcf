# Release Notes

## Version 0.4.2

### Bug fixes

- Minor bug fixes for ``ModifiedCausalForest`` (mainly redundant elements in return of prediction and analysis method deleted).

### New

#### General

- Output files for text, data and figures: So far, whenever a directory existed that has already been used for output, a new directory is created to avoid accidentally overwriting results. However, there is a new keyword for both the ``ModifiedCausalForest`` and the ``OptimalPolicy`` class:
  - [_int_output_no_new_dir](./mcf_api.md#_int_output_no_new_dir): Boolean. Do not create a new directory for outputs when the path already exists. Default is False.

#### Changes concerning the class ``ModifiedCausalForest``

- Mild improvements of output when categorical variables are involved.
- Data used for common support are saved in csv files.
- New keyword [_int_del_forest](./mcf_api.md#_int_del_forest): Boolean. Delete forests from instance. If True, less memory is needed, but the trained instance of the class cannot be reused when calling predict with the same instance again, i.e. the forest has to be retrained. Default is False.
- New keyword [_int_keep_w0](./mcf_api.md#_int_keep_w0): Boolean. Keep all zero weights when computing standard errors (slows down computation). Default is False.
- New keyword [p_ate_no_se_only](./mcf_api.md#p_ate_no_se_only): Boolean (or None). Computes only the ATE without standard errors. Default is False.
- New default value for [gen_iate_eff](./mcf_api.md#gen_iate_eff): The second round IATE estimation is no longer performed by default (i.e. the new default is False).
- New method ``blinder_iates``: Compute 'standard' IATEs as well as IATEs that are to a certain extent blinder than the standard ones. Available keywords:
  - blind_var_x_protected_name : List of strings (or None). Names of protected variables. Names that are explicitly denote as blind_var_x_unrestricted_name or as blind_var_x_policy_name and used to compute IATEs will be automatically added to this list. Default is None.
  - blind_var_x_policy_name : List of strings (or None). Names of decision variables. Default is None.
  - blind_var_x_unrestricted_name : List of strings (or None). Names of unrestricted variables. Default is None.
  - blind_weights_of_blind : Tuple of float (or None). Weights to compute weighted means of blinded and unblinded IATEs. Between 0 and 1. 1 implies all weight goes to fully blinded IATE. Default is None.
  - blind_obs_ref_data : Integer (or None), optional. Number of observations to be used for blinding. Runtime of programme is almost linear in this parameter. Default is 50.
  - blind_seed : Integer, optional. Seed for the random selection of the reference data. Default is 123456.



## Version 0.4.1

### Bug fixes
- Bug fix for AMGATE and Balanced GATE (BGATE)
- Minor bug fixes in Forest and  Optimal Policy module

### New
- We provide the ``change_log.py``, which provides extensive information on past changes and upcoming changes.
- We provide example data and example files on how to use ModifiedCausalForest
  and OptimalPolicy in various ways.
  - The following data files are provided. The names are self-explanatory. The number denotes the sample size, x are
      features, y is outcome, d is treatment, and ps denotes policy scores.:
      - data_x_1000.csv
      - data_x_4000.csv
      - data_x_ps_1_1000.csv
      - data_x_ps_2_1000.csv
      - data_y_d_x_1000.csv
      - data_y_d_x_4000.csv
  - The following example programmes are provided:
      - all_parameters_mcf.py, all_parameters_optpolicy.py
         Contain an explanation of all available parameters / keywords for the
         ModifiedCausalForest and OptimalPolicy classes.
      - min_parameters_mcf.py, min_parameters_optpolicy.py
         Contains the minimum specifications to run the methods of the
         ModifiedCausalForest and OptimalPolicy classes.
      - training_prediction_data_same_mcf.py
         One suggestion on how to proceed when data to train and fill the
         forest are the same as those used to compute the effects.
      - mcf_and_optpol_combined.py
         One suggestion on how to combine mcf and optimal policy estimation in
         a simple split sample approach.

## Version 0.4.0

Both the mcf module and the optimal policy module have undergone major revisions. The goal was to increase scalability and reduce internal complexity of the modules. The entire package now runs on Python 3.11, which is also recommended and tested. Note that all keywords changed compared to prior versions. Refer to the APIs for an updated list. For details on the updated worfklow, consult the respective tutorials.

### What's New

#### Changes concerning the class ``modified_causal_forest``

* Update in the feature selection algorithm. For details refer to [Walkthrough](https://mcfpy.github.io/mcf/#/mcf_walkthrough).
* Update in the common support estimation. For details refer to [Walkthrough](https://mcfpy.github.io/mcf/#/mcf_walkthrough).
* Updates related to GATE estimation:
- Wald tests are no longer provided,
- MGATEs are no longer estimated.
- AMGATEs will be conducted for the same heterogeneity variables as the GATEs.
- New parameter [p_iate_m_ate](https://mcfpy.github.io/mcf/#//mcf_api/p_iate_m_ate)  to compute difference of the IATEs and the ATE. The default is False.
* New parameter [p_iate_eff](https://mcfpy.github.io/mcf/#//mcf_api/p_iate_eff).
* Introduction of the the BGATEs. For details refer to [Walkthrough](https://mcfpy.github.io/mcf/#/mcf_walkthrough).
* Sample reductions for computational speed ups, need to be user-defined. Related options are removed from the mcf:
-  _INT_RED_SPLIT_SAMPLE,
- _INT_RED_SPLIT_SAMPLE_PRED_SHARE,
- _INT_SMALLER_SAMPLE,
- _INT_RED_TRAINING,
- _INT_RED_TRAINING_SHARE,
- _INT_RED_PREDICTION,
- _INT_RED_PREDICTION_SHARE,
- _INT_RED_LARGEST_GROUP_TRAIN,
- _INT_RED_LARGEST_GROUP_TRAIN_SHARE.
* Improved scalability by splitting training data into chunks and taking averages.
* Unified data concept to deal with common support and local centering.

### Name Changes and Default Updates
* All keywords are changed. Please refer to the [Python API](https://mcfpy.github.io/mcf/#/mcf_api_short) and the [extended Python API](https://mcfpy.github.io/mcf/#/mcf_api). The keywords will stay the same in future versions.


## Version 0.3.3

### What's New

* Now runs also on Python 3.10.x.
* Renaming of output: Marginal effects became Moderated effects.
* Speed and memory improvements
  Weight matrix computed in smaller chunks for large data
  There is also a parameter that comes along this change (which should usually
  not be changed by the user)
  _weight_as_sparse_splits  Default value is
  round(Rows of prediction data * rows of Fill_y data / (20'000 * 20'000))
* Additional and improved statistics for balancing tests.

### Bug fixes

* Correction of prognostic score nearest neighbour matching when local
centering was activated.

### Name Changes and Default Updates

* Name changes:
  -* m_share_min --> m_min_share
  -* m_share_max --> m_max_share
  * nw_kern_flag --> nw_kern
  * atet_flag --> atet
  * gatet_flag --> gatet
  * iate_flag --> iate
  * iate_se_flag --> iate_se
  -* iate_eff_flag --> iate_eff
  -* iate_cv_flag --> iate_cv
  * cond_var_flag --> cond_var
  * knn_flag --> knn
  * clean_data_flag --> clean_data

* Default values
  * alpha_reg_min = 0.05
  * alpha_reg_max = 0.15
  * If alpha_reg_grid = 1 (default): alpha = (alpha_reg_min+alpha_reg_ax)/2
  * m_share_min = 0.1
  * m_share_max = 0.6
  * m_grid = 1
  * number of variables used for splitting = share * total # of variable
  * If m_grid == 1: m_share = (m_share_min + m_share_max)/2
  * n_min_min=n_d**0.4/6; at least 4
  * n_min_max=sqrt(n_d)/6, at least ^4 where n_d denotes the number of observations in the smallest treatment arm
  * If n_min_grid == 1: n_min=(n_min_min+n_min_max)/2
  * n_min_treat = n_min_min+n_min_max)/2 / # of treatments / 4. Minimum is 2.

## Version 0.3.2

### What's New

* In estimation use cross-fitting to compute the IATEs. To enable cross-fitting set iate_cv to True. The default is False. The default number of folds is 5 and can be overwritten via the input argument iate_cv_folds. The estimates are stored in the  iate_cv_file.csv. Further information on estimation and descriptives are stored in the iate_cv_file.txt.
* Compare GATE(x) to GATE(x-1), where x is the current evaluation point and x-1 the previous one by setting GATE_MINUS_PREVIOUS to True. The default is False.
* Set n_min_treat to regulate the minimum number of observations in the treatment leaves.
* Experimental support for Dask. The default for multiprocessing is Ray. You may deploy Dask by setting _RAY_OR_DASK ='dask'. Note that with Dask the call of the programme needs to proteced by setting ``__name__ == '__main__'``



### Bug fixes
* Minor bug when GATEs were printed is fixed.
* Updated labels in sorted effects plots.

### Name Changes and Default Updates
* EFFIATE_FLAG = IATE_EFF_FLAG
* SMOOTH_GATES = GATES_SMOOTH
* SMOOTH_GATES_BANDWIDTH = GATES_SMOOTH_BANDWIDTH
* SMOOTH_GATES_NO_EVALUATION_POINTS  = GATES_SMOOTH_NO_EVALUATION_POINTS
* RELATIVE_TO_FIRST_GROUP_ONLY = POST_RELATIVE_TO_FIRST_GROUP_ONLY
* BIN_CORR_YES = POST_BIN_CORR_YES
* BIN_CORR_THRESHOLD = POST_BIN_CORR_THRESHOLD
* Increase in the default for sampling share
* New defaults for feature selection
  - fs_other_sample_share = 0.33
  - fs_rf_threshold = 0.0001
* Defaults for n_min_min increased to n**0.4/10, at least 3; -1: n**0.4/5 - where n is the number of observations in the smallest treatment arm.
* Number of parallel processes set to mp_parallel = 80% of logical cores.
* subsample_factor_eval = True, where True means 2 * subsample size used for tree.


## Version 0.3.1

### What's New

* New experimental feature: A new module is provided (optpolicy_with_mcf)
  that combines mcf estimations of IATEs with optimal policies (black-box and
  policy trees). It also provides out-of-sample evaluations of the
  allocations. For more details refer to Cox, Lechner, Bollens (2022) and
  user_evaluate_optpolicy_with_mcf.py.

### Bug fixes

* csv files for GATE tables can also deal with general treatment definitions
* _mp_with_ray no longer an input argument
* names_pot_iate is an additional return from the estimator. It is a 2-tuple with the list of potentially outcomes.
* [_return_iate_sp](./mcf_api.md#_return_iate_sp) is a new parameter to algorithm to predict and return effects despite [_with_output](./mcf_api.md#_with_output) being set to False.


## Version 0.3.0

### What's New

* The mcf supports an object-oriented interface: new class ModifiedCausalForest and methods (predict, train, and train_predict).
* Delivery of potential outcome estimates for which local centering is reversed by setting [l_centering_undo_iate](./mcf_api.md#l_centering_undo_iate)  to True; default is True.
* Readily available tables for GATEs, AMGATEs, and MGATEs. Genrated tables summarize all estimated causal effects. Tables are stored in respective folders.
* The optimal policy function is generalized to encompass also stochastic treatment allocations.

### Bug fixes

* Training and prediction are done in separate runs.
* Issue in optimal policy learning for unobserved treatment was resolved.

## Version 0.2.6

### Bug fixes

* Bug fix in general_purpose.py

## Version 0.2.5 (yanked)

### Bug fixes

* Bug fix in bootstrap of optimal policy module.

### What's New

* Change in output directory structure.
* Name change of file with predicted IATE (ends <foo>_IATE.csv)
* default value of l_centering_replication changed from False to True.
* More efficient estimation of IATE, referred to as EffIATE

## Version 0.2.4

### Bug fixes

* Bug fix for cases when outcome had no variation when splitting.

### What's New

* File with IATEs also contains indicator of specific cluster in k-means
  clustering.
* Option for guaranteed replicability of results. sklearn.ensemble.RandomForestRegressor does not necessarily replicable results (due to threading). A new keyword argument (l_centering_replication, default is False) is added. Setting this argument to True slows down local centering a but but removes that problem

## Version 0.2.3

### Bug fixes

* Missing information in init.py.

## Version 0.2.2

### Bug fixes

* Bug fix in plotting GATEs.

### What's New

* ATEs are  saved in csv file (same as data for figures and other effects).

## Version 0.2.1

### Bug fixes

* Bug fix in MGATE estimation, which led to program aborting.  

## Version 0.2.0

### Bug fixes

* Bug fix for policy trees under restrictions.
* Bug fix for GATE estimation (when weighting was used).

### What's New

* Main function changed from ``ModifiedCausalForest()``  to ``modified_causal_forest()``.
* Complete seeding of random number generator.
* Keyword modifications:
	* [stop_empty](./mcf_api.md#stop_empty) removed as parameter,
	* [descriptive_stats](./mcf_api.md#_descriptive_stats) becomes [_descriptive_stats](./mcf_api.md#_descriptive_stats),
	* [dpi](./mcf_api.md#_dpi) becomes [_dpi](./mcf_api.md#_dpi),
	* [fontsize](./mcf_api.md#_fontsize) becomes [_fontsize](./mcf_api.md#_fontsize),   
	* [mp_vim_type](./mcf_api.md#_mp_vim_type)
    becomes [_mp_vim_type](./mcf_api.md#_mp_vim_type),
    * [mp_weights_tree_batch](./mcf_api.md#_mp_weights_tree_batch)  becomes [_mp_weights_tree_batch](./mcf_api.md#_mp_weights_tree_batch),
    * [mp_weights_type](./mcf_api.md#_mp_weights_type)  becomes[_mp_weights_type](./mcf_api.md#_mp_weights_type),  
	* [mp_with_ray](./mcf_api.md#_mp_with_ray) becomes [_mp_with_ray](./mcf_api.md#_mp_with_ray),
	*  [no_filled_plot](./mcf_api.md#_no_filled_plot)
    becomes [_no_filled_plot](./mcf_api.md#_no_filled_plot),
	* [show_plots](./mcf_api.md#_show_plots) becomes [_show_plots](./mcf_api.md#_show_plots),  
	* [verbose](./mcf_api.md#_verbose) becomes [_verbose](./mcf_api.md#_verbose),
	* [weight_as_sparse](./mcf_api.md#_weight_as_sparse) becomes [_weight_as_sparse](./mcf_api.md#_weight_as_sparse),
	* [support_adjust_limits](./mcf_api.md#support_adjust_limits) new keyword for common support.
* Experimental version of continuous treatment. Newly introduced keywords here
	* [d_type](./mcf_api.md#d_type),
	* [ct_grid_nn](./mcf_api.md#ct_grid_nn),
	* [ct_grid_w](./mcf_api.md#ct_grid_w),
	* [ct_grid_dr](./mcf_api.md#ct_grid_dr).  
* The optimal policy function contains new rules based on 'black box' approaches, i.e., using the potential outcomes directly to obtain optimal allocations.
* The optimal policy function allows to describe allocations with respect to other policy variables than the ones used for determining the allocation.
* Plots
	* improved plots,
	* new overlapping plots for common support analysis.


## Version 0.1.4

### Bug fixes

- Bug fix for predicting from previously trained and saved forests.
- Bug fix in [mcf_init_function](./mcf_api.md#mcf_init_function) when there are missing values.

### What's New

- [_mp_ray_shutdown](./mcf_api.md#_mp_ray_shutdown) new defaults. If object size is smaller 100,000, the default is False and else True.

## Version 0.1.3

### Bug fixes

- Minor bug fixes, which led to unstable performance.

### What's New

- [subsample_factor](./mcf_api.md#subsample_factor_eval) is split into [subsample_factor_eval](./mcf_api.md#subsample_factor_eval) and [subsample_factor_forest](./mcf_api.md#subsample_factor_forest).
- New default value for [stop_empty](./mcf_api.md#stop_empty).
- Optimal policy module computes the policy tree also sequentially. For this purpose, the ``optpoltree`` API has changed slightly. Renamed input arguments are
  - [ft_yes](./opt-pol_api.md#ft_yes),
  - [ft_depth](./opt-pol_api.md#ft_depth),
  - [ft_min_leaf_size](./opt-pol_api.md#ft_min_leaf_size),
  - [ft_no_of_evalupoints](./opt-pol_api.md#ft_no_of_evalupoints)
  - [ft_yes](./opt-pol_api.md#ft_yes),
- the new input arguments for the sequential tree are:
	- [st_yes](./opt-pol_api.md#st_yes),
	- [st_depth](./opt-pol_api.md#st_depth),
	- [st_min_leaf_size](./opt-pol_api.md#st_min_leaf_size).


## Version 0.1.2

### Bug fixes

- Common support with very few observations is turned off.
- Minor fix of MSE computation for multiple treatments.  

### What's New  
- New default values for  
	- [alpha_reg_grid](./mcf_api.md#alpha_reg_grid),
	- [alpha_reg_max](./mcf_api.md#alpha_reg_max),
	- [alpha_reg_min](./mcf_api.md#alpha_reg_min),
	- [knn_flag](./mcf_api.md#knn_flag),
	- [l_centering](./mcf_api.md#l_centering),
	- [mp_parallel](./mcf_api.md#mp_parallel)
	- [p_diff_penalty](./mcf_api.md#p_diff_penalty),
	- [random_thresholds](./mcf_api.md#random_thresholds),
	- [se_boot_ate](./mcf_api.md#se_boot_ate),
	- [se_boot_gate](./mcf_api.md#se_boot_gate),
	- [se_boot_iate](./mcf_api.md#se_boot_iate),
	- [stop_empty](./mcf_api.md#stop_empty).
- Consistent use of a new random number generator.
- Ray is initialized once.
- Ray can be fine-tuned via
	- [_mp_ray_del](./mcf_api.md#_mp_ray_del),
	- [_mp_ray_shutdown](./mcf_api.md#_mp_ray_shutdown),
	- [_mp_ray_shutdown](./mcf_api.md#_mp_ray_shutdown),
	- [mp_ray_objstore_multiplier](./mcf_api.md#_mp_ray_objstore_multiplier) becomes [_mp_ray_objstore_multiplier](./mcf_api.md#_mp_ray_objstore_multiplier).
- New options to deal with larger data sets:
	- [reduce_split_sample](./mcf_api.md#reduce_split_sample): split sample in a part used for estimation and predicting the effects for given x; large prediction sample may increase running time.
	- [reduce_training](./mcf_api.md#reduce_training): take a random sample from training data.
	- [reduce_prediction](./mcf_api.md#reduce_prediction): take a random sample from prediction data.
	- [reduce_largest_group_train](./mcf_api.md#reduce_largest_group_train): reduce the largest group in the training data; this should be less costly in terms of precision than taking random samples.
- Optional IATEs via [iate_flag](./mcf_api.md#iate_flag) and optional standard errors via [iate_se_flag](./mcf_api.md#iate_se_flag).
- ``ModifiedCausalForest()`` now also returns potential outcomes and their variances.
- [mp_with_ray](./opt-pol_api.md#mp_with_ray) is a new input argument to ``‌optpoltree()`` ;  Ray can be used for multiprocessing when calling ``‌optpoltree()``.
- Block-bootstrap on $w_i*y_i$ is the new clustered standard errors default. This is slower but likely to be more accurate  than the aggregation within-clusters deployed before.

## Version 0.1.1

### Bug fixes
- Minor bug fixes concerning [_with_output](./mcf_api.md#_with_output), [_smaller_sample](./mcf_api.md#_smaller_sample), (A,AM)GATE/IATE-ATE plots, and the sampling weights.

### What's New
- Optional tree-specific subsampling for evaluation sample (subsample variables got new names).
- k-Means cluster indicator for the IATEs saved in file with IATE predictions.
- Evaluation points of GATE figures are included in the output csv-file.
- Exception raised if choice based sampling is activated and there is no treatment information in predictions file.
- New defaults for [random_thresholds](./mcf_api.md#random_thresholds); by default the value is set to 20 percent of the square-root of the number of training observations.
- Stabilizing ``ray`` by deleting references to object store and tasks
- The function ``ModifiedCausalForest()`` returns now ATE, standard error (SE) of the ATE, GATE, SE of the GATE, IATE, SE of the IATE, and the name of the file with the predictions.


## Version 0.1.0

### Bug fixes
 - Bug fix for dealing with missings.
 - Bug fixes for problems computing treatment effects for treatment populations.
 - Bug fixes for the use of panel data and clustering.

### What's New
- [post_kmeans_no_of_groups](./mcf_api.md#post_kmeans_no_of_groups) can now be a list or tuple with multiple values for the number of clusters; the optimal value is chosen through silhouette analysis.
- Detection of numerical variables added; raises an exception for non-numerical inputs.
- All variables used are shown in initial treatment specific statistics to detect common support issues.
- Improved statistics for common support analysis.

### Experimental
- Optimal Policy Tool building policy trees included bases on estimated IATEs (allowing implicitly for constraints and programme costs).
