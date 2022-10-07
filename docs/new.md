# Release Updates

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
* [_return_iate_sp](./core_6.md#_return_iate_sp) is a new parameter to algorithm to predict and return effects despite [_with_output](./core_6.md#_with_output) being set to False.


## Version 0.3.0

### What's New

* The mcf supports an object-oriented interface: new class ModifiedCausalForest and methods (predict, train, and train_predict).
* Delivery of potential outcome estimates for which local centering is reversed by setting [l_centering_undo_iate](./core_6.md#l_centering_undo_iate)  to True; default is True.
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
	* [stop_empty](./core_6.md#stop_empty) removed as parameter,
	* [descriptive_stats](./core_6.md#_descriptive_stats) becomes [_descriptive_stats](./core_6.md#_descriptive_stats),
	* [dpi](./core_6.md#_dpi) becomes [_dpi](./core_6.md#_dpi),
	* [fontsize](./core_6.md#_fontsize) becomes [_fontsize](./core_6.md#_fontsize),   
	* [mp_vim_type](./core_6.md#_mp_vim_type)
    becomes [_mp_vim_type](./core_6.md#_mp_vim_type),
    * [mp_weights_tree_batch](./core_6.md#_mp_weights_tree_batch)  becomes [_mp_weights_tree_batch](./core_6.md#_mp_weights_tree_batch),
    * [mp_weights_type](./core_6.md#_mp_weights_type)  becomes[_mp_weights_type](./core_6.md#_mp_weights_type),  
	* [mp_with_ray](./core_6.md#_mp_with_ray) becomes [_mp_with_ray](./core_6.md#_mp_with_ray),
	*  [no_filled_plot](./core_6.md#_no_filled_plot)
    becomes [_no_filled_plot](./core_6.md#_no_filled_plot),
	* [show_plots](./core_6.md#_show_plots) becomes [_show_plots](./core_6.md#_show_plots),  
	* [verbose](./core_6.md#_verbose) becomes [_verbose](./core_6.md#_verbose),
	* [weight_as_sparse](./core_6.md#_weight_as_sparse) becomes [_weight_as_sparse](./core_6.md#_weight_as_sparse),
	* [support_adjust_limits](./core_6.md#support_adjust_limits) new keyword for common support.
* Experimental version of continuous treatment. Newly introduced keywords here
	* [d_type](./core_6.md#d_type),
	* [ct_grid_nn](./core_6.md#ct_grid_nn),
	* [ct_grid_w](./core_6.md#ct_grid_w),
	* [ct_grid_dr](./core_6.md#ct_grid_dr).  
* The optimal policy function contains new rules based on 'black box' approaches, i.e., using the potential outcomes directly to obtain optimal allocations.
* The optimal policy function allows to describe allocations with respect to other policy variables than the ones used for determining the allocation.
* Plots
	* improved plots,
	* new overlapping plots for common support analysis.


## Version 0.1.4

### Bug fixes

- Bug fix for predicting from previously trained and saved forests.
- Bug fix in [mcf_init_function](./core_6.md#mcf_init_function) when there are missing values.

### What's New

- [_mp_ray_shutdown](./core_6.md#_mp_ray_shutdown) new defaults. If object size is smaller 100,000, the default is False and else True.

## Version 0.1.3

### Bug fixes

- Minor bug fixes, which led to unstable performance.

### What's New

- [subsample_factor](./core_6.md#subsample_factor_eval) is split into [subsample_factor_eval](./core_6.md#subsample_factor_eval) and [subsample_factor_forest](./core_6.md#subsample_factor_forest).
- New default value for [stop_empty](./core_6.md#stop_empty).
- Optimal policy module computes the policy tree also sequentially. For this purpose, the ``optpoltree`` API has changed slightly. Renamed input arguments are
  - [ft_yes](./opt_pol_1.md#ft_yes),
  - [ft_depth](./opt_pol_1.md#ft_depth),
  - [ft_min_leaf_size](./opt_pol_1.md#ft_min_leaf_size),
  - [ft_no_of_evalupoints](./opt_pol_1.md#ft_no_of_evalupoints)
  - [ft_yes](./opt_pol_1.md#ft_yes),
- the new input arguments for the sequential tree are:
	- [st_yes](./opt_pol_1.md#st_yes),
	- [st_depth](./opt_pol_1.md#st_depth),
	- [st_min_leaf_size](./opt_pol_1.md#st_min_leaf_size).


## Version 0.1.2

### Bug fixes

- Common support with very few observations is turned off.
- Minor fix of MSE computation for multiple treatments.  

### What's New  
- New default values for  
	- [alpha_reg_grid](./core_6.md#alpha_reg_grid),
	- [alpha_reg_max](./core_6.md#alpha_reg_max),
	- [alpha_reg_min](./core_6.md#alpha_reg_min),
	- [knn_flag](./core_6.md#knn_flag),
	- [l_centering](./core_6.md#l_centering),
	- [mp_parallel](./core_6.md#mp_parallel)
	- [p_diff_penalty](./core_6.md#p_diff_penalty),
	- [random_thresholds](./core_6.md#random_thresholds),
	- [se_boot_ate](./core_6.md#se_boot_ate),
	- [se_boot_gate](./core_6.md#se_boot_gate),
	- [se_boot_iate](./core_6.md#se_boot_iate),
	- [stop_empty](./core_6.md#stop_empty).
- Consistent use of a new random number generator.
- Ray is initialized once.
- Ray can be fine-tuned via
	- [_mp_ray_del](./core_6.md#_mp_ray_del),
	- [_mp_ray_shutdown](./core_6.md#_mp_ray_shutdown),
	- [_mp_ray_shutdown](./core_6.md#_mp_ray_shutdown),
	- [mp_ray_objstore_multiplier](./core_6.md#_mp_ray_objstore_multiplier) becomes [_mp_ray_objstore_multiplier](./core_6.md#_mp_ray_objstore_multiplier).
- New options to deal with larger data sets:
	- [reduce_split_sample](./core_6.md#reduce_split_sample): split sample in a part used for estimation and predicting the effects for given x; large prediction sample may increase running time.
	- [reduce_training](./core_6.md#reduce_training): take a random sample from training data.
	- [reduce_prediction](./core_6.md#reduce_prediction): take a random sample from prediction data.
	- [reduce_largest_group_train](./core_6.md#reduce_largest_group_train): reduce the largest group in the training data; this should be less costly in terms of precision than taking random samples.
- Optional IATEs via [iate_flag](./core_6.md#iate_flag) and optional standard errors via [iate_se_flag](./core_6.md#iate_se_flag).
- ``ModifiedCausalForest()`` now also returns potential outcomes and their variances.
- [mp_with_ray](./opt_pol_1.md#mp_with_ray) is a new input argument to ``‌optpoltree()`` ;  Ray can be used for multiprocessing when calling ``‌optpoltree()``.
- Block-bootstrap on $w_i*y_i$ is the new clustered standard errors default. This is slower but likely to be more accurate  than the aggregation within-clusters deployed before.

## Version 0.1.1

### Bug fixes
- Minor bug fixes concerning [_with_output](./core_6.md#_with_output), [_smaller_sample](./core_6.md#_smaller_sample), (A,AM)GATE/IATE-ATE plots, and the sampling weights.

### What's New
- Optional tree-specific subsampling for evaluation sample (subsample variables got new names).
- k-Means cluster indicator for the IATEs saved in file with IATE predictions.
- Evaluation points of GATE figures are included in the output csv-file.
- Exception raised if choice based sampling is activated and there is no treatment information in predictions file.
- New defaults for [random_thresholds](./core_6.md#random_thresholds); by default the value is set to 20 percent of the square-root of the number of training observations.
- Stabilizing ``ray`` by deleting references to object store and tasks
- The function ``ModifiedCausalForest()`` returns now ATE, standard error (SE) of the ATE, GATE, SE of the GATE, IATE, SE of the IATE, and the name of the file with the predictions.


## Version 0.1.0

### Bug fixes
 - Bug fix for dealing with missings.
 - Bug fixes for problems computing treatment effects for treatment populations.
 - Bug fixes for the use of panel data and clustering.

### What's New
- [post_kmeans_no_of_groups](./core_6.md#post_kmeans_no_of_groups) can now be a list or tuple with multiple values for the number of clusters; the optimal value is chosen through silhouette analysis.
- Detection of numerical variables added; raises an exception for non-numerical inputs.
- All variables used are shown in initial treatment specific statistics to detect common support issues.
- Improved statistics for common support analysis.

### Experimental
- Optimal Policy Tool building policy trees included bases on estimated IATEs (allowing implicitly for constraints and programme costs).
