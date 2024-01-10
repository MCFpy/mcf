Changelog
=======================
.. 
    Conventions:

    1. Add a horizontal rule ----- before adding a new entry
    2. Nest parameters of functions/methods in single backticks, e.g. `foo`
    3. Cross-references are possible for classes, and their methods and properties:
        - Refer to classes using :py:class:`~module.ClassName`, e.g. :py:class:`~mcf_functions.ModifiedCausalForest`
        - Refer to methods using :py:meth:`~module.ClassName.method_name`, e.g. :py:meth:`~mcf_functions.ModifiedCausalForest.train` 
        - Refer to class properties using :py:attr:`~module.ClassName.property_name`, e.g. :py:attr:`~mcf_functions.ModifiedCausalForest.blind_dict`
    4. Nested lists: You need to separate the lists with a blank line. Otherwise, the parent will be displayed as bold.

        - Wrong (will be bold):
            - A
            - B 

        - Right:

            - A
            - B

:py:class:`~mcf_mini.ModifiedCausalForest` 
:py:class:`~optpol_mini.OptimalPolicy` 

:py:meth:`~mcf_mini.ModifiedCausalForest.train`
:py:meth:`~optpol_mini.OptimalPolicy.solve`

:py:attr:`~mcf_mini.ModifiedCausalForest.blind_dict`
:py:attr:`~optpol_mini.OptimalPolicy.dc_dict`

Version 0.4.3
-------------

Changes concerning the class :py:class:`~mcf_mini.ModifiedCausalForest`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bug fixes
+++++++++

- Minor bug fixes:

    - Weight computation (turned off and sparse weight matrix)
    - KeyError in Gate estimation
    - Corrected sample split when using feature selection

New
+++

- Leaf size adjustments:

  Sometimes, the mcf leads to fairly big leaves due to insufficient observations in each treatment arm. The following changes in default settings and minor code corrections have been implemented. They somewhat reduce leaf sizes, but necessarily lead to more cases, where the data used to populate the leaves will have to ignore more leaves as they cannot be populated with outcomes from all treatment arms.

  In this case, if the problem can be solved be redoing the last split (i.e. using the parent leave instead of the final child leaves), then these two leaves are merged.

  If this does not solve the problem (either because one of the children is split further, or because there are still treatment arms missing in the merged leave), then this leave is not used in the computation of the weights.

  - Default for `cf_n_min_treat` changed to `(n_min_min + n_min_max) / 2 / # of treatments / 10`. Minimum is 1.
  - Defaults for `cf_n_min_min` and `cf_n_min_max` changed to:
    - `n_min_min = round(max((n_d_subsam**0.4) / 10, 1.5) * # of treatments)`
    - `n_min_max = round(max((n_d_subsam**0.5) / 10, 2) * # of treatments)`
  - Default values for tuning parameters of `mcf` are taken into account when observations are used only for feature selection, common support, or local centering.

- Improved computational performance:

  - Speed-up for categorical (unordered) variables due to memorization. This requires some additional memory, but the gains could be substantial.
  - Improved internal computation and storage of estimated forests lead to speed and precision gains (instead of using lists of lists, we now use a list of dictionaries of optimized numpy arrays to save the trees). Since the precision of the new method is higher (by at the same time needing less RAM), this might lead to smallish changes in the results.

- **Experimental**: The method :py:meth:`~mcf_mini.ModifiedCausalForest.sensitivity` has been added. It contains some simulation-based tools to check how well the mcf works in removing selection bias and how sensitive the results are with respect to potentially missing confounding covariates (i.e., those related to treatment and potential outcome) added in the future.

  - Note: This section is currently experimental and thus not yet fully documented and tested. A paper by Armendariz-Pacheco, Frischknecht, Lechner, and Mareckova (2024) will discuss and investigate the different methods in detail. So far, please note that all methods are simulation based.

  - The sensitivity checks consist of the following steps:

    1. Estimate all treatment probabilities.

    2. Remove all observations from treatment states other than one (largest treatment or user-determined).

    3. Use estimated probabilities to simulate treated observations, respecting the original treatment shares (pseudo-treatments).

    4. Estimate the effects of pseudo-treatments. The true effects are known to be zero, so the deviation from 0 is used as a measure of result sensitivity.

    Steps 3 and 4 may be repeated, and results averaged to reduce simulation noise.

  - In this experimental version, the method depends on the following new keywords:

    - `sens_amgate`: Boolean (or None), optional. Compute AMGATEs for sensitivity analysis. Default is False.
    - `sens_bgate`: Boolean (or None), optional. Compute BGATEs for sensitivity analysis. Default is False.
    - `sens_gate`: Boolean (or None), optional. Compute GATEs for sensitivity analysis. Default is False.
    - `sens_iate`: Boolean (or None), optional. Compute IATEs for sensitivity analysis. Default is False.
    - `sens_iate_se`: Boolean (or None), optional. Compute standard errors of IATEs for sensitivity analysis. Default is False.
    - `sens_scenarios`: List or tuple of strings, optional. Different scenarios considered. Default is ('basic',). 'basic': Use estimated treatment probabilities for simulations. No confounding.
    - `sens_cv_k`: Integer (or None), optional. Data to be used for any cross-validation: Number of folds in cross-validation. Default (or None) is 5.
    - `sens_replications`: Integer (or None), optional. Number of replications for simulating placebo treatments. Default is 2.
    - `sens_reference_population`: Integer or float (or None). Defines the treatment status of the reference population used by the sensitivity analysis. Default is to use the treatment with most observed observations.

Changes concerning the class :py:class:`~optpol_mini.OptimalPolicy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- No changes.

-----

Version 0.4.2
-------------

Bug fixes
~~~~~~~~~

- Minor bug fixes for :py:class:`~mcf_mini.ModifiedCausalForest` (mainly redundant elements in return of prediction and analysis method deleted).

New
~~~

General
+++++++

- Output files for text, data and figures: So far, whenever a directory existed that has already been used for output, a new directory is created to avoid accidentally overwriting results. However, there is a new keyword for both the :py:class:`~mcf_mini.ModifiedCausalForest` and the :py:class:`~optpol_mini.OptimalPolicy` class:

    - `_int_output_no_new_dir`: Boolean. Do not create a new directory for outputs when the path already exists. Default is False.

Changes concerning the class :py:class:`~mcf_mini.ModifiedCausalForest`
+++++++++++++++++++++++++++++++++++++++++++++++++++++

- Mild improvements of output when categorical variables are involved.
- Data used for common support are saved in csv files.
- New keyword `_int_del_forest`: Boolean. Delete forests from instance. If True, less memory is needed, but the trained instance of the class cannot be reused when calling predict with the same instance again, i.e. the forest has to be retrained. Default is False.
- New keyword `_int_keep_w0`: Boolean. Keep all zero weights when computing standard errors (slows down computation). Default is False.
- New keyword `p_ate_no_se_only`: Boolean (or None). Computes only the ATE without standard errors. Default is False.
- New default value for `gen_iate_eff`: The second round IATE estimation is no longer performed by default (i.e. the new default is False).
- There is a new experimental features to both the mcf estimation (of IATEs) as well as the optimal policy module. It allows to partially blind the decision with respect to certain variables. The accompanying discussion paper by Nora Bearth, Fabian Muny, Michael Lechner, and Jana Marackova ('Partially Blind Optimal Policy Analysis') is currently written. If you desire more information, please email one of the authors. 

        - New method :py:meth:`~mcf_mini.ModifiedCausalForest.blinder_iates`: Compute 'standard' IATEs as well as IATEs that are to a certain extent blinder than the standard ones. Available keywords:

            - `blind_var_x_protected_name` : List of strings (or None). Names of protected variables. Names that are explicitly denote as blind_var_x_unrestricted_name or as blind_var_x_policy_name and used to compute IATEs will be automatically added to this list. Default is None.
            - `blind_var_x_policy_name` : List of strings (or None). Names of decision variables. Default is None.
            - `blind_var_x_unrestricted_name` : List of strings (or None). Names of unrestricted variables. Default is None.
            - `blind_weights_of_blind` : Tuple of float (or None). Weights to compute weighted means of blinded and unblinded IATEs. Between 0 and 1. 1 implies all weight goes to fully blinded IATE. Default is None.
            - `blind_obs_ref_data` : Integer (or None), optional. Number of observations to be used for blinding. Runtime of programme is almost linear in this parameter. Default is 50.
            - `blind_seed` : Integer, optional. Seed for the random selection of the reference data. Default is 123456.

Changes concerning the class :py:class:`~optpol_mini.OptimalPolicy`
++++++++++++++++++++++++++++++++++++++++++++++

- General keyword change in the :py:class:`~optpol_mini.OptimalPolicy` class. All keywords that started with `int_` now start with `_int_` (in order to use the same conventions as in the :py:class:`~mcf_mini.ModifiedCausalForest` class).

- New keywords:

    - `_pt_select_values_cat`: Approximation method for larger categorical variables. Since we search among optimal trees, for categorical variables variables we need to check for all possible combinations of the different values that lead to binary splits. This number could indeed be huge. Therefore, we compare only pt_no_of_evalupoints * 2 different combinations. Method 1 (pt_select_values_cat == True) does this by randomly drawing values from the particular categorical variable and forming groups only using those values. Method 2 (pt_select_values_cat==False) sorts the values of the categorical variables according to a values of the policy score as one would do for a standard random forest. If this set is still too large, a random sample of the entailed combinations is drawn.  Method 1 is only available for the method 'policy tree eff'. The default is False.
    - `_pt_enforce_restriction`: Boolean (or None). Enforces the imposed restriction (to some extent) during the computation of the policy tree. This can be very time consuming. Default is True.
    - `_pt_eva_cat_mult`: Integer (or None). Changes the number of the evaluation points (pt_no_of_evalupoints) for the unordered (categorical) variables to: pt_eva_cat_mult * pt_no_of_evalupoints (available only for the method 'policy tree eff'). Default is 1.
    - `_gen_variable_importance`: Boolean. Compute variable importance statistics based on random forest classifiers. Default is False.
    - `_var_vi_x_name`: List of strings or None, optional. Names of variables for which variable importance is computed. Default is None.
    - `_var_vi_to_dummy_name`: List of strings or None, optional. Names of variables for which variable importance is computed. These variables will be broken up into dummies. Default is None.

The optimal policy module currently has three methods (:py:meth:`~optpol_mini.OptimalPolicy.best_policy_score`, :py:meth:`~optpol_mini.OptimalPolicy.policy tree`, :py:meth:`~optpol_mini.OptimalPolicypolicy tree eff`):

- :py:meth:`~optpol_mini.OptimalPolicypolicy tree eff` (NEW in 0.4.2) is very similar to 'policy tree'. It uses different approximation rules and uses slightly different coding.  In many cases it should be faster than 'policy tree'.  Default (or None) is 'best_policy_score'.
- :py:meth:`~optpol_mini.OptimalPolicy.best_policy_score` conducts Black-Box allocations, which are obtained by using the scores directly (potentially subject to restrictions). When the Black-Box allocations are used for allocation of data not used for training, the respective scores must be available.
- The implemented :py:meth:`~optpol_mini.OptimalPolicy.policy tree`'s are optimal trees, i.e. all possible trees are checked if they lead to a better performance. If restrictions are specified, then this is incorporated into treatment specific cost parameters. Many ideas of the implementation follow Zhou, Athey, Wager (2022). If the provided policy scores fulfil their conditions (i.e., they use a doubly robust double machine learning like score), then they also provide attractive theoretical properties.

- New method :py:meth:`~optpol_mini.OptimalPolicy.evaluate_multiple`: Evaluate several allocations simultaneously.  Parameters:

    - `allocations_dic` : Dictionary. Contains DataFrame's with specific allocations.
    - `data_df` : DataFrame. Data with the relevant information about potential outcomes which will be used to evaluate the allocations.

-----

Version 0.4.1
-------------

Bug fixes
~~~~~~~~~

- Bug fix for AMGATE and Balanced GATE (BGATE)
- Minor bug fixes in Forest and Optimal Policy module

New
~~~

- We provide the change_log.py script, which provides extensive information on past changes and upcoming changes.
- We provide example data and example files on how to use :py:class:`~mcf_mini.ModifiedCausalForest` and :py:class:`~optpol_mini.OptimalPolicy` in various ways.

    - The following data files are provided. The names are self-explanatory. The number denotes the sample size, x are features, y is outcome, d is treatment, and ps denotes policy scores.:

        - data_x_1000.csv
        - data_x_4000.csv
        - data_x_ps_1_1000.csv
        - data_x_ps_2_1000.csv
        - data_y_d_x_1000.csv
        - data_y_d_x_4000.csv

    - The following example programmes are provided:

        - all_parameters_mcf.py, all_parameters_optpolicy.py: Contains an explanation of all available parameters / keywords for the :py:class:`~mcf_mini.ModifiedCausalForest` and :py:class:`~optpol_mini.OptimalPolicy` classes.
        - min_parameters_mcf.py, min_parameters_optpolicy.py: Contains the minimum specifications to run the methods of the :py:class:`~mcf_mini.ModifiedCausalForest` and :py:class:`~optpol_mini.OptimalPolicy` classes.
        - training_prediction_data_same_mcf.py: One suggestion on how to proceed when data to train and fill the forest are the same as those used to compute the effects.
        - mcf_and_optpol_combined.py: One suggestion on how to combine mcf and optimal policy estimation in a simple split sample approach.

-----

Version 0.4.0
-------------

Both the mcf module and the optimal policy module have undergone major revisions. The goal was to increase scalability and reduce internal complexity of the modules. The entire package now runs on Python 3.11, which is also recommended and tested. Note that all keywords changed compared to prior versions. Refer to the APIs for an updated list. For details on the updated worfklow, consult the respective tutorials.

What's New
~~~~~~~~~~

Changes concerning the class :py:class:`~mcf_mini.ModifiedCausalForest`:
++++++++++++++++++++++++++++++++++++++++++++++++++++++

- Update in the feature selection algorithm.
- Update in the common support estimation.
- Updates related to GATE estimation:
  - Wald tests are no longer provided,
  - MGATEs are no longer estimated.
  - AMGATEs will be conducted for the same heterogeneity variables as the GATEs.
  - New parameter `p_iate_m_ate` to compute difference of the IATEs and the ATE. The default is False.
- New parameter `p_iate_eff`.
- Introduction of the BGATEs.
- Sample reductions for computational speed ups, need to be user-defined. Related options are removed from the mcf:

    - `_int_red_split_sample`
    - `_int_red_split_sample_pred_share`
    - `_int_smaller_sample`
    - `_int_red_training`
    - `_int_red_training_share`
    - `_int_red_prediction`
    - `_int_red_prediction_share`
    - `_int_red_largest_group_train`
    - `_int_red_largest_group_train_share`

- Improved scalability by splitting training data into chunks and taking averages.
- Unified data concept to deal with common support and local centering.

Name Changes and Default Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- All keywords are changed. Please refer to the :doc:`python_api`.

-----

Version 0.3.3
-------------

What's New
~~~~~~~~~~

- Now runs also on Python 3.10.x.
- Renaming of output: Marginal effects became Moderated effects.
- Speed and memory improvements:

    - Weight matrix computed in smaller chunks for large data
    - There is also a parameter that comes along this change (which should usually not be changed by the user)
    - `_weight_as_sparse_splits`  Default value is round(Rows of prediction data * rows of Fill_y data / (20'000 * 20'000))
    
- Additional and improved statistics for balancing tests.

Bug fixes
~~~~~~~~~

- Correction of prognostic score nearest neighbour matching when local centering was activated.

Name Changes and Default Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Name changes:

    - `m_share_min` --> `m_min_share`
    - `m_share_max` --> `m_max_share`
    - `nw_kern_flag` --> `nw_kern`
    - `atet_flag` --> `atet`
    - `gatet_flag` --> `gatet`
    - `iate_flag` --> `iate`
    - `iate_se_flag` --> `iate_se`
    - `iate_eff_flag` --> `iate_eff`
    - `iate_cv_flag` --> `iate_cv`
    - `cond_var_flag` --> `cond_var`
    - `knn_flag` --> `knn`
    - `clean_data_flag` --> `clean_data`

- Default values

    - `alpha_reg_min` = 0.05
    - `alpha_reg_max` = 0.15
    - If `alpha_reg_grid` = 1 (default): `alpha` = (`alpha_reg_min` + `alpha_reg_ax`)/2
    - `m_share_min` = 0.1
    - `m_share_max` = 0.6
    - `m_grid` = 1
    - number of variables used for splitting = share * total # of variable
    - If `m_grid` == 1: `m_share` = (`m_share_min` + `m_share_max`)/2
    - `n_min_min` = `n_d` ** 0.4/6; at least 4
    - `n_min_max` = sqrt(`n_d`)/6, at least ^4 where n_d denotes the number of observations in the smallest treatment arm
    - If `n_min_grid` == 1: `n_min`=(`n_min_min` + `n_min_max`)/2
    - `n_min_treat` = `n_min_min` + `n_min_max`)/2 / # of treatments / 4. Minimum is 2.

-----

Version 0.3.2
-------------

What's New
~~~~~~~~~~

- In estimation use cross-fitting to compute the IATEs. To enable cross-fitting set iate_cv to True. The default is False. The default number of folds is 5 and can be overwritten via the input argument iate_cv_folds. The estimates are stored in the  iate_cv_file.csv. Further information on estimation and descriptives are stored in the iate_cv_file.txt.
- Compare GATE(x) to GATE(x-1), where x is the current evaluation point and x-1 the previous one by setting GATE_MINUS_PREVIOUS to True. The default is False.
- Set n_min_treat to regulate the minimum number of observations in the treatment leaves.
- Experimental support for Dask. The default for multiprocessing is Ray. You may deploy Dask by setting _RAY_OR_DASK ='dask'. Note that with Dask the call of the programme needs to proteced by setting ``__name__ == '__main__'``

Bug fixes
~~~~~~~~~

- Minor bug when GATEs were printed is fixed.
- Updated labels in sorted effects plots.

Name Changes and Default Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `effiate_flag` = `iate_eff_flag`
- `smooth_gates` = `gates_smooth`
- `smooth_gates_bandwidth` = `gates_smooth_bandwidth`
- `smooth_gates_no_evaluation_points` = `gates_smooth_no_evaluation_points`
- `relative_to_first_group_only` = `post_relative_to_first_group_only`
- `bin_corr_yes` = `post_bin_corr_yes`
- `bin_corr_threshold` = `post_bin_corr_threshold`
- Increase in the default for sampling share
- New defaults for feature selection
  - `fs_other_sample_share` = 0.33
  - `fs_rf_threshold` = 0.0001
- Defaults for `n_min_min` increased to n**0.4/10, at least 3; -1: n**0.4/5 - where n is the number of observations in the smallest treatment arm.
- Number of parallel processes set to `mp_parallel` = 80% of logical cores.
- `subsample_factor_eval` = True, where True means 2 * subsample size used for tree.

Version 0.3.1
-------------

What's New
~~~~~~~~~~

- New experimental feature: A new module is provided (optpolicy_with_mcf) that combines mcf estimations of IATEs with optimal policies (black-box and policy trees). It also provides out-of-sample evaluations of the allocations. For more details refer to Cox, Lechner, Bollens (2022) and user_evaluate_optpolicy_with_mcf.py.

Bug fixes
~~~~~~~~~

- csv files for GATE tables can also deal with general treatment definitions
- `_mp_with_ray` no longer an input argument
- names_pot_iate is an additional return from the estimator. It is a 2-tuple with the list of potentially outcomes.
- `return_iate_sp` is a new parameter to algorithm to predict and return effects despite `with_output` being set to False.

-----

Version 0.3.0
-------------

What's New
~~~~~~~~~~

- The mcf supports an object-oriented interface: new class :py:class:`~mcf_mini.ModifiedCausalForest` and methods (:py:meth:`~mcf_mini.ModifiedCausalForest.predict`, :py:meth:`~mcf_mini.ModifiedCausalForest.train` and :py:meth:`~mcf_mini.ModifiedCausalForest.train_predict`).
- Delivery of potential outcome estimates for which local centering is reversed by setting `l_centering_undo_iate` to True; default is True.
- Readily available tables for GATEs, AMGATEs, and MGATEs. Generated tables summarize all estimated causal effects. Tables are stored in respective folders.
- The optimal policy function is generalized to encompass also stochastic treatment allocations.

Bug fixes
~~~~~~~~~

- Training and prediction are done in separate runs.
- Issue in optimal policy learning for unobserved treatment was resolved.

-----

Version 0.2.6
-------------

Bug fixes
~~~~~~~~~

- Bug fix in general_purpose.py

-----

Version 0.2.5 (yanked)
----------------------

Bug fixes
~~~~~~~~~

- Bug fix in bootstrap of optimal policy module.

What's New
~~~~~~~~~~

- Change in output directory structure.
- Name change of file with predicted IATE (ends <foo>_IATE.csv)
- default value of `l_centering_replication` changed from False to True.
- More efficient estimation of IATE, referred to as EffIATE

-----

Version 0.2.4
-------------

Bug fixes
~~~~~~~~~

- Bug fix for cases when outcome had no variation when splitting.

What's New
~~~~~~~~~~

- File with IATEs also contains indicator of specific cluster in k-means clustering.
- Option for guaranteed replicability of results. sklearn.ensemble.RandomForestRegressor does not necessarily replicable results (due to threading). A new keyword argument (l_centering_replication, default is False) is added. Setting this argument to True slows down local centering a but but removes that problem

-----

Version 0.2.3
-------------

Bug fixes
~~~~~~~~~

- Missing information in init.py.

-----

Version 0.2.2
-------------

Bug fixes
~~~~~~~~~

- Bug fix in plotting GATEs.

What's New
~~~~~~~~~~

- ATEs are saved in csv file (same as data for figures and other effects).

-----

Version 0.2.1
-------------

Bug fixes
~~~~~~~~~

- Bug fix in MGATE estimation, which led to program aborting.

-----

Version 0.2.0
-------------

Bug fixes
~~~~~~~~~

- Bug fix for policy trees under restrictions.
- Bug fix for GATE estimation (when weighting was used).

What's New
~~~~~~~~~~

- Main function changed from ``ModifiedCausalForest()`` to ``modified_causal_forest()``.
- Complete seeding of random number generator.
- Keyword modifications:

    - `stop_empty` removed as parameter,
    - `descriptive_stats` becomes `_descriptive_stats`,
    - `dpi` becomes `_dpi`,
    - `fontsize` becomes `_fontsize`,
    - `mp_vim_type` becomes `_mp_vim_type`,
    - `mp_weights_tree_batch` becomes `_mp_weights_tree_batch`,
    - `mp_weights_type` becomes `_mp_weights_type`,
    - `mp_with_ray` becomes `_mp_with_ray`,
    - `no_filled_plot` becomes `_no_filled_plot`,
    - `show_plots` becomes `_show_plots`,
    - `verbose` becomes `_verbose`,
    - `weight_as_sparse` becomes `_weight_as_sparse`,
    - `support_adjust_limits` new keyword for common support.

- Experimental version of continuous treatment. Newly introduced keywords here

    - `d_type`
    - `ct_grid_nn`
    - `ct_grid_w`
    - `ct_grid_dr`

- The optimal policy function contains new rules based on 'black box' approaches, i.e., using the potential outcomes directly to obtain optimal allocations.
- The optimal policy function allows to describe allocations with respect to other policy variables than the ones used for determining the allocation.
- Plots:

    - improved plots
    - new overlapping plots for common support analysis

-----

Version 0.1.4
-------------

Bug fixes
~~~~~~~~~

- Bug fix for predicting from previously trained and saved forests.
- Bug fix in `mcf_init_function` when there are missing values.

What's New
~~~~~~~~~~

- `_mp_ray_shutdown` new defaults. If object size is smaller 100,000, the default is False and else True.

-----

Version 0.1.3
-------------

Bug fixes
~~~~~~~~~

- Minor bug fixes, which led to unstable performance.

What's New
~~~~~~~~~~

- `subsample_factor` is split into `subsample_factor_eval` and `subsample_factor_forest`.
- New default value for `stop_empty`.
- Optimal policy module computes the policy tree also sequentially. For this purpose, the ``optpoltree`` API has changed slightly. Renamed input arguments are

    - `ft_yes`
    - `ft_depth`
    - `ft_min_leaf_size`
    - `ft_no_of_evalupoints`
    - `ft_yes`

- the new input arguments for the sequential tree are:

    - `st_yes`
    - `st_depth`
    - `st_min_leaf_size`

-----

Version 0.1.2
-------------

Bug fixes
~~~~~~~~~

- Common support with very few observations is turned off.
- Minor fix of MSE computation for multiple treatments.  

What's New  
~~~~~~~~~~

- New default values for  

    - `alpha_reg_grid`
    - `alpha_reg_max`
    - `alpha_reg_min`
    - `knn_flag`
    - `l_centering`
    - `mp_parallel`
    - `p_diff_penalty`
    - `random_thresholds`
    - `se_boot_ate`
    - `se_boot_gate`
    - `se_boot_iate`
    - `stop_empty`

- Consistent use of a new random number generator.
- Ray is initialized once.
- Ray can be fine-tuned via

    - `_mp_ray_del`
    - `_mp_ray_shutdown`,
    - `mp_ray_objstore_multiplier` becomes `_mp_ray_objstore_multiplier`

- New options to deal with larger data sets:

    - `reduce_split_sample`: split sample in a part used for estimation and predicting the effects for given x; large prediction sample may increase running time.
    - `reduce_training`: take a random sample from training data.
    - `reduce_prediction`: take a random sample from prediction data.
    - `reduce_largest_group_train`: reduce the largest group in the training data; this should be less costly in terms of precision than taking random samples.

- Optional IATEs via `iate_flag` and optional standard errors via `iate_se_flag`.
- ``ModifiedCausalForest()`` now also returns potential outcomes and their variances.
- `mp_with_ray` is a new input argument to ``‌optpoltree()``;  Ray can be used for multiprocessing when calling ``‌optpoltree()``.
- Block-bootstrap on :math:`w_i \times y_i` is the new clustered standard errors default. This is slower but likely to be more accurate  than the aggregation within-clusters deployed before.

-----

Version 0.1.1
-------------

Bug fixes
~~~~~~~~~

- Minor bug fixes concerning `with_output`, `smaller_sample`, (A,AM)GATE/IATE-ATE plots, and the sampling weights.

What's New
~~~~~~~~~~

- Optional tree-specific subsampling for evaluation sample (subsample variables got new names).
- k-Means cluster indicator for the IATEs saved in file with IATE predictions.
- Evaluation points of GATE figures are included in the output csv-file.
- Exception raised if choice based sampling is activated and there is no treatment information in predictions file.
- New defaults for `random_thresholds`; by default the value is set to 20 percent of the square-root of the number of training observations.
- Stabilizing ``ray`` by deleting references to object store and tasks
- The function ``ModifiedCausalForest()`` returns now ATE, standard error (SE) of the ATE, GATE, SE of the GATE, IATE, SE of the IATE, and the name of the file with the predictions.

-----

Version 0.1.0
-------------

Bug fixes
~~~~~~~~~~

- Bug fix for dealing with missings.
- Bug fixes for problems computing treatment effects for treatment populations.
- Bug fixes for the use of panel data and clustering.

What's New
~~~~~~~~~~

- `post_kmeans_no_of_groups` can now be a list or tuple with multiple values for the number of clusters; the optimal value is chosen through silhouette analysis.
- Detection of numerical variables added; raises an exception for non-numerical inputs.
- All variables used are shown in initial treatment-specific statistics to detect common support issues.
- Improved statistics for common support analysis.

Experimental
~~~~~~~~~~~~

- Optimal Policy Tool building policy trees included bases on estimated IATEs (allowing implicitly for constraints and programme costs).
