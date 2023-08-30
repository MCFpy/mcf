"""
Modified Causal Forest - Python implementation.

Change-log.

Commercial and non-commercial use allowed as compatible with the license for
Python and its modules (and Creative Commons Licence CC BY-SA).

Michael Lechner, SEW, University of St. Gallen, Switzerland

Version: 0.4.0

-*- coding: utf-8 -*- .

Change-log:
New in version 0.0.1
- Optional: Bootstrap standard errors
  SE_BOOT_ATE       # False: No Bootstrap standard errors for ATE, ...
  SE_BOOT_GATE      # > 99: (w_ji y_i) are bootstrapped SE_BOOT_*** times
  SE_BOOT_IATE      # (weights renormalized). Default is False
- Bug fix in objective function (incorrect weighting and incorrect
                                 aggregation for tuning and VI)
- Changes in dealing with common support
  SUPPORT_MAX_DEL_TRAIN = 0.5
    If share of observations in training data used for forest data that are
    OFF support is larger than SUPPORT_MAX_DEL_TRAIN (0-1), programme will be
   terminated and user should change input data.
  Common support checks and corrections are now done before any estimation and
  relate to all data files.
- Use Ray 1.4.1 or higher for Python 3.9 support [pip install ray[default]]
- If specified path for output does already exist, a new path will be created
  and used (solves some issue with writing rights that occured occassionaly)
###############################################################################
New in version 0.1.0
- Kmeans uses a ranges of number of clusters and determines optimal cluster
  by silhoutte analysis (POST_KMEANS_NO_OF_GROUPS can now be a list or tuple
                         with possible cluster numbers)
- (Experimental) Optimal Policy Tool included bases on estimated IATEs
  (allowing implicitly for constraints and programme costs) building
  policy trees
- Bug fixes for problems computing treatment effects for treatment populations
- Detection of numerical variables added. Raises Exception.
- All variables used are shown in initial treatment specific statistics
  (important information to detect common support issues)
- Documentation only: Treatment specific statistics will only be printed for
  those variables used to check the balancing of the sample
- Fix some bugs for use of panel data and clustering
- Improved stat's for common support analysis
- Bug fix for dealing with missings in data cleaning
-------------------------------------------------------------------------------
New in version 0.1.1
- Optional tree-specific subsampling for evaluation sample (subsample variables
                                                            got new names)
- Evaluation points of GATE figures are included in csv-file
- Minor bug fixes (_with_output, _smaller_sample, (A,AM)GATE/IATE-ATE plots,
                   sampling weights)
- k means cluster indicator saved in file with IATE predictions
- Exception raised if choice based sampling is activated and there is no
  treatment information in predictions file
- New defaults for 'random_thresholds'
- Further stabilizing ray by deleting references to object store and tasks
- MCF returns now ATE, Standarderror(ATE), GATE, SE(GATE), IATE, SE(IATE), and
                  the name of the file with the predictions
-------------------------------------------------------------------------------
New in version 0.1.2
- Minor bug fixes (common support with very few observations deselected;
                   _with_output=False led to error in optimal policy modul)
- New default values for penalty function, random_thresholds, alpha regularity,
  number of bootstrap replications for SE estimation, Local centering,
  STOP_EMPTY
- Consistent use of new random number generator
- Documentation on the web is now much clearer which version is the default
  version used (MSE based with matching on outcome score & penalty function)
- As a new default Ray is only initialized ones and uses workers=logical_cores
  - 2;  further ways to finetune Ray are added (_mp_ray_del,_mp_ray_shutdown,
  _mp_ray_shutdown, mp_ray_objstore_multiplier becomes
  _mp_ray_objstore_multiplier).
- For obtaining cluster standard errors, using block-bootstrap on the w_i*y_i
  terms are now the default. This is slower but likely to me more accurate
  (less conservative) than the aggregation within-clusters used before.
- There are few options included now that help with large data sets
  a) reduce_split_sample: Split sample in parts used for estimation the effects
     and the predicting the effects for given x (outcome information is not
     needed for that part). This may also be useful/required for some optimal
     policy analysis. Note that having a large prediction sample may
     substantially increase computation time.
  While this sample split is done in the beginning of the programme,
  the following sample reductions are performed after determining common
  support:
  b) reduce_training: takes random sample from training data
     reduce_prediction: takes random sample from prediction data
  c) reduce_largest_group_train: Reduces the largest group in training. This
     should be less costly in terms of precision lost than taking random
     samples.
- Computing IATEs and their standard erros is optional
  (default is to compute IATE and their standard errors). If IATEs and/or their
  standard errors are not needed this may significantly speed up the programme
  (IATE_FLAG, IATE_SE_FLAG)
- Additional returns from main function ModifiedCausalForest()
- Change of default for variance computation. While Nadaray-Watson estimation
  (previous default) gives a better approximaton of the variance,
  k-NN is much faster, in particular for larger datasets. Therefore, k-NN is
  the new default.
- Fix of possible bug in MSE computation for multiple treatments (+ speed up
  for more than 4 treatments)
- Optimal policy module may use Ray for multiprocessing
-------------------------------------------------------------------------------
New in version 0.1.3
- Minor bug fixes that led to crashing the programme
- New default for stop_empty
- Optimal policy module has the new option to compute a sequential policy tree
- Default for SUBSAMPLE_FACTOR_EVAL is now False (note that values and
                                                  description changed somewhat)
-------------------------------------------------------------------------------
New in version 0.1.4
- _MP_RAY_SHUTDOWN has new default.
- Bug fix for predicting from previously trained and saved forests.
- Bug fix in mcf_init_function when there are missing values
###############################################################################
New in version 0.2.0
- Improved plots.
- New overlap plots for common support analysis.
- Bug fix for GATE estimation (only relevant when weighting is used)
- Some keyword arguments changed names:
    verbose -> _verbose, descriptive_stats -> _descriptive_stats,
    show_plots -> _show_plots, fontsize -> _fontsize, dpi -> _dpi,
    no_filled_plot -> _no_filled_plot, mp_with_ray -> _mp_with_ray,
    mp_vim_type -> _mp_vim_type, mp_weights_type -> _mp_weights_type,
    mp_weights_tree_batch -> _mp_weights_tree_batch,
    weight_as_sparse -> _weight_as_sparse
- New keyword added for common support: support_adjust_limits (see description
                                                               below)
- stop_empty removed as parameter
- Main function changed name: ModifiedCausalForest -> modified_causal_forest
- Results replicate for discrete treatments (complete seeding of random number
                                             generators added)
- Experimental version of continuous treatment module added
 (not yet fully tested, method description will be added in the future)
  - new keyword arguments relating to continuous treatments:
    d_type, ct_grid_nn, ct_grid_w, ct_grid_dr
- The optimal policy modul contains new rules based on 'black box' approaches,
  i.e., using the potential outcomes directly to obtain optimal allocations
- The optimal policy modul allows for describing allocations with respect to
  other policy variables than the ones used for determining the allocation.
  If an observed allocation exists, results will also be computed (i) relative
  to an such an allocation , (ii) for those who's allocated treatment is
  different to the observed treatment
- Bug fix for policy trees under restrictions
- Black Box Optimal Policy allocations comes now with some bootstrap results
  (given the policy scores) as a crude measure for the uncertainty
-------------------------------------------------------------------------------
New in version 0.2.1
- Bug fix in mgate
-------------------------------------------------------------------------------
New in version 0.2.2
- Bug fix in plotting Gates.
- ATEs are now saved in csv file (same as data for figures and other effects).
-------------------------------------------------------------------------------
New in version 0.2.3
- Nothing, just a bug removal for pip install
-------------------------------------------------------------------------------
New in version 0.2.4
- Bug fix for cases when y had no variation when splitting
- File with IATEs also contains indicator of specific cluster in k-means
  clustering (post_est_stats == True and post_kmeans_yes == True)
- There is a problem some in sklearn.ensemble.RandomForestRegressor that leads
  to (slightly) different results when rerunning the programming if there is
  local centering: A new keyword arguments is added that slows down local
  centering a but but removes that problem (l_centering_replication,
                                            default is False)
-------------------------------------------------------------------------------
New in version 0.2.5
- Some small increases in code efficiency
- Minor bug fixes in estimation of mcf
- Bug fix in bootstrap of optimal policy module
- Better organisation of output
    - Name change of file with predicted IATE. It ends as X_IATE.csv
    - Key results in short summary file
    - More intuitive directory structure
    - default value of l_centering_replication changed from False to True
- More efficient estimation of IATE -> EffIATE
    - EffIATE is computed by reversing the role of training and estimation
      sample and taking mean of both runs (this is the new default)
    - No inference is available for the EffIATE, but it will be more efficient
      than the IATE of the first round (as single sample splitting is mainly
                                        needed for inference)
    - Efficient versions of GATEs and ATEs are not provided as inference
      appears to be crucial for these parameters (which is lost in the
                                                  averaging step)
    - EFFIATE_FLAG is a new control variable for this feature
    - eff_iate will be saved together with iate and returned by the
      modified_causal_forest()
###############################################################################
New in version 0.3.0
- Bug fixes:
    - Training and prediction are done in separate runs of programme
    - Issue in opt policy for case when treatment is unobserved is resolved
- New Features:
    - The mcf can be run either with a functional interface or an OOP interface
      - Functional:
          (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
                  pred_outfile, eff_pot_y, eff_iate
                  ) = mcf.modified_causal_forest
        It can seperately train the forest or predict the forest.  This is
        governed by the train_mcf and predict_mcf parameters. If training and
        prediction is separated, all information for prediction is contained in
        files in the directory used for the output.
        When predicting in a separate step, everything is reloaded from this
        directory.
        Although, training and prediction can be separated it is usually most
        efficient to do it together.
      - Object-orientated:
          The is a mcf object with three methods. Proceed as follows:
          (i) Create the mcf object:  mymcf = mcf.ModifiedCausalForest()
          (ii) Use the train method if training only:  mymcf.train()
               * The information needed for prediction is saved on file.
          (iii) Use the predict method if prediction only:
              (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
               pred_outfile, eff_pot_y, eff_iate) =  mymcf.predict()
          (ii+iii) Use the train_predict method if training and prediction
                   us performed in one step (usually most efficient):
                  (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
                   pred_outfile, eff_pot_y, eff_iate) = mymcf.train_predict()
         NOTE: Efficient IATE estimation is not yet implemented for OO version.
    - Replicability of results when there is local centering is added as an
      option (new variable: l_centering_replication). Default  is True.
    - Plots are now more consistent with each other.
    - Predicted potential outcomes for individual data points (IATEs) contain
      also additional predictions for which local centering is reversed
      (new variable: l_centering_undo_iate). Default is True.
    - Nicely formated tables for Gates, AMGates, and MGates are created and
      saved in the respective folders (by Hugo Bodory).
    - Improved optimal policy module (new black-box simulation algorithm added)
        - Main idea: Account for uncertainty of the policy score in the
          allocation decision beyond doing non-zero allocations only for
          effects which are statistically different from zero (which is already
                                                               implemented)
        - Implementation: Draw value from the normal distribution of
          each IATE[i] assuming N(EFFECT_VS_0[i], EFFECT_VS_0_SE [i]), i=1:M.
          If largest value is positive: Assign treatment corresponding to this
                                        IATE.
          If largest value is negative: Assign zero treatment.
        - This approach is much less sophisticated but in the spirit of
          Kitagawa,Lee, Qiu (2022, arXiv) using the sampling distribution
          instead of the posterior distribution.
        - A note on the assumed correlation structure of the IATEs in the case
          of multiple treatments: Implicitly this simulation strategy assumes
          (incorrectly) that the IATEs are independent (which of course is no
          restriction for the binary treatment)
        - Necessary condition: EFFECT_VS_0 and EFFECT_VS_0_SE must be available
        - Control variable: BB_STOCHASTIC (default is False implying
                                           deterministic assignment)
        - Advantages of this approach:
            * Takes sampling uncertainty into account.
            * Average ret of optimal allocation should have much less sampling
              variance (i.e. it will be robust to small changes of the
                        policy-scores)
        - Disadvantages of this approach:
            * Black-box nature
            * Some additional simulation uncertainty is added.
            * Estimator of sampling uncertainty of estimated IATE is required.
            * It has some ad-hoc notion
            * Not yet tested in a simulation or theoretical analysis
            * Incorrect correlation assumptions in case of multiple treatments
        - Stochastic simulations are currently only used for black-box
          approaches
-------------------------------------------------------------------------------
New in version 0.3.1
- Bug fixes:
    - csv files for nice tables with GATEs can also deal with more general
      treatment definitions
    - _mp_with_ray is no longer an argument
    - names_pot_iate is an additional return from the estimator. It is a
      2-tuple with the list of potentially outcomes (all, relative_to_treat0)
    - _return_iate_sp is a new parameter to algorithm to predict and return
      effects despite _with_output = False
- Improved optimal policy modul
  - New experimental feature: A new module is provided (optpolicy_with_mcf)
    that combines mcf estimations of IATEs with optimal policies (black-box and
    policy trees). It also provides out-of-sample evaluations of the
    allocations. More details are provided in the latest version of Cox,
    Lechner, Bollens (2022) and and in the file
    user_evaluate_optpolicy_with_mcf.py
- The files user_mcf_full.py and user_evaluate_optpolicy_with_mcf.py are now
  provided as examples on how to the mcf with all parameters available.
-------------------------------------------------------------------------------
New in version 0.3.2
- From now on, the change log is in a separate (this!) file.
- New option to compute IATE(x) by cross-fitting (only when estimation
  and prediction data are identical), these effects will be saved for further
  use.
  New parameters: iate_cv (def: False), iate_cv_folds (def: 5)
  Additional returns from estimation (iate_cv_file: csv file that contains
  original data as well as predictions of x-fitted potential outcomes and
  IATEs(x); iate_cv_names is a list with the name of the corresponding
  variables of potential outcomes and effects). There is also an additional
  *.txt file (in the same file as the other *.txt files) that contains some
  information about estimation and descriptive statistics.
- New feature to compare GATEs not only to average effects, but also to
  compare them to GATES computed at the previous evaluation point;
  GATES_MINUS_PREVIOUS; default is False
  If true, GATE estimation is a it slower as it is not optimized for
  multiprocessing (conceptionally impossible). Not available for MGATE.
  No plots shown.
- The returned gate is now a tuple containing, the gate, the gate
  difference, the marginal gates, the marginal gate difference,
  the average marginal gate, and the average marginal gate difference
  gate_se has the same structure as gate and contains the corresponding
  standard errors.
- New parameter: n_min_treat (def: 3) - minimum number of observations per
  treatment in leaf.
  A higher value reduces the risk that leaf cannot be filled with outcomes
  from all treatment arms in evaluation subsample. There is no grid based
  tuning for this parameter.
  This impacts the minimum leaf size which will be at least
  n_min_treat * number of treatments
- Minor bug fix for the printing of GATEs, in local centering and optimal
  policy when no additional outcome variables are used.
- Updated call to Kmeans as algorithm 'full' will be deprecated in scikitlearn.
- Updated axes labeling on sorted effects
- Updated default values:
    - Increased default value sampling share when there is subsampling in the
      evaluation sample (to avoid too many empty leaves)
    - New defaults for feature selection: fs_other_sample_share = 0.33,
        fs_rf_threshold = 0.0001
    - Defaults for n_min_min increased to n**0.4/10, at least 3; -1: n**0.4/5
      (where n is the number of observations in the smallest treatment arm)
      all values are subsequently multiplied by no of treatments
    - mp_parallel = 80% of logical cores (reduced for increased stability of
                                          computation)
    - subsample_factor_eval = True (True means 2 * subsample size used for tree
      building (to avoid too many empty leaves)
      Float (>0): Multiplier of subsample size used for tree building
- Name change of keywords for more internal consistency:
    EFFIATE_FLAG -> IATE_EFF_FLAG
    SMOOTH_GATES -> GATES_SMOOTH
    SMOOTH_GATES_BANDWIDTH -> GATES_SMOOTH_BANDWIDTH
    SMOOTH_GATES_NO_EVALUATION_POINTS  -> GATES_SMOOTH_NO_EVALUATION_POINTS
    RELATIVE_TO_FIRST_GROUP_ONLY -> POST_RELATIVE_TO_FIRST_GROUP_ONLY
    BIN_CORR_YES -> POST_BIN_CORR_YES
    BIN_CORR_THRESHOLD -> POST_BIN_CORR_THRESHOLD
- Multiprocessing
  You may ray (default) or dask for multiprocessing  in mcf and optpolicyy.
  Dask is still experimental. Try it (only) if you run into trouble with ray.
  It may be slower in many situations.
  _RAY_OR_DASK ='ray' (default) or _RAY_OR_DASK ='dask'.
  Note that this requires to protect the call of the programme by
  if __name__ == '__main__':
  Because of this change, in optpolicy the keyword mp_with_ray become
  _ray_or_dask with a new syntax
-------------------------------------------------------------------------------
New in version 0.3.3
- It now runs also on Python 3.10
- Bug fixes
  Correction of prognostic score nearest neighbour matching when local
  centering was activated.
- Speed and memory improvements
  Weight matrix computed in smaller chunks for large data
  There is also a parameter that comes along this change (which should usually
  not be changed by the user)
  _weight_as_sparse_splits  Default value is
  round(Rows of prediction data * rows of Fill_y data / (20'000 * 20'000))
- Additional and improved statistics for balancing tests.
- Renaming of output: Marginal effects became Moderated effects
- Name change: m_share_min --> m_min_share, m_share_max --> m_max_share
   nw_kern_flag --> nw_kern, atet_flag --> atet, gatet_flag --> gatet,
   iate_flag --> iate, iate_se_flag --> iate_se, iate_eff_flag --> iate_eff,
   iate_cv_flag --> iate_cv, cond_var_flag --> cond_var, knn_flag --> knn,
   clean_data_flag --> clean_data
- New default values:
  alpha_reg_min = 0.05 alpha_reg_max = 0.15
    If alpha_reg_grid = 1 (default) alpha = (alpha_reg_min+alpha_reg_ax)/2
  m_share_min = 0.1, m_share_max = 0.6, m_grid = 1
     number of variables used for splitting = share * total # of variable
     If m_grid == 1: m_share = (m_share_min + m_share_max)/2
  N_min_min=n_d**0.4/6; at least 4, n_min_max=sqrt(n_d)/6, at least ^4
     (n_d denotes the number of observations in the smallest treatment arm)
     All values are multiplied by the number of treatments.
  n_min_grid=1, if n_min_grid == 1: n_min=(n_min_min+n_min_max)/2
     n_min_treat = n_min_min+n_min_max)/2 / # of treatments / 4. Minimum is 2.
- Improved common support check and adjustment
    - Uses Classification Forest as new default and obtain cut-off limits
    out of sample. To determine the forest, only 20% of the training data
    will be used. Due to the inherent overfitting of the classifier,
    these obsertvations are likely to be removed from the training data.
    New keyword:
    SUPPORT_CF = None
    If True or None, probilities from a single classification forest are used.
    Othterwise, probilities are obtained from a series of regression forests.
    (default is True)
    The default of SUPPORT_ADJUST_LIMITS is set to zero if SUPPORT_CF = True.
- New default for M_RANDOM_POISSON
    True if M > 10, otherwise False (to avoid getting to few variables that do
                                     not have enough variation)
    If covariates used for splitting at specific node have no variation, a new
    set of variables is drawn and a new splitting attempt is conducted. This is
    repeated up to 3 times.
- Important note: When using predict as method, it is important that
                  outpath is correctly specified as location where the train
                  method has saved files needed for prediction. Usually, this
                  is the outpath specified while using train. However, if train
                  is rerun oder this path already exists, train will create a
                  new outpath by adding integers to the file name. This needs
                  to be taken into account when using predict.
###############################################################################
New in version 0.4.0
- This version is a major update of the mcf package. The goal is to increase
  its scalability and reduce the internal complexity (and dependencies). The
  major changes concern as the mcf estimation as well as the optimal policy
  module. Let us start with the mcf estimation, the changes to the optimal
  policy module is discussed below.
- Major general changes
    - From the point of the user, the functional version of the mcf is gone
      (although, under the hut the module remains mainly functional).
      The main methods of the ModifiedCausalForest class are
        train (prepare the data, do feature selection, check common support,
               and local centering).
        predict (compute the effects)
        analyse (print and analyse the estimated heterogeneous effects)
    - The data is no longer passed as csv-file but as a pandas DataFrame.
    - All keywords have been renamed. The new names better reflect the
      use-type.

    - New syntax:
         mymcf = mcf.ModifiedCausalForest(**params) Initialise class
                 See the file mcf_user_full for all parameters.
              mymcf : mcf-class object.

         tree_df, fill_y_df = mymcf.train(training_data_df)
              training_data_df: data used to train the forest (pd.DataFrame)
              tree_df : Data used to construct the forest.  (pd.DataFrame)
              fill_y_df : Data used to population the leaves with outcomes.
                          (pd.DataFrame)

         results = mymcf.predict(predictiondata_df)
              predictiondata_df: Data to compute the effects. All covariates
                                 used for training must be part of this.
                                 Treatment and outcome information is not
                                 necessary. (pd.DataFrame)
              results : Dictionary with all effects.
              results = {
                  'ate': ate, 'ate_se': ate_se,
                  'ate effect_list': ate_effect_list,
                  'gate': gate, 'gate_se': gate_se,
                  'gate_diff': gate_diff, 'gate_diff_se': gate_diff_se,
                  'amgate': amgate, 'amgate_se': amgate_se,
                  'amgate_diff': amgate_diff, 'amgate_diff_se': amgate_diff_se,
                  'iate': iate, 'iate_se': iate_se, 'iate_eff': iate_eff,
                  'iate_data_df': iate_pred_df,
                  'iate_names_dic': iate_names_dic,
                  'bala': bala, 'bala_se': bala_se, 'bala_effect_list':
                      bala_effect_list
                  Variables without '_df' are list or numpy arrays, Variables
                  ending with '_df' are pd.DataFrames.
                  iate_pred_df contains the outcome variables as well as the
                  IATE as well as the corresponding outcomes. Thus, it may
                  subsequently be used for optimal policy analysis.

        results_with_cluster_id_df = mymcf.analyse(results)
                  results : Dict. as output by the predict method.
                  results_with_cluster_id_df: Same as results, but iate_pred_df
                  contains the cluster indicator for each observation as used
                  by the k-means analysis.

- The feature selection algorithm has been fundamentally changed.
    - The idea is to delete a feature if it is irrelevant in the reduced forms
      for the treatment AND the outcome.
    - As default, feature selection is performed on a different random sample
      than the remaining mcf computations.
    - Reduced forms are computed with random forest classifiers or random
      forest regression, depending on the type of variable.
    - Irrelevance is measured by variable importance measures based on
      randomly permuting a single variable and checking its reduction in either
      accuracy (classification) or R2 (regression) compared to the test set
      prediction based on the full model.
    - If the correlation of two variables to be deleted is larger than 0.5, one
      of the two variables is not removed from list of features. Variables
      used to compute GATEs, MGATEs, AMGATEs or a specified as always needed
      are not deleted either. They are not removed also if they are specified
      in 'var_x_name_remain_ord' or 'var_x_name_remain_unord' or are needed
      otherwise.
- Common support
    - Estimation of probabilities is now always based on the random forest
      classifier (the option 'cs_cfo' does no longer exist)
    - If common support is based on cross-validation, then all training data
      will be used in this step.
- GATES
    - Wald tests are no longer provided.
    - Moderated GATES: MGATEs are no longer provided, but AMGATES remain.
    - AMGATEs will be conducted for the same heterogeneity variables as the
      GATES. Seperate variables for the AMGATES no longer exist.
      A new parameter is used instead: AMGATE : Boolean, True: Compute AMGATES.
      The default is False.
- IATES
    - A new parameter has been added so that the use can decide whether to
      compute IATE(x)-ATE or not.
      'p_iate_m_ate' :  True: IATE(x) - ATE is estimated, including inference
      if p_iate_se == True. Increaes computation time.   Default is False.
- BGATES (new causal effect)
    - Effect heterogeneity as estimated by GATEs may either come from
      differences in the heterogeneity variable or the fact that other
      features covary with the heterogeneity variable. Sometimes this makes the
      interpretation of the GATEs less useful.
      However, while GATEs do not keep any other background characteristics
      constant, AMGATEs keep ALL background charcteristics constant.
      In this case, difference in causal effects can only be due to the
      respective heterogeneity variable (or UNobservables related to the
      heterogeneity variable). However, there may be situations where it is
      either not useful for the interpretation to keep ALL background variables
      constant, and/or it is technically challenging to do so, for example if
      some of the background variables are very highly correlated with the
      the heterogeneity variable. In such a situation it may be useful to
      balance only a prespecified set of covariates. This parameter is called
      the Balanced Group Average Treatment Effect (BGATE) by
      Bearth and Lechner (2023). While they propose a double machine learning
      estimator, here we implemented a simulation based alternative that can
      be easily used by the mcf algorithms.
      The following algorithm is implemented (similar to the AMGATE):
          i) Draw a random sample from the prediction data of size n. Keep only
                      the heterogeneity variable (Z) and the variables to
                      balance on (B).
          ii) If Z has n_z different values, replicate this data n_z times. In
                      each of the n_z folds set z to a specific value.
                      n_all = n_z x n
          iii) For each of the n_all observations (i)), draw an observation in
                      the prediction data that is closest to it in terms of b_i
                      and z_i (nearest neighbour in terms of b_i, z_i). If
                      there more than one nearest neighbour, randomly chosen
                      one of them. Form a new sample with all selected
                      neighbours.
          iv) With this new sample, GATEs and their standard errors are
                      estimated in a standard mcf fashion. However, since they
                      are based on modified data, they are to be interpretated
                      as BGATES.
     Two new parameters relate the BGATE:
         p_bgate : Boolean. True: BGATEs will be computed. Default is False.
            True requires to specify the variable names to balance on in
         var_bgate_name : List. Names of variables to balance on. Default is
            the other heterogeneity variables if there are any.

- Reduction of sample of training and prediction
  The previous versions had options to systematically reduce samples. This
  options are now removed because due to new data concept, the user can do such
  reductions easily outside the programme.
  The following options DO NOT EXIST ANYMORE:
  _INT_RED_SPLIT_SAMPLE = False            # Default is False
  _INT_RED_SPLIT_SAMPLE_PRED_SHARE = None  # 0<..<1: Share used for prediction
  #   Split sample randomly in parts used for estimation the effects and the
  #   predicting of the effects for given x (outcome information is not needed
  #   for that part). This may also be useful/required for some optimal policy
  #   analysis.
  #   This works only if indata and preddata are identical or preddata is not
  #   specified.
  _INT_SMALLER_SAMPLE = None    # 0<test_only<1: test prog.with smaller sample
  #   While this sample split is done in the beginning of the programme,
  #   the following sample reductions are performed after determining common
  #   support (only observations on common support are used).
  _INT_RED_TRAINING = False       # Random sample of indata. Default is False.
  _INT_RED_TRAINING_SHARE = None  # 0<...<1: Share to keep. Default is 0.5
  _INT_RED_PREDICTION = False     # Random sample of preddata. Default False.
  _INT_RED_PREDICTION_SHARE = None  # 0<...<1: Share to keep.Default is 0.5
  _INT_RED_LARGEST_GROUP_TRAIN = False  # Reduces the largest treatment group.
     Should be less costly in terms of precision lost than taking random
     samples.
  _INT_RED_LARGEST_GROUP_TRAIN_SHARE = None  # 0<...<1: Default (None) is 0.5

- Unified data concept to deal with common support and local centering
  LC_CS_CV = True     # True: Use crossvalidation (def).
                        False: Use random sample that will not used for CF.
  LC_CS_SHARE = None  # Share of data used for estimating E(y|x).
                        0.1-0.9 (def = 0.25)
  LC_CS_CV_K = None   # if LC_CS_CV: # of folds used in crossvalidation(def:5).

- Improved scalability (new feature)
  Idea:   Randomly split training data for large samples in chunks
          & take average.
  CF_CHUNKS_MAXSIZE = None  # Maximum allowed number of observations per block.
    If larger than sample size, no random splitting.
    Default (None): round(60000 + sqrt(number of observations - 60000))

- Due to a bug in scikit-learn,replicability was a problem when multiprocessing
  is used in scikit-learn methods (njobs keyword). To avoid this during
  training, there a new keyword 'gen_replication'. If True, multiprocessing in
  skikit-learn is avoided. This slows down computations but leads to
  replicability of the results over several runs of the programme. Currently
  scikit-learn in training is used in local centering, common support, and
  feature selection. It defaults to False, since the bug has been fixed in the
  subsequent version. In the previous version this keyword refered only to
  local centering.

- New defaults
    - P_IATE_SE defaults now to False (instead of True).
    - GEN_IATE_EFF defaults now to False (instead of True).
    - Subsample used to compute common support (25%)
    - Default of support_adjust_limits is always (no_of_treat - 2) * 0.05
    - CF_N_MIN_MAX largest minimum leaf size.  max(sqrt(n_d) / 6, 3) instead of
      max(sqrt(n_d) / 6, 3)
    - P_BT_YES (balancing test) is now False.
    - GEN_REPLICATION default now to False as the problem with scikit-learn
      that led to introduction of this keyword seems to be solved.

- Data preparation
    - Added check whether variable name is contained more than once in data

- Dask is no longer used for multiprocessing (ray only). Therefore, the keyword
 '__ray_or_dask' does no longer exist.

- This version contains also a major update of the optimal policy module of the
  mcf package. The goal is to reduce the internal complexity
  (and dependencies; new methods can now be added easily).
- Major general changes
    - From the point of the user, the functional version of the optimal policy
      module is gone (although, under the hut the module remains mainly
                     functional).
      The methods of the OptimalPolicy class are solve, allocate, and
      evaluate (plus print_time_strings_all_steps).

    - The data is no longer passed as csv-file but as a pandas DataFrame.
    - All keywords have been renamed. The new names better reflect the
      use-type. The file optpolicy_user_full contains an explanations for all
      parameters of this module.

- New syntax:
    myoptp = op.OptimalPolicy(**params)
             Initiate an instance of the Optimal Policy Class.
    alloc_df = myoptp.solve(train_df, data_title=TRAINDATA)
             Solve the methods on some data containing the policy scores.
             This method outputs a DataFrame with the obtained allocation in
             the training data.
    alloc_df = myoptp.allocate(pred_df, data_title=PREDDATA)
             Use the algorithm obtained from the train method to allocate
             treatmnent in in another data.
             This method outputs a DataFrame with the obtained allocation in
             the training data.
    results_eva = myoptp.evaluate(alloc_df, pred_df, data_title=PREDDATA)
             Evalute the obtained allocation and compare it to the observed
             allocation (if available) as well as a random allocation.
             This method outputs a dictionary with the results that can also
             be printed.

- Both the mcf and the optimal policy module run on Python 3.11 (which is the
  recommended version.
-------------------------------------------------------------------------------
New in version 0.4.1
- Bug fix for AMGATE and Balanced GATE (BGATE)
- Minor bug fixes in Forest and  Optimal Policy module
- Improved docstrings for classes and methods --> improved documentation
- We provide example data and example files on how to use ModifiedCausalForest
  and OptimalPolicy in various ways.
  - The following data files are provided:
      data_x_1000.csv
      data_x_4000.csv
      data_x_ps_1_1000.csv
      data_x_ps_2_1000.csv
      data_y_d_x_1000.csv
      data_y_d_x_4000.csv
      The names are self-explanatory. The number denotes the sample size, x are
      features, y is outcome, d is treatment, and ps denotes policy scores.
  - The following example programmes are provided:
      all_parameters_mcf.py, all_parameters_optpolicy.py
         Contain an explanation of all available parameters / keywords for the
         ModifiedCausalForest and OptimalPolicy classes.
      min_parameters_mcf.py, min_parameters_optpolicy.py
         Contains the minimum specifications to run the methods of the
         ModifiedCausalForest and OptimalPolicy classes.
      training_prediction_data_same_mcf.py
         One suggestion on how to proceed when data to train and fill the
         forest are the same as those used to compute the effects.
      mcf_and_optpol_combined.py
         One suggestion on how to combine mcf and optimal policy estimation in
         a simple split sample approach.
-------------------------------------------------------------------------------
New in version 0.4.2

# TODO: Optimize memory by allowing to delete forest after mcf object after
        using it to compute weights: If deleted, this does not allow to use the
        object for prediction again (if needed, save it beforehand) --> reduces
        demand on memory ... this is the part currently in comments.
# TODO: Check what is the difference between the various IATEs saved in the
       results dictionary. They seem to be identical.

# TODO: Beatify output by use the original variable name and not 'PR'
#       Use 'PRtransformPR' as extension and remove this extension in output
#       whenever found.

"""
