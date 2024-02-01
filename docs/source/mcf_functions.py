class ModifiedCausalForest:
    """
    The class contains all methods necessary for a complete :py:class:`~mcf_functionsModifiedCausalForest` estimation.

    Parameters
    ----------
    cf_alpha_reg_grid : Integer (or None), optional
        Minimum remaining share when splitting leaf: Number of grid values.
        If grid is used, optimal value is determined by out-of-bag
        estimation of objective function. Default (or None) is 1.
        
    cf_alpha_reg_max : Float (or None), optional
        Minimum remaining share when splitting leaf: Largest value of
        grid (keep it below 0.2). Default (or None) is 0.15.
        
    cf_alpha_reg_min : Float (or None), optional
        Minimum remaining share when splitting leaf: Smallest value of
        grid (keep it below 0.2). Default (or None) is 0.05.
        
    cf_boot : Integer (or None), optional
        Number of Causal Trees. Default (or None) is 1000.
        
    cf_chunks_maxsize : Integer (or None), optional
        Randomly split training data in chunks and take average of the
        estimated parameters to improve scalability (increases speed and
        reduces demand on memory, but may increase finite sample bias
        somewhat).
        If cf_chunks_maxsize is larger than sample size, there is no random
        splitting.
        If None:   
        
        .. math:: 
        
            \\text{cf_chunks_maxsize} = 75000 + \\frac{{(\\text{number of observations} - 75000)^{0.8}}}{{(\\text{no_of_treatment} - 1)}}
        
        Default is None.
        
    cf_n_min_grid : Integer (or None), optional
        Minimum leaf size: Number of grid values.
        If grid is used, optimal value is determined by out-of-bag
        estimation of objective function. Default (or None) is 1.
        
    cf_n_min_max : Integer (or None), optional
        Minimum leaf size: Largest minimum leaf size.
        If None: 

        .. math:: 
        
            \\text{A} = \\frac{\\sqrt{\\text{number of observations in the smallest treatment group}}^{0.5}}{10}, \\text{at least} 2
        
        :math:`\\text{cf_n_min_max} = \\text{round}(A \\times \\text{number of treatments})`
        Default is None.
        
    cf_n_min_min : Integer (or None), optional
        Minimum leaf size: Smallest minimum leaf size.
        If None: 

        .. math::

            \\text{A} = \\text{number of observations in smallest treatment group}^{0.4} / 10, \\text{at least} 1.5
                      
        :math:`\\text{cf_n_min_min} = \\text{round}(A \\times \\text{number of treatments})`
        Default is None.
        
    cf_n_min_treat : Integer (or None), optional
        Minimum number of observations per treatment in leaf.
        A higher value reduces the risk that a leaf cannot be filled with
        outcomes from all treatment arms in the evaluation subsample.
        There is no grid based tuning for this parameter.
        This parameter impacts the minimum leaf size which will be at least
        to :math:`\\text{n_min_treat} \\times \\text{number of treatments}`
        None:  
                
        .. math::
        
            \\frac{\\frac{{\\text{n_min_min}} + {\\text{n_min_max}}}{2}}{\\text{number of treatments} \\times 10}, \\text{at least} 1
        
        Default is None.
        
    cf_match_nn_prog_score : Boolean (or None), optional
        Choice of method of nearest neighbour matching.
        True: Prognostic scores. False: Inverse of covariance matrix of
        features. Default (or None) is True.
        
    cf_nn_main_diag_only : Boolean (or None), optional
        Nearest neighbour matching: Use main diagonal of covariance matrix
        only. Only relevant if match_nn_prog_score == False.
        Default (or None) is False.
        
    cf_m_grid : Integer (or None), optional
        Number of variables used at each new split of tree: Number of grid
        values.
        If grid is used, optimal value is determined by out-of-bag
        estimation of objective function. Default (or None) is 1.
        
    cf_m_random_poisson : Boolean (or None), optional
        Number of variables used at each new split of tree:
        True: Number of randomly selected variables is stochastic for each 
        split, drawn from a Poisson distribution. Grid gives mean
        value of 1 + poisson distribution (m-1) (m is determined by
        cf_m_share parameters).
        False : No additional randomisation. Default (or None) is True.
        
    cf_m_share_max : Float (or None), optional
        Share of variables used at each new split of tree: Maximum.
        Default (or None) is 0.6.
        
    cf_m_share_min : Float (or None), optional
        Share of variables used at each new split of tree: Maximum.
        Default (or None) is 0.1.
        
    cf_mce_vart : Integer (or None), optional
        Splitting rule for tree building:
        0: mse's of regression only considered
        1: mse+mce criterion (default)
        2: -var(effect): heterogeneity maximising splitting rule of
        Wager & Athey (2018)
        3: randomly switching between outcome-mse+mce criterion
        & penalty functions
        Default (or None) is 1.
        
    cf_p_diff_penalty : Integer (or None), optional
        Penalty function (depends on value of mce_vart):
        mce_vart == 0: Irrelevant
        mce_vart == 1: Multiplier of penalty (in terms of var(y))
        0: no penalty
        None: 
        
        .. math::
              
            \\frac{2 \\times (\\text{n} \\times \\text{subsam_share})^{0.9}}{\\text{n} \\times \\text{subsam_share}} \\times \\sqrt{\\frac{\\text{no_of_treatments} \\times (\\text{no_of_treatments} - 1)}{2}}

        mce_vart == 2: Multiplier of penalty (in terms of MSE(y) value
        function without splits) for penalty.
        0: no penalty
        None: 

        .. math::
        
            \\frac{100 \\times 4 \\times (n \\times \\text{f_c.subsam_share})^{0.8}}{n \\times \\text{f_c.subsam_share}}
              
        mce_vart == 3: Probability of using p-score (0-1)
        None: 0.5.
        Increase value if balancing tests indicate problems.
        Default is None.
        
    cf_subsample_factor_forest : Float (or None), optional
        Multiplier of default size of subsampling sample (S) used to build
        tree. 

        .. math::
        
            S = \\min(0.67, \\frac{2 \\times (n^{0.8})}{n}), \\text{n: # of training observations} 

        :math:`S \\times \\text{cf_subsample_factor_forest}, \\text{is not larger than 80%.}` 
        Default (or None) is 1.
        
    cf_subsample_factor_eval : Float or Boolean  (or None), optional
        Size of subsampling sample used to populate tree.
        False: No subsampling in evaluation subsample
        True or None: :math:`(2 \\times subsample size)` used for tree building (to
        avoid too many empty leaves)
        Float (>0): Multiplier of subsample size used for tree building
        In particular for larger samples, using subsampling in evaluation
        will speed up computations and reduces demand on memory.
        Tree-specific subsampling in evaluation sample increases speed
        at which asymtotic bias disappears (at the expense of slower
        disappearance of the variance; however, simulations so far show no
        relevant impact). Default is None.
        
    cf_random_thresholds : Integer (or None), optional
        Use only a random selection of values for splitting (continuous
        feature only; re-randomize for each splitting decision; fewer
        thresholds speeds up programme but may lead to less accurate
        results).
        0: no random thresholds.
        > 0: number of random thresholds used for ordered variables.
        None:
        :math:`4 + \\text{number of training observations}^{0.2}`
        Default is None.
        
    cf_vi_oob_yes : Boolean (or None), optional
        Variable importance for causal forest computed by permuting
        single variables and comparing share of increase in objective
        function of mcf (computed with out-of-bag data).
        Default (or None) is False.
        
    cs_type : Integer (or None), optional
        Common support adjustment: Method.
        0: No common support adjustment
        1,2: Support check based on estimated classification forests.
        1: Min-max rules for probabilities in treatment subsamples.
        2: Enforce minimum and maximum probabilities for all obs
        all but one probability
        Observations off support are removed. Out-of-bag predictions
        are used to avoid overfitting (which would lead to a too
        large reduction in the number of observations).
        Default (or None) is 1.
        
    cs_adjust_limits : Float (or None), optional
        Common support adjustment: Accounting for multiple treatments.
        None: 
        :math:`(\\text{number of treatments} - 2) \\times 0.05`
        If cs_type > 0:
        :math:`\text{upper limit} \\times = 1+support_adjust_limits`,
        :math:`lower limit \\times = 1-support_adjust_limits`. 
        The restrictiveness of the common support criterion increases with
        the number of treatments. This parameter allows to reduce this
        restrictiveness. Default is None.
        
    cs_max_del_train : Float (or None), optional
        Common support adjustment: If share of observations in training
        data used that are off support is larger than cs_max_del_train
        (0-1), an exception is raised. In this case, user should change
        input data. Default (or None) is 0.5.
           
    cs_min_p : Float (or None), optional
        Common support adjustment: If cs_type == 2, observations are
        deleted if :math:`p(d=m|x)` is less or equal than cs_min_p for at least
        one treatment. Default (or None) is 0.01.
           
    cs_quantil : Float (or None), optional
        Common support adjustment: How to determine upper and lower bounds.
        If CS_TYPE == 1: 1 or None: min-max rule
        < 1: respective quantil
        Default (or None) is 1.
        
    ct_grid_dr : Integer (or None), optional
        Number of grid point for discretization of continuous treatment
        (with 0 mass point; grid is defined in terms of quantiles of
        continuous part of treatment): Dose response function.
        Default (or None) is 100.
        
    ct_grid_nn : Integer (or None), optional
        Number of grid point for discretization of continuous treatment
        (with 0 mass point; grid is defined in terms of quantiles of
        continuous part of treatment): Neighbourhood matching.
        Default (or None) is 10.
        
    ct_grid_w : Integer (or None), optional
        Number of grid point for discretization of continuous treatment
        (with 0 mass point; grid is defined in terms of quantiles of
        continuous part of treatment): Weights. Default (or None) is 10.
        
    dc_clean_data : Boolean (or None), optional
        Clean covariates. Remove all rows with missing observations and
        unnecessary variables from DataFrame. Default (or None) is True.
        
    dc_check_perfectcorr : Boolean (or None)
        Screen and clean covariates: Variables that are perfectly
        correlated with each others will be deleted.
        Default (or None) is True.
        
    dc_min_dummy_obs : Integer (or None), optional
        Screen covariates: If > 0 dummy variables with
        less than dc_min_dummy_obs observations in one category will be
        deleted. Default (or None) is 10.
        
    dc_screen_covariates : Boolean (or None), optional
        Screen and clean covariates. Default (or None) is True.
        
    fs_yes : Boolean (or None), optional
        Feature selection before building causal forest: A feature is
        deleted if it is irrelevant in the reduced forms for the treatment
        AND the outcome. Reduced forms are computed with random forest
        classifiers or random forest regression, depending on the type of
        variable. Irrelevance is measured by variable importance measures
        based on randomly permuting a single variable and checking its
        reduction in either accuracy (classification) or R2 (regression)
        compared to the test set prediction based on the full model.
        Exceptions: (i) If the correlation of two variables to be deleted
        is larger than 0.5, one of the two variables is kept.
        (ii) Variables used to compute GATEs, BGATEs, CBGATEs. Nor are
        variables removed if they are contained in 'var_x_name_remain_ord'
        or 'var_x_name_remain_unord' or are needed otherwise.
        If the number of variables is very large (and the space of
        relevant features is much sparser, then using feature selection is
        likely to improve computational and statistical properties of the
        mcf etimator). Default (or None) is False.
        
    fs_rf_threshold : Integer or Float (or None), optional
        Feature selection: Threshold in terms of relative loss of variable
        importance in %. Default (or None) is 1.
        
    fs_other_sample : Boolean (or None), optional
        True (default): Random sample from training data used. These
        observations will not be used for causal forest.
        False: Use the same sample as used for causal forest estimation.
        Default (or None) is True.
        
    fs_other_sample_share : Float (or None), optional
        Feature selection: Share of sample used for feature selection
        (only relevant if fs_other_sample is True).
        Default (or None) is 0.33.
        
    gen_d_type : String (or None), optional
        Type of treatment. 'discrete': Discrete treatment.
        'continuous': Continuous treatment.
        Default (or None) is 'discrete'.
        
    gen_iate_eff : Boolean (or None), optional
        Additionally, compute more efficient IATE (IATE are estimated twice
        and averaged where role of tree_building and tree_filling sample is
        exchanged; X-fitting). No inference is not attempted for these
        parameters. Default (or None) is False.
        
    gen_mp_parallel : Integer (or None), optional
        Number of parallel processes (using ray on CPU). The smaller this
        value is, the slower the programme, the smaller its demand on RAM.
        None: 80% of logical cores. Default is None.
        
    gen_outfiletext : String (or None), optional
        File for text output. (\*.txt) file extension will be added.
        None: 'txtFileWithOutput'. Default is None.
        
    gen_outpath : String (or None), optional
        Path were the output is written too (text, estimated effects, etc.)
        If specified directory does not exist, it will be created.
        None: An (\*/out) directory below the current directory is used.
        Default is None.
        
    gen_output_type : Integer (or None), optional
        Destination of text output. 0: Terminal. 1: File. 2: Terminal and
        file. Default (or None) is 2.
        
    gen_panel_data : Boolean (or None), optional
        Panel data used. p_cluster_std is set to True.
        Default (or None) is False.
        
    gen_panel_in_rf : Boolean (or None), optional
        Panel data used: Use panel structure also when building the random
        samples within the forest procedure. Default (or None) is True.
        
    gen_weighted : Boolean (or None), optional
        Use of sampling weights to be provided in var_w_name.
        Default (or None) is False.
        
    lc_cs_cv : Boolean (or None), optional
        Data to be used for local centering & common support adjustment.
        True: Crossvalidation. False: Random sample not to be used for
        forest building. Default (or None) is True.
        
    lc_cs_cv_k : Integer (or None), optional
        Data to be used for local centering & common support adjustment:
        Number of folds in cross-validation (if lc_cs_cv is True).
        Default (or None) is 5.
        
    lc_cs_share : Float (or None), optional
        Data to be used for local centering & common support adjustment:
        Share of trainig data (if lc_cs_cv is False).
        Default (or None) is 0.25.
        
    lc_yes : Boolean (or None), optional
        Local centering. Default (or None) is True.
        
    lc_uncenter_po : Boolean (or None), optional
        Predicted potential outcomes are re-adjusted for local centering
        are added to data output (iate and iate_eff in results dictionary).
        Default (or None) is True.
        
    p_atet : Boolean (or None), optional
        Compute effects for specific treatment groups. Only possible if
        treatment is included in prediction data.
        Default (or None) is False.
        
    p_gatet : Boolean (or None), optional
        Compute effects for specific treatment groups. Only possible if
        treatment is included in prediction data.
        Default (or None) is False.
        
    p_cbgate : Boolean (or None), optional
        Estimate a GATE that is balanced in all other features.
        Default (or None) is False.
        
    p_bgate : Boolean (or None), optional
        Estimate a GATE that is balanced in selected features (as specified
        in var_bgate_name. Default (or None) is False.
        
    p_gates_minus_previous : Boolean (or None), optional
        Estimate increase of difference of GATEs, CBGATEs, BGATEs when
        evaluated at next larger observed value.
        Default (or None) is False.
        
    p_gates_smooth : Boolean (or None), optional
        Alternative way to estimate GATEs for continuous features. Instead
        of discretizing variable, its GATE is evaluated at
        p_gates_smooth_no_evalu_points. Since there are likely to be no
        observations, a local neighbourhood around the evaluation points is
        considered. Default (or None) is True. 
        
    p_gates_smooth_bandwidth : Float (or None), optional
        Multiplier for bandwidth used in (C)BGATE estimation with smooth
        variables. Default is 1.
        
    p_gates_smooth_no_evalu_points : Integer (or None), optional
        Number of evaluation points for discretized variables in GATE
        estimation. Default (or None) is 50.
        
    p_gate_no_evalu_points : Integer (or None), optional
        Number of evaluation points for discretized variables in (C)BGATE
        estimation. Default (or None) is 50.
        
    p_bgate_sample_share : Float (or None), optional
        Implementation of (C)BGATE estimation is very cpu intensive.
        Therefore, random samples are used to speed up the programme if
        there are number observations  / number of evaluation points > 10.
        None: 
        If observation in prediction data (n) < 1000: 1
        If n >= 1000: 
        
        .. math::
        
            1000 + \\frac{{(n - 1000)^{\\frac{3}{4}}}}{{\\text{evaluation points}}}
        
        Default is None.
        
    p_max_cats_z_vars : Integer (or None), optional
        Maximum number of categories for discretizing continuous z
        variables. None: :math:`text{Number of observations}^{0.3}`
        Default is None.
        
    p_iate : Boolean (or None), optional
        IATEs will be estimated. Default (or None) is True.
        
    p_iate_se : Boolean (or None), optional
        Standard errors of IATEs will be estimated.
        Default (or None) is False.
        
    p_iate_m_ate : Boolean (or None), optional
        IATEs minus ATE will be estimated. Default (or None) is False.
        
    p_ci_level : Float (or None), optional
        Confidence level for bounds used in plots.
        Default (or None) is 0.9.
        
    p_cond_var : Boolean (or None), optional
        True: Conditional mean & variances are used.
        False: Variance estimation uses :math:`wy_i = w_i \\times y_i` directly.
        Default (or None) is True.
        
    p_knn : Boolean (or None), optional
      True: k-NN estimation. False: Nadaraya-Watson estimation.
      Nadaray-Watson estimation gives a better approximaton of the
      variance, but k-NN is much faster, in particular for larger datasets.
      Default (or None) is True.
      
    p_knn_min_k : Integer (or None), optional
        Minimum number of neighbours k-nn estimation.
        Default (or None) is 10.
        
    p_nw_bandw : Float (or None), optional
        Bandwidth for nw estimation: Multiplier of Silverman's optimal
        bandwidth. Default (or None) is 1.
        
    p_nw_kern : Integer (or None), optional
        Kernel for Nadaraya-Watson estimation: 1: Epanechikov
        2: normal pdf. Default (or None) is 1.
        
    p_max_weight_share : Float (or None), optional
        Truncation of extreme weights. Maximum share of any weight, 0 <,
        <= 1. Enforced by trimming excess weights and renormalisation for
        each (BG,G,I,CBG)ATE separately. Because of renormalisattion, the
        final weights could be somewhat above this threshold.
        Default (or None) is 0.05.
        
    p_cluster_std : Boolean (or None), optional
        Clustered standard errors. Always True if gen_panel_data is True.
        Default (or None) is False.
        
    p_se_boot_ate : Integer or Boolean (or None), optional
        Bootstrap of standard errors for ATE. Specify either a Boolean (if
        True, number of bootstrap replications will be set to 199) or an
        integer corresponding to the number of bootstrap replications (this
        implies True). None: 199 replications p_cluster_std is True,
        and False otherwise. Default is None.
        
    p_se_boot_gate : Integer or Boolean (or None), optional
        Bootstrap of standard errors for GATE. Specify either a Boolean (if
        True, number of bootstrap replications will be set to 199) or an
        integer corresponding to the number of bootstrap replications (this
        implies True). None: 199 replications p_cluster_std is True,
        and False otherwise. Default is None.
        
    p_se_boot_iate : Integer or Boolean (or None), optional
        Bootstrap of standard errors for IATE. Specify either a Boolean (if
        True, number of bootstrap replications will be set to 199) or an
        integer corresponding to the number of bootstrap replications (this
        implies True). None: 199 replications p_cluster_std is True,
        and False otherwise. Default is None.
        
    p_bt_yes : Boolean (or None), optional
        ATE based balancing test based on weights. Relevance of this test
        in its current implementation is not fully clear.
        Default (or None) is True.
        
    p_choice_based_sampling : Boolean (or None), optional
        Choice based sampling to speed up programme if treatment groups
        have very different sizes. Default (or None) is False.
        
    p_choice_based_probs : List of Floats (or None)
        Choice based sampling:  Sampling probabilities to be specified.
        These weights are used for (G,B,CB)ATEs only. Treatment information
        must be available in prediction data. Default is None.
        
    p_ate_no_se_only : Boolean (or None)
        Computes only the ATE without standard errors. Default is False.
        
    post_est_stats : Boolean (or None), optional
        Descriptive Analyses of IATEs (p_iate must be True).
        Default (or None) is True.
        
    post_relative_to_first_group_only : Boolean (or None), optional
        Descriptive Analyses of IATEs: Use only effects relative to
        treatment with lowest treatment value. Default (or None) is True.
        
    post_bin_corr_yes : Boolean (or None), optional
        Descriptive Analyses of IATEs: Checking the binary correlations of
        predictions with features. Default (or None) is True.
        
    post_bin_corr_threshold : Float, optional
        Descriptive Analyses of IATEs: Minimum threshhold of absolute
        correlation to be displayed. Default is 0.1.
        
    post_kmeans_yes : Boolean (or None), optional
        Descriptive Analyses of IATEs: Using k-means clustering to analyse
        patterns in the estimated effects. Default (or None) is True.
        
    post_kmeans_no_of_groups : Integer or List or Tuple (or None), optional
        Descriptive Analyses of IATEs: Number of clusters to be build in
        k-means.
        None: List of 5 values: [a, b, c, d, e]; c = 5 to 10;
        depending on number of observations; c<7: a=c-2, b=c-1, d=c+1,
        e=c+2, else a=c-4, b=c-2, d=c+2, e=c+4. Default is None.
        
    post_kmeans_max_tries : Integer (or None), optional
        Descriptive Analyses of IATEs: Maximum number of iterations of
        k-means to achive convergence. Default (or None) is 1000.
        
    post_kmeans_replications : Integer (or None), optional
        Descriptive Analyses of IATEs: Number of replications with random
        start centers to avoid local extrema. Default (or None) is 10.
        
    post_random_forest_vi : Boolean (or None), optional
        Descriptive Analyses of IATEs: Variable importance measure of
        random forest used to learn factors influencing IATEs.
        Default (or None) is True.
         
    post_plots : Boolean (or None), optional
        Descriptive Analyses of IATEs: Plots of estimated treatment
        effects. Default (or None) is True.
        
    p_knn_const : Boolean (or None), optional
        Multiplier of default number of observation used in movering
        average of analyses method. Default (or None) is 1.
        
    var_bgate_name :  String or List of strings (or None), optional
        Variables to balance the GATEs on. Only relevant if P_BGATE is
        True. The distribution of these variables is kept constant when a
        BGATE is computed. None: Use the other heterogeneity variables
        (var_z_ ...) (if there are any) for balancing. Default is None.
        
    var_cluster_name :  String or List of string (or None)
        Name of variable defining clusters. Only relevant if p_cluster_std
        is True. Default is None.
        
    var_d_name : String or List of string (or None), optional
        Name of treatment variable. Must be provided to use the train
        method. Can be provided for the :py:meth:`~mcf_functionsModifiedCausalForest.predict` method.
        
    var_id_name : String or List of string (or None)
        Identifier. None: Identifier will be added the data.
        Default is None.
        
    var_w_name : String or List of string (or None), optional
        Name of weight. Only relevant if gen_weighted is True.
        Default is None.
        
    var_x_balance_name_ord : String or List of strings (or None), optional
        Name of ordered variables to be used in balancing tests. Only
        relevant if p_bt_yes is True. Default is None.
        
    var_x_balance_name_unord : String or List of strings (or None),
                               optional
        Name of Ordered variables to be used in balancing tests. Treatment
        specific descriptive statistics are only printed for those
        variables. Default is None.
        
    var_x_name_always_in_ord : String or List of strings (or None),
                               optional
        Name of ordered variables that always checked on when deciding on
        the next split during tree building. Only relevant for train
        method. Default is None.
        
    var_x_name_always_in_unord : String or List of strings (or None),
                                 optional
        Name of Unordered variables that always checked on when deciding on
        the next split during tree building. Only relevant for train
        method. Default is None.
        
    var_x_name_remain_ord : String or List of strings (or None), optional
        Name of ordered variables that cannot be removed by feature
        selection. Only relevant for :py:meth:`~mcf_functionsModifiedCausalForest.train` method. Default is None.
        
    var_x_name_remain_unord : String or List of strings (or None), optional
        Name of unordered variables that cannot be removed by feature
        selection. Only relevant for :py:meth:`~mcf_functionsModifiedCausalForest.train` method. Default is None.
        
    var_x_name_ord : String or List of strings (or None), optional
        Name of ordered features. Either ordered or unordered features
        must be provided. Default is None.
        
    var_x_name_unord : String or List of strings (or None), optional
        Name of unordered features. Either ordered or unordered features
        must be provided. Default is None.
        
    var_y_name : String or List of strings (or None), optional
        Name of outcome variables. If several variables are specified,
        either var_y_tree_name is used for tree building, or (if
        var_y_tree_name is None), the 1st variable in the list is used.
        Only necessary for :py:meth:`~mcf_functionsModifiedCausalForest.train` method. Default is None.
        
    var_y_tree_name : String or List of string (or None), optional
        Name of outcome variables to be used to build trees. This is only
        relevant if many outcome variables are specified in var_y_name.
        Only relevant for :py:meth:`~mcf_functionsModifiedCausalForest.train` method. Default is None.
        
    var_z_name_list : String or List of strings (or None), optional
        Names of ordered variables with many values to define
        causal heterogeneity. They will be discretized (and dependening
        p_gates_smooth) also treated as continuous. If not already included
        in var_x_name_ord, they will be added to the list of features.
        Default is None.
        
    var_z_name_ord : String or List of strings (or None), optional
        Names of ordered variables with not so many values to define causal
        heterogeneity. If not already included in var_x_name_ord, they will
        be added to the list of features. Default is None.
        
    var_z_name_unord : String or List of strings (or None), optional
        Names of unordered variables with not so many values to define
        causal heterogeneity. If not already included in var_x_name_ord,
        they will be added to the list of features. Default is None.
        
    _int_cuda : Boolean (or None).
        Use CUDA based GPU if available on hardware. Default is True.
        
    _int_descriptive_stats : Boolean (or None), optional
        Print descriptive stats if _int_with_output is True.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.
        
    _int_show_plots : Boolean (or None), optional
        Execute show() command if _int_with_output is True.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.
        
    _int_dpi : Integer (or None), optional
        dpi in plots. Default (or None) is 500.
        Internal variable, change default only if you know what you do.
        
    _int_fontsize : Integer (or None), optional
        Font for legends, from 1 (very small) to 7 (very large).
        Default (or None) is 2.
        Internal variable, change default only if you know what you do.
        
    _int_no_filled_plot : Integer (or None), optional
        Use filled plot if more than _int_no_filled_plot different values.
        Default (or None) is 20.
        Internal variable, change default only if you know what you do.
        
    _int_max_cats_cont_vars : Integer (or None), optional
        Discretise continuous variables: _int_max_cats_cont_vars is maximum
        number of categories for continuous variables. This speeds up the
        programme but may introduce some bias. None: No use of
        discretisation to speed up programme. Default is None.
        Internal variable, change default only if you know what you do.
        
    _int_max_save_values : Integer (or None), optional
        Save value of features in table only if less than
        _int_max_save_values different values. Default (or None) is 50.
        Internal variable, change default only if you know what you do.
        
    _int_mp_ray_del : Tuple of strings (or None), optional
        'refs': Delete references to object store.
        'rest': Delete all other objects of Ray task.
        'none': Delete no objects.
        These 3 options can be combined. . Default  is ('refs',).
        Internal variable, change default only if you know what you do.
        
    _int_mp_ray_objstore_multiplier : Float (or None), optional
        Changes internal default values for  Ray object store. Change above
        1 if programme crashes because object store is full. Only relevant
        if _int_mp_ray_shutdown is True. Default (or None) is 1.
        Internal variable, change default only if you know what you do.
        
    _int_mp_ray_shutdown : Boolean (or None), optional
        When computing the mcf repeatedly like in Monte Carlo studies,
        setting _int_mp_ray_shutdown to True may be a good idea.
        None: False if obs < 100000, True otherwise. Default is None.
        Internal variable, change default only if you know what you do.
        
    _int_mp_vim_type : Integer (or None), optional
        Type of multiprocessing when computing variable importance
        statistics:
        1: variable based (fast, lots of memory),
        2: bootstrap based (slower, less memory)
        None: 1 if obs < 20000, 2 otherwise. Default is None.
        Internal variable, change default only if you know what you do.
        
    _int_mp_weights_tree_batch : Integer (or None), optional
        Number of batches to split data in weight computation: The smaller
        the number of batches, the faster the programme and the more memory
        is needed. None: Automatically determined. Default is None.
        Internal variable, change default only if you know what you do.
        
    _int_mp_weights_type : Integer (or None), optional
        Type of multiprocessing when computing weights:
        1: groups-of-obs based (fast, lots of memory)
        2: tree based (takes forever, less memory)
        Default (or None) is 1.
        Internal variable, change default only if you know what you do.
        Value of 2 will be internally changed to 1 if multiprocessing.
        
    _int_output_no_new_dir : Boolean
        Do not create a new directory when the path already exists. Default
        is False.
        
    _int_report : Boolean
        Provide information for McfOptPolReports to construct informative
        reports. Default is True.
        
    _int_return_iate_sp : Boolean (or None), optional
        Return all data with predictions despite _int_with_output is False
        (useful for cross-validation and simulations).
        Default (or None) is False.
        Internal variable, change default only if you know what you do.
        
    _int_replication : Boolean (or None), optional
        If True all scikit-learn based computations will NOT use multi-
        processing. Default (or None) is False.
        
    _int_seed_sample_split : Integer (or None), optional
        Seeding is redone when building forest.
        Default (or None) is 67567885.
        Internal variable, change default only if you know what you do.
        
    _int_share_forest_sample : Float (or None), optional
        Share of sample used build forest. Default (or None) is 0.5.
        Internal variable, change default only if you know what you do.
        
    _int_verbose :  Boolean (or None), optional
        Additional output about running of mcf if _int_with_output is True.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.
        
    _int_weight_as_sparse :  Boolean (or None), optional
        Save weights matrix as sparse matrix. Default (or None) is True.
        Internal variable, change default only if you know what you do.
        
    _int_weight_as_sparse_splits : Integer (or None), optional
        Compute sparse weight matrix in several chuncks.
        Default: None: 

        .. math:: 

            \\text{int}(\\frac{{\\text{Rows of prediction data} \\times \\text{rows of Fill_y data}}}{{20000 \\times 20000}})
                           
        Default is None.
        Internal variable, change default only if you know what you do.
        
    _int_with_output : Boolean (or None), optional
        Print output on txt file and/or console. Default (or None) is True.
        Internal variable, change default only if you know what you do.
        
    _int_del_forest : Boolean (or None), optional
        Delete forests from instance. If True, less memory is needed, but
        the trained instance of the class cannot be reused when calling
        predict with the same instance again, i.e. the forest has to be
        retrained when applied again. Default is False.
        
    _int_keep_w0 : Boolean (or None), optional.
        Keep all zeros weights when computing standard errors (slows down
        computation and may lead to undesirable behaviour).
        Default is False.
    """

    def __init__(
            self,
            cf_alpha_reg_grid=1, cf_alpha_reg_max=0.15, cf_alpha_reg_min=0.05,
            cf_boot=1000, cf_chunks_maxsize=None, cf_n_min_grid=1,
            cf_n_min_max=None, cf_n_min_min=None, cf_n_min_treat=None,
            cf_nn_main_diag_only=False, cf_m_grid=1, cf_m_random_poisson=True,
            cf_m_share_max=0.6, cf_m_share_min=0.1,
            cf_match_nn_prog_score=True, cf_mce_vart=1, cf_p_diff_penalty=None,
            cf_subsample_factor_eval=None, cf_subsample_factor_forest=1,
            cf_random_thresholds=None, cf_vi_oob_yes=False,
            cs_adjust_limits=None, cs_max_del_train=0.5, cs_min_p=0.01,
            cs_quantil=1, cs_type=1,
            ct_grid_dr=100, ct_grid_nn=10, ct_grid_w=10,
            dc_check_perfectcorr=True, dc_clean_data=True, dc_min_dummy_obs=10,
            dc_screen_covariates=True, fs_rf_threshold=1, fs_other_sample=True,
            fs_other_sample_share=0.33, fs_yes=False,
            gen_d_type='discrete', gen_iate_eff=False, gen_panel_data=False,
            gen_mp_parallel=None, gen_outfiletext=None, gen_outpath=None,
            gen_output_type=2, gen_panel_in_rf=True, gen_weighted=False,
            lc_cs_cv=True, lc_cs_cv_k=5, lc_cs_share=0.25,
            lc_uncenter_po=True, lc_yes=True,
            p_cbgate=False, p_ate_no_se_only=False,
            p_atet=False, p_bgate=False, p_bt_yes=True,
            p_choice_based_sampling=False, p_choice_based_probs=None,
            p_ci_level=0.90, p_cluster_std=False, p_cond_var=True,
            p_gates_minus_previous=False, p_gates_smooth=True,
            p_gates_smooth_bandwidth=1, p_gates_smooth_no_evalu_points=50,
            p_gatet=False, p_gate_no_evalu_points=50,
            p_bgate_sample_share=None, p_iate=True, p_iate_se=False,
            p_iate_m_ate=False, p_knn=True, p_knn_const=1, p_knn_min_k=10,
            p_nw_bandw=1, p_nw_kern=1, p_max_cats_z_vars=None,
            p_max_weight_share=0.05, p_se_boot_ate=None, p_se_boot_gate=None,
            p_se_boot_iate=None, var_bgate_name=None, var_cluster_name=None,
            post_bin_corr_threshold=0.1, post_bin_corr_yes=True,
            post_est_stats=True, post_kmeans_no_of_groups=None,
            post_kmeans_max_tries=1000, post_kmeans_replications=10,
            post_kmeans_yes=True, post_random_forest_vi=True,
            post_relative_to_first_group_only=True, post_plots=True,
            var_d_name=None, var_id_name=None, var_w_name=None,
            var_x_balance_name_ord=None, var_x_balance_name_unord=None,
            var_x_name_always_in_ord=None, var_x_name_always_in_unord=None,
            var_x_name_remain_ord=None, var_x_name_remain_unord=None,
            var_x_name_ord=None, var_x_name_unord=None, var_y_name=None,
            var_y_tree_name=None, var_z_name_list=None,
            var_z_name_ord=None, var_z_name_unord=None,
            _int_cuda=True, _int_del_forest=False,
            _int_descriptive_stats=True, _int_dpi=500,
            _int_fontsize=2, _int_keep_w0=False, _int_no_filled_plot=20,
            _int_max_cats_cont_vars=None, _int_max_save_values=50,
            _int_mp_ray_del=('refs',), _int_mp_ray_objstore_multiplier=1,
            _int_mp_ray_shutdown=None, _int_mp_vim_type=None,
            _int_mp_weights_tree_batch=None, _int_mp_weights_type=1,
            _int_output_no_new_dir=False, _int_red_largest_group_train=False,
            _int_replication=False, _int_report=True, _int_return_iate_sp=False,
            _int_seed_sample_split=67567885, _int_share_forest_sample=0.5,
            _int_show_plots=True, _int_verbose=True,
            _int_weight_as_sparse=True, _int_weight_as_sparse_splits=None,
            _int_with_output=True
            ):
        
        self._int_dict = mcf_init.int_init(
            cuda=_int_cuda, del_forest=_int_del_forest,
            descriptive_stats=_int_descriptive_stats, dpi=_int_dpi,
            fontsize=_int_fontsize, no_filled_plot=_int_no_filled_plot,
            max_save_values=_int_max_save_values, mp_ray_del=_int_mp_ray_del,
            mp_ray_objstore_multiplier=_int_mp_ray_objstore_multiplier,
            mp_weights_tree_batch=_int_mp_weights_tree_batch,
            mp_weights_type=_int_mp_weights_type,
            return_iate_sp=_int_return_iate_sp,
            replication=_int_replication,
            seed_sample_split=_int_seed_sample_split,
            share_forest_sample=_int_share_forest_sample,
            show_plots=_int_show_plots, verbose=_int_verbose,
            weight_as_sparse=_int_weight_as_sparse, keep_w0=_int_keep_w0,
            with_output=_int_with_output, mp_ray_shutdown=_int_mp_ray_shutdown,
            mp_vim_type=_int_mp_vim_type,
            output_no_new_dir=_int_output_no_new_dir, report=_int_report,
            weight_as_sparse_splits=_int_weight_as_sparse_splits,
            max_cats_cont_vars=_int_max_cats_cont_vars,
            p_ate_no_se_only=p_ate_no_se_only)
        gen_dict = mcf_init.gen_init(
            self.int_dict,
            d_type=gen_d_type, iate_eff=gen_iate_eff,
            mp_parallel=gen_mp_parallel, outfiletext=gen_outfiletext,
            outpath=gen_outpath, output_type=gen_output_type,
            weighted=gen_weighted, panel_data=gen_panel_data,
            panel_in_rf=gen_panel_in_rf)
        self._dc_dict = mcf_init.dc_init(
            check_perfectcorr=dc_check_perfectcorr,
            clean_data=dc_clean_data, min_dummy_obs=dc_min_dummy_obs,
            screen_covariates=dc_screen_covariates)
        self._ct_dict = {'grid_dr': ct_grid_dr, 'grid_nn': ct_grid_nn,
                        'grid_w': ct_grid_w}
        self.fs_dict = mcf_init.fs_init(
            rf_threshold=fs_rf_threshold, other_sample=fs_other_sample,
            other_sample_share=fs_other_sample_share, yes=fs_yes)
        self._cs_dict = mcf_init.cs_init(
            gen_dict,
            max_del_train=cs_max_del_train, min_p=cs_min_p, quantil=cs_quantil,
            type_=cs_type, adjust_limits=cs_adjust_limits)
        self.lc_dict = mcf_init.lc_init(
            cs_cv=lc_cs_cv, cs_cv_k=lc_cs_cv_k, cs_share=lc_cs_share,
            undo_iate=lc_uncenter_po, yes=lc_yes)
        self._cf_dict = mcf_init.cf_init(
            alpha_reg_grid=cf_alpha_reg_grid, alpha_reg_max=cf_alpha_reg_max,
            alpha_reg_min=cf_alpha_reg_min, boot=cf_boot,
            chunks_maxsize=cf_chunks_maxsize,
            nn_main_diag_only=cf_nn_main_diag_only, m_grid=cf_m_grid,
            m_share_max=cf_m_share_max, m_share_min=cf_m_share_min,
            m_random_poisson=cf_m_random_poisson,
            match_nn_prog_score=cf_match_nn_prog_score, mce_vart=cf_mce_vart,
            vi_oob_yes=cf_vi_oob_yes, n_min_grid=cf_n_min_grid,
            n_min_max=cf_n_min_max, n_min_min=cf_n_min_min,
            n_min_treat=cf_n_min_treat,
            p_diff_penalty=cf_p_diff_penalty,
            subsample_factor_eval=cf_subsample_factor_eval,
            subsample_factor_forest=cf_subsample_factor_forest,
            random_thresholds=cf_random_thresholds)
        p_dict = mcf_init.p_init(
            gen_dict,
            ate_no_se_only=p_ate_no_se_only, cbgate=p_cbgate, atet=p_atet,
            bgate=p_bgate, bt_yes=p_bt_yes,
            choice_based_sampling=p_choice_based_sampling,
            choice_based_probs=p_choice_based_probs, ci_level=p_ci_level,
            cluster_std=p_cluster_std, cond_var=p_cond_var,
            gates_minus_previous=p_gates_minus_previous,
            gates_smooth=p_gates_smooth,
            gates_smooth_bandwidth=p_gates_smooth_bandwidth,
            gates_smooth_no_evalu_points=p_gates_smooth_no_evalu_points,
            gatet=p_gatet, gate_no_evalu_points=p_gate_no_evalu_points,
            bgate_sample_share=p_bgate_sample_share,
            iate=p_iate, iate_se=p_iate_se, iate_m_ate=p_iate_m_ate, knn=p_knn,
            knn_const=p_knn_const, knn_min_k=p_knn_min_k, nw_bandw=p_nw_bandw,
            nw_kern=p_nw_kern, max_cats_z_vars=p_max_cats_z_vars,
            max_weight_share=p_max_weight_share,
            se_boot_ate=p_se_boot_ate, se_boot_gate=p_se_boot_gate,
            se_boot_iate=p_se_boot_iate)
        self._post_dict = mcf_init.post_init(
            p_dict,
            bin_corr_threshold=post_bin_corr_threshold,
            bin_corr_yes=post_bin_corr_yes, est_stats=post_est_stats,
            kmeans_no_of_groups=post_kmeans_no_of_groups,
            kmeans_max_tries=post_kmeans_max_tries,
            kmeans_replications=post_kmeans_replications,
            kmeans_yes=post_kmeans_yes, random_forest_vi=post_random_forest_vi,
            relative_to_first_group_only=post_relative_to_first_group_only,
            plots=post_plots)
        self._var_dict, self._gen_dict, self._p_dict = mcf_init.var_init(
            gen_dict, self.fs_dict, p_dict,
            bgate_name=var_bgate_name, cluster_name=var_cluster_name,
            d_name=var_d_name, id_name=var_id_name, w_name=var_w_name,
            x_balance_name_ord=var_x_balance_name_ord,
            x_balance_name_unord=var_x_balance_name_unord,
            x_name_always_in_ord=var_x_name_always_in_ord,
            x_name_always_in_unord=var_x_name_always_in_unord,
            x_name_remain_ord=var_x_name_remain_ord,
            x_name_remain_unord=var_x_name_remain_unord,
            x_name_ord=var_x_name_ord, x_name_unord=var_x_name_unord,
            y_name=var_y_name, y_tree_name=var_y_tree_name,
            z_name_list=var_z_name_list, z_name_ord=var_z_name_ord,
            z_name_unord=var_z_name_unord)
        self._blind_dict = self._sens_dict = None
        self.data_train_dict = self._var_x_type = self._var_x_values = None
        self._forest, self._time_strings = None, {}
        self.report = {'predict_list': [],   # Needed for multiple predicts
                       'analyse_list': []}

    @property
    def blind_dict(self):
        """
        Dictionary, parameters to compute (partially) blinded IATEs.
        """
        return self._blind_dict 

    @blind_dict.setter
    def blind_dict(self, value):
        self._blind_dict = value
        
    @property
    def cf_dict(self): 
        """
        Dictionary, parameters used in training the forest (directly).
        """
        return self._cf_dict 

    @cf_dict.setter
    def cf_dict(self, value):
        self._cf_dict = value
        
    @property
    def cs_dict(self): 
        """
        Dictionary, parameters used in common support adjustments.
        """
        return self._cs_dict 

    @cs_dict.setter
    def cs_dict(self, value):
        self._cs_dict = value

    @property
    def ct_dict(self): 
        """
        Dictionary, parameters used in dealing with continuous treatments.
        """
        return self._ct_dict 

    @ct_dict.setter
    def ct_dict(self, value):
        self._ct_dict = value
        
    @property
    def int_dict(self): 
        """
        Dictionary, internal parameters used in various parts of the class.
        """
        return self._int_dict
        
    @int_dict.setter
    def int_dict(self, value):
        self._int_dict = value
        
    @property
    def dc_dict(self): 
        """
        Dictionary, parameters used in data cleaning.
        """
        return self._dc_dict

    @dc_dict.setter
    def dc_dict(self, value):
        self._dc_dict = value

    @property
    def fs_dict(self): 
        """
        Dictionary, parameters used in feature selection.
        """
        return self._fs_dict

    @fs_dict.setter
    def fs_dict(self, value):
        self._fs_dict = value
        
    @property
    def forest(self): 
        """
        List of list containing the estimated causal forest.
        """
        return self._forest

    @forest.setter
    def forest(self, value):
        self._forest = value

    @property
    def gen_dict(self): 
        """
        Dictionary, general parameters used in various parts of the programme.
        """
        return self._gen_dict
        
    @gen_dict.setter
    def gen_dict(self, value):
        self._gen_dict = value

    @property
    def p_dict(self): 
        """
        Dictionary, parameters used by prediction method.
        """
        return self._p_dict

    @p_dict.setter
    def p_dict(self, value):
        self._p_dict = value
        
    @property
    def post_dict(self): 
        """
        Dictionary, parameters used in analyse method.
        """
        return self._post_dict

    @post_dict.setter
    def post_dict(self, value):
        self._post_dict = value

    @property
    def sens_dict(self): 
        """
        Dictionary, parameters used in sensitivity method.
        """
        return self._sens_dict

    @sens_dict.setter
    def sens_dict(self, value):
        self._sens_dict = value

    @property
    def time_strings(self): 
        """
        String, detailed information on how the long the different methods needed.
        """
        return self._time_strings

    @time_strings.setter
    def time_strings(self, value):
        self._time_strings = value

    @property
    def var_dict(self): 
        """
        Dictionary, variable names.
        """
        return self._var_dict

    @var_dict.setter
    def var_dict(self, value):
        self._var_dict = value
        
    @property
    def var_x_type(self): 
        """
        Dictionary, types of covariates (internal).
        """
        return self._var_x_type

    @var_x_type.setter
    def var_x_type(self, value):
        self._var_x_type = value

    @property
    def var_x_values(self): 
        """
        Dictionary, values of covariates (internal).
        """
        return self._var_x_values
        
    @var_x_values.setter
    def var_x_values(self, value):
        self._var_x_values = value
    
    def train(self, data_df):
        """
        Build the modified causal forest on the training data.

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the causal forest. It must contain information
            about outcomes, treatment, and features.

        Returns
        -------
        tree_df : DataFrame
            Dataset used to build the forest.
        fill_y_df : DataFrame
            Dataset used to populate the forest with outcomes.

        """
        time_start = time()
        # Check treatment data
        data_df, _ = mcf_data.data_frame_vars_upper(data_df)
        data_df = mcf_data.check_recode_treat_variable(self, data_df)
        # Initialise again with data information. Order of the following
        # init functions important. Change only if you know what you do.
        mcf_init.var_update_train(self, data_df)
        mcf_init.gen_update_train(self, data_df)
        mcf_init.ct_update_train(self, data_df)
        mcf_init.cs_update_train(self)            # Used updated gen_dict info
        mcf_init.cf_update_train(self, data_df)
        mcf_init.int_update_train(self)
        mcf_init.p_update_train(self)

        if self.int_dict['with_output']:
            ps.print_dic_values_all(self, summary_top=True, summary_dic=False)

        # Prepare data: Add and recode variables for GATES (Z)
        #             Recode categorical variables to prime numbers, cont. vars
        data_df = mcf_data.create_xz_variables(self, data_df, train=True)
        if self.int_dict['with_output'] and self.int_dict['verbose']:
            mcf_data.print_prime_value_corr(self.data_train_dict,
                                            self.gen_dict, summary=False)

        # Clean data and remove missings and unncessary variables
        if self.dc_dict['clean_data']:
            data_df, report = mcf_data.clean_data(
                self, data_df, train=True)
            if self.gen_dict['with_output']:
                self.report['training_obs'] = report
        if self.dc_dict['screen_covariates']:   # Only training
            (self.gen_dict, self.var_dict, self.var_x_type, self.var_x_values,
             report
             ) = mcf_data.screen_adjust_variables(self, data_df)
            if self.gen_dict['with_output']:
                self.report['removed_vars'] = report
        time_1 = time()

        # Descriptives by treatment
        if self.int_dict['descriptive_stats'] and self.int_dict['with_output']:
            ps.desc_by_treatment(self, data_df, summary=False, stage=1)

        # Feature selection
        if self.fs_dict['yes']:
            data_df, report = mcf_fs.feature_selection(self, data_df)
            if self.gen_dict['with_output']:
                self.report['fs_vars_deleted'] = report
        # Split sample for tree building and tree-filling-with-y
        tree_df, fill_y_df = mcf_data.split_sample_for_mcf(self, data_df)
        del data_df
        time_2 = time()

        # Compute Common support
        if self.cs_dict['type']:
            (tree_df, fill_y_df, rep_sh_del, obs_remain, rep_fig
             ) = mcf_cs.common_support(self, tree_df, fill_y_df, train=True)
            if self.gen_dict['with_output']:
                self.report['cs_t_share_deleted'] = rep_sh_del
                self.report['cs_t_obs_remain'] = obs_remain
                self.report['cs_t_figs'] = rep_fig

        # Descriptives by treatment on common support
        if self.int_dict['descriptive_stats'] and self.int_dict['with_output']:
            ps.desc_by_treatment(self, pd.concat([tree_df, fill_y_df], axis=0),
                                 summary=True, stage=1)
        time_3 = time()

        # Local centering
        if self.lc_dict['yes']:
            (tree_df, fill_y_df, _, report) = mcf_lc.local_centering(
                self, tree_df, fill_y_df)
            if self.gen_dict['with_output']:
                self.report["lc_r2"] = report
        time_4 = time()

        # Train forest
        if self.int_dict['with_output']:
            ps.variable_features(self, summary=False)
        (self.cf_dict, self.forest, time_vi, report) = mcf_fo.train_forest(
            self, tree_df, fill_y_df)
        if self.gen_dict['with_output']:
            self.report['cf'] = report
        time_end = time()
        time_string = ['Data preparation and stats I:                   ',
                       'Feature preselection:                           ',
                       'Common support:                                 ',
                       'Local centering (recoding of Y):                ',
                       'Training the causal forest:                     ',
                       '  ... of which is time for variable importance: ',
                       '\nTotal time training:                            ']
        time_difference = [time_1 - time_start, time_2 - time_1,
                           time_3 - time_2, time_4 - time_3,
                           time_end - time_4, time_vi,
                           time_end - time_start]
        if self.int_dict['with_output']:
            time_train = ps.print_timing(
                self.gen_dict, 'Training', time_string, time_difference,
                summary=True)
            self.time_strings['time_train'] = time_train
        return tree_df, fill_y_df

    def predict(self, data_df):
        """
        Compute all effects given a causal forest estimated with :py:meth:`~mcf_functionsModifiedCausalForest.train` method.

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the predictions. It must contain information
            about features (and treatment if effects for treatment specific
            subpopulations are desired as well).

        Returns
        -------
        results : Dictionary.
            Results. This dictionary has the following structure:
            'ate': ATE, 'ate_se': Standard error of ATE,
            'ate effect_list': List of names of estimated effects,
            'gate': GATE, 'gate_se': SE of GATE,
            'gate_diff': GATE minus ATE,
            'gate_diff_se': Standard error of GATE minus ATE,
            'cbgate': cbGATE (all covariates balanced),
            'cbgate_se': Standard error of CBGATE,
            'cbgate_diff': CBGATE minus ATE,
            'cbgate_diff_se': Standard error of CBGATE minus ATE,
            'bgate': BGATE (only prespecified covariates balanced),
            'bgate_se': Standard error of BGATE,
            'bgate_diff': BGATE minus ATE,
            'bgate_diff_se': Standard errror of BGATE minus ATE,
            'gate_names_values': Dictionary: Order of gates parameters
            and name and values of GATE effects.
            'iate': IATE, 'iate_se': Standard error of IATE,
            'iate_eff': (More) Efficient IATE (IATE estimated twice and
            averaged where role of tree_building and tree_filling
            sample is exchanged),
            'iate_data_df': DataFrame with IATEs,
            'iate_names_dic': Dictionary containing names of IATEs,
            'bala': Effects of balancing tests,
            'bala_se': Standard error of effects of balancing tests,
            'bala_effect_list': Names of effects of balancing tests
        """
        time_start = time()
        report = {}
        data_df, _ = mcf_data.data_frame_vars_upper(data_df)
        # Initialise again with data information
        data_df = mcf_init.p_update_pred(self, data_df)
        # Check treatment data
        if self.p_dict['d_in_pred']:
            data_df = mcf_data.check_recode_treat_variable(self, data_df)
        # self.var_dict = mcf_init.ct_update_pred(self)
        mcf_init.int_update_pred(self, len(data_df))
        mcf_init.post_update_pred(self, data_df)
        if self.int_dict['with_output']:
            ps.print_dic_values_all(self, summary_top=True, summary_dic=False,
                                    train=False)

        # Prepare data: Add and recode variables for GATES (Z)
        #             Recode categorical variables to prime numbers, cont. vars
        data_df = mcf_data.create_xz_variables(self, data_df, train=False)
        if self.int_dict['with_output'] and self.int_dict['verbose']:
            mcf_data.print_prime_value_corr(self.data_train_dict,
                                            self.gen_dict, summary=False)
        # Clean data and remove missings and unncessary variables
        if self.dc_dict['clean_data']:
            data_df, report['prediction_obs'] = mcf_data.clean_data(
                self, data_df, train=False)
        # Descriptives by treatment on common support
        if (self.p_dict['d_in_pred'] and self.int_dict['descriptive_stats']
                and self.int_dict['with_output']):
            ps.desc_by_treatment(self, data_df, summary=False, stage=3)
        time_1 = time()

        # Common support
        if self.cs_dict['type']:
            (data_df, _, report['cs_p_share_deleted'],
             report['cs_p_obs_remain'], _) = mcf_cs.common_support(
                 self, data_df, None, train=False)
        data_df = data_df.copy().reset_index()
        if (self.p_dict['d_in_pred'] and self.int_dict['descriptive_stats']
                and self.int_dict['with_output']):
            ps.desc_by_treatment(self, data_df, summary=True, stage=3)
        time_2 = time()

        # Local centering for IATE
        if self.lc_dict['yes'] and self.lc_dict['uncenter_po']:
            (_, _, y_pred_x_df, _) = mcf_lc.local_centering(self, data_df,
                                                            None, train=False)
        else:
            y_pred_x_df = 0
        time_3 = time()
        time_delta_weight = time_delta_ate = time_delta_bala = 0
        time_delta_iate = time_delta_gate = time_delta_cbgate = 0
        time_delta_bgate = 0
        ate_dic = bala_dic = iate_dic = iate_m_ate_dic = iate_eff_dic = None
        gate_dic = gate_m_ate_dic = cbgate_dic = cbgate_m_ate_dic = None
        bgate_dic = bgate_m_ate_dic = None
        only_one_fold_one_round = (self.cf_dict['folds'] == 1
                                   and len(self.cf_dict['est_rounds']) == 1)
        for fold in range(self.cf_dict['folds']):
            for round_ in self.cf_dict['est_rounds']:
                time_w_start = time()
                if only_one_fold_one_round:
                    forest_dic = self.forest[fold][0]
                else:
                    forest_dic = deepcopy(
                        self.forest[fold][0 if round_ == 'regular' else 1])
                if self.int_dict['with_output'] and self.int_dict['verbose']:
                    print(f'\n\nWeight maxtrix {fold+1} /',
                          f'{self.cf_dict["folds"]} forests, {round_}')
                weights_dic = mcf_w.get_weights_mp(
                    self, data_df, forest_dic, round_ == 'regular')
                time_delta_weight += time() - time_w_start
                time_a_start = time()
                if round_ == 'regular':
                    # Estimate ATE n fold
                    (w_ate, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
                        self, data_df, weights_dic)
                    # Aggregate ATEs over folds
                    ate_dic = mcf_est.aggregate_pots(
                        self, y_pot_f, y_pot_var_f, txt_w_f, ate_dic, fold,
                        title='ATE')
                else:
                    w_ate = None
                time_delta_ate += time() - time_a_start
                # Compute balancing tests
                time_b_start = time()
                if round_ == 'regular':
                    if self.p_dict['bt_yes']:
                        (_, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
                            self, data_df, weights_dic, balancing_test=True)
                        # Aggregate Balancing results over folds
                        bala_dic = mcf_est.aggregate_pots(
                            self, y_pot_f, y_pot_var_f, txt_w_f, bala_dic,
                            fold, title='Balancing check: ')
                time_delta_bala += time() - time_b_start

                # BGATE
                time_bgate_start = time()
                if round_ == 'regular' and self.p_dict['bgate']:
                    (y_pot_bgate_f, y_pot_var_bgate_f, y_pot_mate_bgate_f,
                     y_pot_mate_var_bgate_f, bgate_est_dic, txt_w_f, txt_b,
                     ) = mcf_gate.bgate_est(self, data_df, weights_dic,
                                            w_ate, forest_dic,
                                            gate_type='BGATE')
                    bgate_dic = mcf_est.aggregate_pots(
                        self, y_pot_bgate_f, y_pot_var_bgate_f, txt_w_f,
                        bgate_dic, fold, pot_is_list=True, title='BGATE')
                    if y_pot_mate_bgate_f is not None:
                        bgate_m_ate_dic = mcf_est.aggregate_pots(
                            self, y_pot_mate_bgate_f, y_pot_mate_var_bgate_f,
                            txt_w_f, bgate_m_ate_dic, fold, pot_is_list=True,
                            title='BGATE minus ATE')
                time_delta_bgate += time() - time_bgate_start

                # CBGATE
                time_cbg_start = time()
                if round_ == 'regular' and self.p_dict['cbgate']:
                    (y_pot_cbgate_f, y_pot_var_cbgate_f, y_pot_mate_cbgate_f,
                     y_pot_mate_var_cbgate_f, cbgate_est_dic, txt_w_f, txt_am,
                     ) = mcf_gate.bgate_est(self, data_df, weights_dic,
                                            w_ate, forest_dic,
                                            gate_type='CBGATE')
                    cbgate_dic = mcf_est.aggregate_pots(
                        self, y_pot_cbgate_f, y_pot_var_cbgate_f, txt_w_f,
                        cbgate_dic, fold, pot_is_list=True, title='CBGATE')
                    if y_pot_mate_cbgate_f is not None:
                        cbgate_m_ate_dic = mcf_est.aggregate_pots(
                            self, y_pot_mate_cbgate_f, y_pot_mate_var_cbgate_f,
                            txt_w_f, cbgate_m_ate_dic, fold, pot_is_list=True,
                            title='CBGATE minus ATE')
                time_delta_cbgate += time() - time_cbg_start
                if self.int_dict['del_forest']:
                    del forest_dic['forest']

                # IATE
                time_i_start = time()
                if self.p_dict['iate']:
                    y_pot_eff = None
                    (y_pot_f, y_pot_var_f, y_pot_m_ate_f, y_pot_m_ate_var_f,
                     txt_w_f) = mcf_iate.iate_est_mp(
                         self, weights_dic, w_ate, round_ == 'regular')
                    if round_ == 'regular':
                        y_pot_iate_f = y_pot_f.copy()
                        y_pot_varf = (None if y_pot_var_f is None
                                      else y_pot_var_f.copy())
                        iate_dic = mcf_est.aggregate_pots(
                            self, y_pot_iate_f, y_pot_varf, txt_w_f, iate_dic,
                            fold, title='IATE')
                        if y_pot_m_ate_f is not None:
                            iate_m_ate_dic = mcf_est.aggregate_pots(
                                self, y_pot_m_ate_f, y_pot_m_ate_var_f,
                                txt_w_f, iate_m_ate_dic, fold,
                                title='IATE minus ATE')
                    else:
                        y_pot_eff = (y_pot_iate_f + y_pot_f) / 2
                        iate_eff_dic = mcf_est.aggregate_pots(
                            self, y_pot_eff, None, txt_w_f, iate_eff_dic, fold,
                            title='IATE eff')
                time_delta_iate += time() - time_i_start

                # GATE
                time_g_start = time()
                if round_ == 'regular' and self.p_dict['gate']:
                    (y_pot_gate_f, y_pot_var_gate_f, y_pot_mate_gate_f,
                     y_pot_mate_var_gate_f, gate_est_dic, txt_w_f
                     ) = mcf_gate.gate_est(self, data_df, weights_dic, w_ate)
                    gate_dic = mcf_est.aggregate_pots(
                        self, y_pot_gate_f, y_pot_var_gate_f, txt_w_f,
                        gate_dic, fold, pot_is_list=True, title='GATE')
                    if y_pot_mate_gate_f is not None:
                        gate_m_ate_dic = mcf_est.aggregate_pots(
                            self, y_pot_mate_gate_f, y_pot_mate_var_gate_f,
                            txt_w_f, gate_m_ate_dic, fold, pot_is_list=True,
                            title='GATE minus ATE')
                time_delta_gate += time() - time_g_start
            if not only_one_fold_one_round and self.int_dict['del_forest']:
                self.forest[fold] = None
                # Without those two deletes, it becomes impossible to reuse
                # the same forest for several data sets, which is bad.
        if self.int_dict['del_forest']:
            self.forest = None
        del weights_dic

        # ATE
        time_a_start = time()
        ate, ate_se, ate_effect_list = mcf_ate.ate_effects_print(
            self, ate_dic, y_pred_x_df, balancing_test=False)
        time_delta_ate += time() - time_a_start

        # GATE
        time_g_start = time()
        if self.p_dict['gate']:
            (gate, gate_se, gate_diff, gate_diff_se, report['fig_gate']
             ) = mcf_gateout.gate_effects_print(self, gate_dic, gate_m_ate_dic,
                                                gate_est_dic, ate, ate_se,
                                                gate_type='GATE')
        else:
            gate = gate_se = gate_diff = gate_diff_se = gate_est_dic = None
        time_delta_gate += time() - time_g_start

        # BGATE
        time_bgate_start = time()
        if self.p_dict['bgate']:
            (bgate, bgate_se, bgate_diff, bgate_diff_se, report['fig_bgate']
             ) = mcf_gateout.gate_effects_print(
                 self, bgate_dic, bgate_m_ate_dic, bgate_est_dic, ate,
                 ate_se, gate_type='BGATE', special_txt=txt_b)
        else:
            bgate = bgate_se = bgate_diff = bgate_diff_se = None
            bgate_est_dic = None
        time_delta_bgate += time() - time_bgate_start

        # CBGATE
        time_cbg_start = time()
        if self.p_dict['cbgate']:
            (cbgate, cbgate_se, cbgate_diff, cbgate_diff_se,
             report['fig_cbgate']) = mcf_gateout.gate_effects_print(
                 self, cbgate_dic, cbgate_m_ate_dic, cbgate_est_dic, ate,
                 ate_se, gate_type='CBGATE', special_txt=txt_am)
        else:
            cbgate = cbgate_se = cbgate_diff = cbgate_diff_se = None
            cbgate_est_dic = None
        time_delta_cbgate += time() - time_cbg_start
        # Collect some information for results_dic
        if (self.p_dict['gate'] or self.p_dict['bgate']
                or self.p_dict['cbgate']):
            gate_names_values = mcf_gateout.get_names_values(
                self, gate_est_dic, bgate_est_dic, cbgate_est_dic)
        else:
            gate_names_values = None
        # IATE
        time_i_start = time()
        if self.p_dict['iate']:
            (iate, iate_se, iate_eff, iate_names_dic, iate_df,
             report['iate_text']) = mcf_iate.iate_effects_print(
                 self, iate_dic, iate_m_ate_dic, iate_eff_dic, y_pred_x_df)
            data_df.reset_index(drop=True, inplace=True)
            iate_df.reset_index(drop=True, inplace=True)
            iate_pred_df = pd.concat([data_df, iate_df], axis=1)
        else:
            iate_eff = iate = iate_se = iate_df = iate_pred_df = None
            iate_names_dic = None
        time_delta_iate += time() - time_i_start

        # Balancing test
        time_b_start = time()
        if self.p_dict['bt_yes']:
            bala, bala_se, bala_effect_list = mcf_ate.ate_effects_print(
                self, bala_dic, None, balancing_test=True)
        else:
            bala = bala_se = bala_effect_list = None
        time_delta_bala += time() - time_b_start
        # Collect results
        results = {
            'ate': ate, 'ate_se': ate_se, 'ate effect_list': ate_effect_list,
            'gate': gate, 'gate_se': gate_se,
            'gate_diff': gate_diff, 'gate_diff_se': gate_diff_se,
            'gate_names_values': gate_names_values,
            'cbgate': cbgate, 'cbgate_se': cbgate_se,
            'cbgate_diff': cbgate_diff, 'cbgate_diff_se': cbgate_diff_se,
            'bgate': bgate, 'bgate_se': bgate_se,
            'bgate_diff': bgate_diff, 'bgate_diff_se': bgate_diff_se,
            'iate': iate, 'iate_se': iate_se, 'iate_eff': iate_eff,
            'iate_data_df': iate_pred_df, 'iate_names_dic': iate_names_dic,
            'bala': bala, 'bala_se': bala_se, 'bala_effect_list':
                bala_effect_list
                   }
        if self.int_dict['with_output']:
            results_dic = results.copy()
            del results_dic['iate_data_df']
            report['mcf_pred_results'] = results
        self.report['predict_list'].append(report.copy())
        time_end = time()
        if self.int_dict['with_output']:
            time_string = [
                'Data preparation and stats II:                  ',
                'Common support:                                 ',
                'Local centering (recoding of Y):                ',
                'Weights:                                        ',
                'ATEs:                                           ',
                'GATEs:                                          ',
                'BGATEs:                                         ',
                'CBGATEs:                                        ',
                'IATEs:                                          ',
                'Balancing test:                                 ',
                '\nTotal time prediction:                          ']
            time_difference = [
                time_1 - time_start, time_2 - time_1, time_3 - time_2,
                time_delta_weight, time_delta_ate, time_delta_gate,
                time_delta_bgate, time_delta_cbgate, time_delta_iate,
                time_delta_bala, time_end - time_start]
            ps.print_mcf(self.gen_dict, self.time_strings['time_train'])
            time_pred = ps.print_timing(
                self.gen_dict, 'Prediction', time_string, time_difference,
                summary=True)
            self.time_strings['time_pred'] = time_pred
        return results

    def analyse(self, results):
        """
        Analyse estimated IATE with various descriptive tools.

        Parameters
        ----------
        results : Dictionary
            Contains estimation results. This dictionary must have the same
            structure as the one returned from the predict method.

        Raises
        ------
        ValueError
            Some of the attribute are not compatible with running this method.

        Returns
        -------
        results_plus_cluster : Dictionary
            Same as the results dictionary, but the DataFrame with estimated
            IATEs contains an additional integer with a group label that comes
            from k-means clustering.

        """
        report = {}
        if (self.int_dict['with_output'] and self.post_dict['est_stats'] and
                self.int_dict['return_iate_sp']):
            time_start = time()
            report['fig_iate'] = mcf_post.post_estimation_iate(self, results)
            time_end_corr = time()
            if self.post_dict['kmeans_yes']:
                (results_plus_cluster, report['knn_table']
                 ) = mcf_post.k_means_of_x_iate(self, results)
            else:
                report['knn_table'] = None
            time_end_km = time()
            if self.post_dict['random_forest_vi']:
                mcf_post.random_forest_of_iate(self, results)
            time_string = [
                'Correlational analysis and plots of IATE:       ',
                'K-means clustering of IATE:                     ',
                'Random forest analysis of IATE:                 ',
                '\nTotal time post estimation analysis:            ']
            time_difference = [
                time_end_corr - time_start, time_end_km - time_end_corr,
                time() - time_end_km, time() - time_start]
            ps.print_mcf(self.gen_dict, self.time_strings['time_train'],
                         summary=True)
            ps.print_mcf(self.gen_dict, self.time_strings['time_pred'],
                         summary=True)
            ps.print_timing(self.gen_dict, 'Analysis of IATE', time_string,
                            time_difference, summary=True)
            self.report['analyse_list'].append(report.copy())
        else:
            raise ValueError(
                '"Analyse" method produces output only if all of the following'
                ' parameters are True:'
                f'\nint_with_output: {self.int_dict["with_output"]}'
                f'\npos_test_stats: {self.post_dict["est_stats"]}'
                f'\nint_return_iate_sp: {self.int_dict["return_iate_sp"]}')
        return results_plus_cluster

    def blinder_iates(
        self, data_df, blind_var_x_protected_name=None,
        blind_var_x_policy_name=None, blind_var_x_unrestricted_name=None,
        blind_weights_of_blind=None, blind_obs_ref_data=50,
            blind_seed=123456):
        """
        Compute IATEs that causally depend less on protected variables.

        Parameters
        ----------
        data_df : DataFrame.
            Contains data needed to predict the various adjusted IATES.
        blind_var_x_protected_name : List of strings (or None), optional
            Names of protected variables. Names that are
            explicitly denoted as blind_var_x_unrestricted_name or as
            blind_var_x_policy_name and used to compute IATEs will be
            automatically added to this list. Default is None.
        blind_var_x_policy_name : List of strings (or None), optional
            Names of decision variables. Default is None.
        blind_var_x_unrestricted_name : List of strings (or None), optional
            Names of unrestricted variables. Default is None.
        blind_weights_of_blind : Tuple of float (or None), optional
            Weights to compute weighted means of blinded and unblinded IATEs.
            Between 0 and 1. 1 implies all weight goes to fully blinded IATEs.
        blind_obs_ref_data : Integer (or None), optional
            Number of observations to be used for blinding. Runtime of
            programme is almost linear in this parameter. Default is 50.
        blind_seed : Integer, optional.
            Seed for the random selection of the reference data.
            Default is 123456.

        Returns
        -------
        blinded_dic : Dictionary.
            Contains potential outcomes that do not fully
            depend on some protected attributes (and non-modified IATE).
        data_on_support_df : DataFrame.
            Features that are on the common support.
        var_x_policy_ord_name : List of strings, optional
            Ordered variables to be used to build the decision rules.
        var_x_policy_unord_name : List of strings.
            Unordered variables to be used to build the decision rules.
        var_x_blind_ord_name : List of strings, optional
            Ordered variables to be used to blind potential outcomes.
        var_x_blind_unord_name : List of strings.
            Unordered variables to be used to blind potential outcomes.
        """
        self.blind_dict = mcf_init.blind_init(
            var_x_protected_name=blind_var_x_protected_name,
            var_x_policy_name=blind_var_x_policy_name,
            var_x_unrestricted_name=blind_var_x_unrestricted_name,
            weights_of_blind=blind_weights_of_blind,
            obs_ref_data=blind_obs_ref_data,
            seed=blind_seed)

        if self.int_dict['with_output']:
            time_start = time()
        with_output = self.int_dict['with_output']

        (blinded_dic, data_on_support_df, var_x_policy_ord_name,
         var_x_policy_unord_name, var_x_blind_ord_name, var_x_blind_unord_name
         ) = mcf_fair.make_fair_iates(self, data_df, with_output=with_output)

        self.int_dict['with_output'] = with_output
        if self.int_dict['with_output']:
            time_difference = [time() - time_start]
            time_string = ['Total time for blinding IATEs:                  ']
            ps.print_timing(self.gen_dict, 'Blinding IATEs', time_string,
                            time_difference, summary=True)
        return (blinded_dic, data_on_support_df, var_x_policy_ord_name,
                var_x_policy_unord_name, var_x_blind_ord_name,
                var_x_blind_unord_name)

    def sensitivity(self, train_df, predict_df=None, results=None,
                    sens_cbgate=None, sens_bgate=None, sens_gate=None,
                    sens_iate=None, sens_iate_se=None, sens_scenarios=None,
                    sens_cv_k=None, sens_replications=2,
                    sens_reference_population=None):
        """
        Compute simulation based sensitivity indicators.

        Parameters
        ----------
        train_df : DataFrame.
            Data with real outcomes, treatments, and covariates. Data will be
            transformed to compute sensitivity indicators.
        predict_df : DataFrame (or None), optinal.
            Prediction data to compute all effects for. This data will not be
            changed in the computation process. Only covariate information is
            used from this dataset. If predict_df is not a DataFrame,
            train_df will be used instead.
        results : dictionary, optional.
            The standard output dictionary from the predict method is expected.
            If this dictionary contains estimated IATEs, the same data as in
            the predict method will be used, IATEs are computed under the no
            effect (basic) scenario and these IATEs are compared to the IATEs
            contained in the results dictionary. If the dictionary does not
            contain estimated IATEs, passing it has no consequence.
        sens_cbgate : Boolean (or None), optional
            Compute CBGATEs for sensitivity analysis. Default is False.
        sens_bgate : Boolean (or None), optional
            Compute BGATEs for sensitivity analysis. Default is False.
        sens_gate : Boolean (or None), optional
            Compute GATEs for sensitivity analysis. Default is False.
        sens_iate : Boolean (or None), optional
            Compute IATEs for sensitivity analysis. If the results dictionary
            is passed, and it contains IATEs, then the default value is True,
            and False otherwise.
        sens_iate_se : Boolean (or None), optional
            Compute Standard errors of IATEs for sensitivity analysis. Default
            is False.
        sens_scenarios : List or tuple of strings, optional.
            Different scenarios considered. Default is ('basic',).
            'basic' : Use estimated treatment probabilities for simulations.
            No confounding.
        sens_cv_k : Integer (or None), optional
            Data to be used for any cross-validation: Number of folds in
            cross-validation. Default (or None) is 5.
        sens_replications : Integer (or None), optional.
            Number of replications for simulating placebo treatments. Default
            is 2.
        sens_reference_population: integer or float (or None)
            Defines the treatment status of the reference population used by
            the sensitivity analysis. Default is to use the treatment with most
            observed observations.

        Returns
        -------
        results_avg : Dictionary
            Same content as for the predict method but (if applicable) averaged
            over replications.

        """
        if (isinstance(results, dict)
            and 'iate_data_df' in results and 'iate_names_dic' in results
                and isinstance(results['iate_data_df'], pd.DataFrame)):
            predict_df = results['iate_data_df']
            iate_df = predict_df[results['iate_names_dic'][0]['names_iate']]
        else:
            iate_df = None
        if not isinstance(predict_df, pd.DataFrame):
            predict_df = train_df.copy()
        self.sens_dict = mcf_init.sens_init(
            self.p_dict, cbgate=sens_cbgate, bgate=sens_bgate, gate=sens_gate,
            iate=sens_iate, iate_se=sens_iate_se, scenarios=sens_scenarios,
            cv_k=sens_cv_k, replications=sens_replications,
            reference_population=sens_reference_population, iate_df=iate_df)
        if self.int_dict['with_output']:
            time_start = time()
        results_avg, plots_iate, txt_ate = mcf_sens.sensitivity_analysis(
            self, train_df, predict_df, self.int_dict['with_output'], iate_df,
            seed=9345467)
        self.report['sens_plots_iate'] = plots_iate
        self.report['sens_txt_ate'] = txt_ate
        if self.int_dict['with_output']:
            time_difference = [time() - time_start]
            time_string = ['Total time for sensitivity analysis:            ']
            ps.print_timing(self.gen_dict, 'Sensitiviy analysis', time_string,
                            time_difference, summary=True)
        return results_avg
