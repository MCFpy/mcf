from ray import is_initialized, shutdown
from pandas import DataFrame

from mcf.mcf_general import check_reduce_dataframe
from mcf.mcf_iv_functions import train_iv_main, predict_iv_main
from mcf import mcf_init_functions as mcf_init

from mcf.mcf_inf_for_alloc_functions import predict_different_allocations_main
from mcf.mcf_print_stats_functions import print_mcf
from mcf.mcf_sensitivity_functions import sensitivity_main
from mcf.mcf_unconfound_functions import train_main, predict_main, analyse_main
from mcf.mcf_unconfound_functions import blinder_iates_main

class ModifiedCausalForest:
    """
    Estimation of treatment effects with the Modified Causal Forest.

    Parameters
    ----------
    var_y_name : String or List of strings (or None), optional
        Name of outcome variables. If several variables are specified,
        either var_y_tree_name is used for tree building, or (if
        var_y_tree_name is None), the 1st variable in the list is used.
        Only necessary for :meth:`~ModifiedCausalForest.train` method.
        Default is None.

    var_d_name : String or List of string (or None), optional
        Name of treatment variable. Must be provided to use
        the :meth:`~ModifiedCausalForest.train` method. Can be provided for the
        :meth:`~ModifiedCausalForest.predict` method.

    var_x_name_ord : String or List of strings (or None), optional
        Name of ordered features (including dummy variables).
        Either ordered or unordered features must be provided. Default is None.

    var_x_name_unord : String or List of strings (or None), optional
        Name of unordered features. Either ordered or unordered features
        must be provided.
        Default is None.

    var_x_name_balance_bgate :  String or List of strings (or None), optional
        Variables to balance the GATEs on. Only relevant if p_bgate is
        True. The distribution of these variables is kept constant when a
        BGATE is computed. None: Use the other heterogeneity variables
        (var_z_...) (if there are any) for balancing. Default is None.

    var_cluster_name :  String or List of string (or None), optional
        Name of variable defining clusters. Only relevant if p_cluster_std
        is True.
        Default is None.

    var_id_name : String or List of string (or None), optional
        Name of identifier. None: Identifier will be added to the data.
        Default is None.

    var_iv_name : String or List of string (or None), optional
        Name of binary instrumental variable. Only relevant if train_iv method
        is used.
        Default is None.

    var_x_name_balance_test_ord : String or List of strings (or None), optional
        Name of ordered variables to be used in balancing tests. Only
        relevant if p_bt_yes is True.
        Default is None.

    var_x_name_balance_test_unord : String or List of strings (or None),
        optional
        Name of ordered variables to be used in balancing tests. Treatment
        specific descriptive statistics are only printed for those
        variables.
        Default is None.

    var_x_name_always_in_ord : String or List of strings (or None), optional
        Name of ordered variables that are always checked on when deciding on
        the next split during tree building. Only relevant for
        :meth:`~ModifiedCausalForest.train` method.
        Default is None.

    var_x_name_always_in_unord : String or List of strings (or None),
                                 optional
        Name of unordered variables that always checked on when deciding on
        the next split during tree building. Only relevant for
        :meth:`~ModifiedCausalForest.train`  method.
        Default is None.

    var_x_name_remain_ord : String or List of strings (or None), optional
        Name of ordered variables that cannot be removed by feature
        selection. Only relevant for
        :meth:`~ModifiedCausalForest.train` method.
        Default is None.

    var_x_name_remain_unord : String or List of strings (or None), 
                              optional
        Name of unordered variables that cannot be removed by feature
        selection. Only relevant for :meth:`~ModifiedCausalForest.train` method.
        Default is None.

    var_w_name : String or List of string (or None), optional
        Name of weight. Only relevant if gen_weighted is True.
        Default is None.

    var_z_name_list : String or List of strings (or None), optional
        Names of ordered variables with many values to define
        causal heterogeneity. They will be discretized and (dependening
        p_gates_smooth) also treated as continuous. If not already included
        in var_x_name_ord, they will be added to the list of features.
        Default is None.

    var_z_name_ord : String or List of strings (or None), optional
        Names of ordered variables with not so many values to define causal
        heterogeneity. If not already included in var_x_name_ord, they will
        be added to the list of features.
        Default is None.

    var_z_name_unord : String or List of strings (or None), optional
        Names of unordered variables with not so many values to define
        causal heterogeneity. If not already included in var_x_name_ord,
        they will be added to the list of features.
        Default is None.

    var_y_tree_name : String or List of string (or None), optional
        Name of outcome variables to be used to build trees. Only
        relevant if multiple outcome variables are specified in var_y_name.
        Only relevant for :meth:`~ModifiedCausalForest.train` method.
        Default is None.

    cf_alpha_reg_grid : Integer (or None), optional
        Minimum remaining share when splitting leaf: Number of grid values.
        If grid is used, optimal value is determined by out-of-bag
        estimation of objective function.
        Default (or None) is 1.

    cf_alpha_reg_max : Float (or None), optional
        Minimum remaining share when splitting leaf: Largest value of
        grid (keep it below 0.2).
        Default (or None) is 0.15.

    cf_alpha_reg_min : Float (or None), optional
        Minimum remaining share when splitting leaf: Smallest value of
        grid (keep it below 0.2).
        Default (or None) is 0.05.

    cf_boot : Integer (or None), optional
        Number of Causal Trees.
        Default (or None) is 1000.

    cf_chunks_maxsize : Integer (or None), optional
        For large samples, randomly split the training data into equally sized
        chunks, train a forest in each chunk, and estimate effects for each
        forest. Final effect estimates are obtained by averaging effects
        obtained for each forest. This procedures improves scalability by
        reducing computation time (at the possible price of a somewhat larger
        finite sample bias).
        If cf_chunks_maxsize is larger than the sample size, there is no random
        splitting.
        The default (None) is dependent on the size of the training data:
        If there are less than 90'000 training observations: No splitting.
        Otherwise:
        
        .. math::
        
            \\text{cf_chunks_maxsize} = 90000 + \\frac{{(\\text{number of observations} - 90000)^{0.8}}}{{(\\text{# of treatments} - 1)}}

        Default is None.

    cf_compare_only_to_zero : Boolean (or None), optional
       If True, the computation of the MCE ignores all elements not
       related to the first treatment (which usually is the control group). This
       speeds up computation, should give better effect estimates, and may
       be attractive when interest is only in the comparisons of each
       treatment to the control group and not among each other. This may also
       be attractive for optimal policy analysis based on using estimated
       potential outcomes normalized by the estimated potential outcome of the
       control group (i.e., IATEs of treatments vs. control group).
       Default (or None) is False.

    cf_n_min_grid : Integer (or None), optional
        Minimum leaf size: Number of grid values.
        If grid is used, optimal value is determined by out-of-bag
        estimation of objective function.
        Default (or None) is 1.

    cf_n_min_max : Integer (or None), optional
        Minimum leaf size: Largest minimum leaf size.
        If None :

        .. math::

            \\text{A} = \\frac{\\sqrt{\\text{number of observations in the smallest treatment group}}^{0.5}}{10}, \\text{at least 2} 

        :math:`\\text{cf_n_min_max} = \\text{round}(A \\times \\text{number of treatments})`
        Default is None.

    cf_n_min_min : Integer (or None), optional
        Minimum leaf size: Smallest minimum leaf size.
        If None:

        .. math::

            \\text{A} = \\text{number of observations in smallest treatment group}^{0.4} / 10, \\text{at least 1.5} 

        :math:`\\text{cf_n_min_min} = \\text{round}(A \\times \\text{number of treatments})`
        Default is None.

    cf_n_min_treat : Integer (or None), optional
        Minimum number of observations per treatment in leaf.
        A higher value reduces the risk that a leaf cannot be filled with
        outcomes from all treatment arms in the evaluation subsample.
        There is no grid based tuning for this parameter.
        This parameter impacts the minimum leaf size which will be at least
        to :math:`\\text{n_min_treat} \\times \\text{number of treatments}`
        None :

        .. math::

            \\frac{\\frac{{\\text{n_min_min}} + {\\text{n_min_max}}}{2}}{\\text{number of treatments} \\times 10}, \\text{at least 1} 

        Default is None.

    cf_match_nn_prog_score : Boolean (or None), optional
        Choice of method of nearest neighbour matching.
        True : Prognostic scores. False: Inverse of covariance matrix of
        features.
        Default (or None) is True.

    cf_nn_main_diag_only : Boolean (or None), optional
        Nearest neighbour matching: Use main diagonal of covariance matrix
        only. Only relevant if match_nn_prog_score == False.
        Default (or None) is False.

    cf_m_grid : Integer (or None), optional
        Number of variables used at each new split of tree: Number of grid
        values.
        If grid is used, optimal value is determined by out-of-bag
        estimation of objective function.
        Default (or None) is 1.

    cf_m_random_poisson : Boolean (or None), optional
        Number of variables used at each new split of tree:
        True : Number of randomly selected variables is stochastic for each
        split, drawn from a Poisson distribution. Grid gives mean
        value of 1 + poisson distribution (m-1) (m is determined by
        cf_m_share parameters).
        False : No additional randomisation.
        Default (or None) is True.

    cf_m_share_max : Float (or None), optional
        Share of variables used at each new split of tree: Maximum.
        Default (or None) is 0.6.
        If variables randomly selected for splitting do not show any variation
        in leaf considered for splitting, then all variables will be used for
        that split.

    cf_m_share_min : Float (or None), optional
        Share of variables used at each new split of tree: Minimum.
        Default (or None) is 0.1.
        If variables randomly selected for splitting do not show any variation
        in leaf considered for splitting, then all variables will be used for
        that split.

    cf_mce_vart : Integer (or None), optional
        Splitting rule for tree building:
        0 : mse's of regression only considered.
        1 : mse+mce criterion (default).
        2 : -var(effect): heterogeneity maximising splitting rule of
        Wager & Athey (2018).
        3 : randomly switching between outcome-mse+mce criterion
        & penalty functions.
        Default (or None) is 1.

    cf_p_diff_penalty : Integer (or None), optional
        Penalty function (depends on the value of `mce_vart`).

        `mce_vart == 0`
            Irrelevant (no penalty).
    
        `mce_vart == 1`
            Multiplier of penalty (in terms of `var(y)`).
            0 : No penalty.
            None :
    
            .. math::
    
                \\frac{2 \\times (\\text{n} \\times \\text{subsam_share})^{0.9}}{\\text{n} \\times \\text{subsam_share}} \\times \\sqrt{\\frac{\\text{no_of_treatments} \\times (\\text{no_of_treatments} - 1)}{2}}
    
        `mce_vart == 2`
            Multiplier of penalty (in terms of MSE(y) value function without splits) for penalty.  
            0 : No penalty.
            None :
    
            .. math::
    
                \\frac{100 \\times 4 \\times (n \\times \\text{f_c.subsam_share})^{0.8}}{n \\times \\text{f_c.subsam_share}}
    
        `mce_vart == 3`
            Probability of using p-score (0-1). None : 0.5. Increase value if balancing tests indicate problems. 
            Default is None.
    
    cf_penalty_type : String (or None), optional
        Type of penalty function.
        'mse_d':  MSE of treatment prediction in daughter leaf (new in 0.7.0)
        'diff_d': Penalty as squared leaf difference (as in Lechner, 2018)
        Note that an important advantage of 'mse_d' that it can also be used
        for tuning (due to its computation, this is not possible for 'diff_d').
        Default (or None) is 'mse_d'.

    cf_random_thresholds : Integer (or None), optional
        Use only a random selection of values for splitting (continuous
        feature only; re-randomize for each splitting decision; fewer
        thresholds speeds up programme but may lead to less accurate
        results).
        0 : No random thresholds.
        > 0 : Number of random thresholds used for ordered variables.
        None :
        :math:`4 + \\text{number of training observations}^{0.2}`
        Default is None.

    cf_subsample_factor_forest : Float (or None), optional
        Multiplier of default size of subsampling sample (S) used to build
        tree.

        .. math::

            S = \\max((n^{0.5},min(0.67 \\n, \\frac{2 \\times (n^{0.85})}{n}))), \\text{n: # of training observations} 

        :math:`S \\times \\text{cf_subsample_factor_forest}, \\text{is not larger than 80%.}` 
        Default (or None) is 1.

    cf_subsample_factor_eval : Float or Boolean (or None), optional
        Size of subsampling sample used to populate tree.
        False: No subsampling in evaluation subsample.
        True or None: :math:(2 \\times \\text{subsample size}) used for
        tree building (to avoid too many empty leaves).
        Float (>0): Multiplier of subsample size used for tree building.
        In particular for larger samples, using subsampling in evaluation
        will speed up computations and reduces demand on memory.
        Tree-specific subsampling in evaluation sample increases speed
        at which the asymtotic bias disappears (at the expense of a slower
        disappearance of the variance; however, simulations so far show no
        relevant impact).
        Default is None.

    cf_tune_all : Boolean (or None), optional
        Tune all parameters. If True, all *_grid keywords will be set to 3.
        User specified values are respected if larger than 3.
        Default (or None) is False.

    cf_vi_oob_yes : Boolean (or None), optional
        Variable importance for causal forest computed by permuting
        single variables and comparing share of increase in objective
        function of mcf (computed with out-of-bag data).
        Default (or None) is False.

    cs_type : Integer (or None), optional
        Common support adjustment: Method.
        0 : No common support adjustment.
        1,2 : Support check based on estimated classification forests.
        1 : Min-max rules for probabilities in treatment subsamples.
        2 : Enforce minimum and maximum probabilities for all obs
        all but one probability.
        Observations off support are removed. Out-of-bag predictions
        are used to avoid overfitting (which would lead to a too
        large reduction in the number of observations).
        Default (or None) is 1.

    cs_adjust_limits : Float (or None), optional
        Common support adjustment: Accounting for multiple treatments.
        None :
        :math:`(\\text{number of treatments} - 2) \\times 0.05`
        If cs_type > 0:
        :math:`\\text{upper limit} \\times = 1 + \\text{support_adjust_limits}`,
        :math:`\\text{lower limit} \\times = 1 - \\text{support_adjust_limits}`.
        The restrictiveness of the common support criterion increases with
        the number of treatments. This parameter allows to reduce this
        restrictiveness.
        Default is None.

    cs_max_del_train : Float (or None), optional
        Common support adjustment: If share of observations in training
        data used that are off support is larger than cs_max_del_train
        (0-1), an exception is raised. In this case, user should change
        input data.
        Default (or None) is 0.5.

    cs_min_p : Float (or None), optional
        Common support adjustment: If cs_type == 2, observations are
        deleted if :math:`p(d=m|x)` is less or equal than cs_min_p for at least
        one treatment. Default (or None) is 0.01.

    cs_quantil : Float (or None), optional
        Common support adjustment: How to determine upper and lower bounds.
        If CS_TYPE == 1: 1 or None : Min-max rule.
        < 1 : Respective quantile.
        Default (or None) is 1.

    ct_grid_dr : Integer (or None), optional
        Number of grid point for discretization of continuous treatment
        (with 0 mass point; grid is defined in terms of quantiles of
        continuous part of treatment) for dose response function.
        Default (or None) is 100.

    ct_grid_nn : Integer (or None), optional
        Number of grid point for discretization of continuous treatment
        (with 0 mass point; grid is defined in terms of quantiles of
        continuous part of treatment) for neighbourhood matching.
        Default (or None) is 10.

    ct_grid_w : Integer (or None), optional
        Number of grid point for discretization of continuous treatment
        (with 0 mass point; grid is defined in terms of quantiles of
        continuous part of treatment) for weights.
        Default (or None) is 10.

    dc_clean_data : Boolean (or None), optional
        Clean covariates. Remove all rows with missing observations and
        unnecessary variables from DataFrame.
        Default (or None) is True.

    dc_check_perfectcorr : Boolean (or None), optional
        Screen and clean covariates: Variables that are perfectly
        correlated with each others will be deleted.
        Default (or None) is True.

    dc_min_dummy_obs : Integer (or None), optional
        Screen covariates: If > 0 dummy variables with
        less than dc_min_dummy_obs observations in one category will be
        deleted. Default (or None) is 10.

    dc_screen_covariates : Boolean (or None), optional
        Screen and clean covariates.
        Default (or None) is True.

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
        (ii) Variables used to compute GATEs, BGATEs, CBGATEs.
        Variables contained in 'var_x_name_remain_ord'
        or 'var_x_name_remain_unord', or are needed otherwise, are not removed.
        If the number of variables is very large (and the space of
        relevant features is much sparser, then using feature selection is
        likely to improve computational and statistical properties of the
        mcf etimator).
        Default (or None) is False.

    fs_rf_threshold : Integer or Float (or None), optional
        Feature selection: Threshold in terms of relative loss of variable
        importance in %.
        Default (or None) is 1.

    fs_other_sample : Boolean (or None), optional
        True : Random sample from training data used. These
        observations will not be used for causal forest.
        False : Use the same sample as used for causal forest estimation.
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
        exchanged; X-fitting). No inference is attempted for these
        parameters.
        Default (or None) is False.

    gen_mp_parallel : Integer (or None), optional
        Number of parallel processes (using ray on CPU). The smaller this
        value is, the slower the programme, the smaller its demands on RAM.
        None : 80% of logical cores.
        Default is None.

    gen_outfiletext : String (or None), optional
        File for text output. (.txt) file extension will be added.
        None : 'txtFileWithOutput'.
        Default is None.

    gen_outpath : String or Pathlib object (or None), optional
        Path were the output is written too (text, estimated effects, etc.)
        If specified directory does not exist, it will be created.
        None : An (.../out) directory below the current directory is used.
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

    lc_yes : Boolean (or None), optional
        Local centering. The predicted value of the outcome from a regression
        with all features (but without the treatment) is subtracted from the
        observed outcomes (using 5-fold cross-fitting). The best method for the
        regression is selected among scikit-learn's Random Forest, Support
        Vector Machines, and AdaBoost Regression based on their out-of-sample
        mean squared error. The method selection is either performed on the
        subsample used to build the forest ((1-lc_cs_share) for training,
        lc_cs_share for test).
        Default (or None) is True.

    lc_estimator : String (or None), optional
        The estimator used for local centering. Possible choices are
        scikit-learn's regression methods
        'RandomForest', 'RandomForestNminl5', 'RandomForestNminls5',
        'SupportVectorMachine', 'SupportVectorMachineC2',
        'SupportVectorMachineC4',
        'AdaBoost', 'AdaBoost100', 'AdaBoost200',
        'GradBoost', 'GradBoostDepth6', 'GradBoostDepth12', 'LASSO',
        'NeuralNet', 'NeuralNetLarge',  'NeuralNetLarger', 'Mean'.
        If set to 'automatic', the estimator with the lowest out-of-sample
        mean squared error (MSE) is selected. Whether this selection is based on
        cross-validation or a test sample is governed by the keyword lc_cs_cv.
        'Mean' is included for the cases when none of the methods have
        explanatory power.
        Default (or None) is 'RandomForest'.

    lc_uncenter_po : Boolean (or None), optional
        Predicted potential outcomes are re-adjusted for local centering
        are added to data output (iate and iate_eff in results dictionary).
        Default (or None) is True.

    lc_cs_cv : Boolean (or None), optional
        Data to be used for local centering & common support adjustment.
        True : Crossvalidation.
        False : Random sample not to be used for forest building.
        Default (or None) is True.

    lc_cs_cv_k : Integer (or None), optional
        Data to be used for local centering & common support adjustment:
        Number of folds in cross-validation (if lc_cs_cv is True).
        Default (or None) depends on the size of the training sample 
        (N): N < 100'000: 5;  100'000 <= N < 250'000: 4 250'000 <= N < 500'000: 3, 500'000 <= N: 2.

    lc_cs_share : Float (or None), optional
        Data to be used for local centering & common support adjustment:
        Share of trainig data (if lc_cs_cv is False).
        Default (or None) is 0.25.

    p_atet : Boolean (or None), optional
        Compute effects for specific treatment groups. Only possible if
        treatment is included in prediction data.
        Default (or None) is False.

    p_gates_minus_previous : Boolean (or None), optional
        Estimate increase of difference of GATEs, CBGATEs, BGATEs when
        evaluated at next larger observed value.
        Default (or None) is False.

    p_gates_no_evalu_points : Integer (or None), optional
        Number of evaluation points for discretized variables in (CB)(B)GATE
        estimation.
        Default (or None) is 50.

    p_gates_smooth : Boolean (or None), optional
        Alternative way to estimate GATEs for continuous features. Instead
        of discretizing variable, its GATE is evaluated at
        p_gates_smooth_no_evalu_points. Since there are likely to be no
        observations, a local neighbourhood around the evaluation points is
        considered.
        Default (or None) is True.

    p_gates_smooth_bandwidth : Float (or None), optional
        Multiplier for bandwidth used in (C)BGATE estimation with smooth
        variables.
        Default (or None) is 1.

    p_gates_smooth_no_evalu_points : Integer (or None), optional
        Number of evaluation points for discretized variables in GATE
        estimation.
        Default (or None) is 50.

    p_gatet : Boolean (or None), optional
        Compute effects for specific treatment groups. Only possible if
        treatment is included in prediction data.
        Default (or None) is False.

    p_bgate : Boolean (or None), optional
        Estimate a GATE that is balanced in selected features (as specified
        in var_x_name_balance_bgate).
        Default (or None) is False.

    p_cbgate : Boolean (or None), optional
        Estimate a GATE that is balanced in all other features.
        Default (or None) is False.

    p_bgate_sample_share : Float (or None), optional
        Implementation of (C)BGATE estimation is very cpu intensive.
        Therefore, random samples are used to speed up the programme if
        there are number observations  / number of evaluation points > 10.
        None :
        If observation in prediction data (n) < 1000: 1
        If n >= 1000:

        .. math::

            1000 + \\frac{{(n - 1000)^{\\frac{3}{4}}}}{{\\text{evaluation points}}}

        Default is None.

    p_max_cats_z_vars : Integer (or None), optional
        Maximum number of categories for discretizing continuous z
        variables.
        None : :math:`\\text{Number of observations}^{0.3}`
        Default is None.

    p_iate : Boolean (or None), optional
        IATEs will be estimated.
        Default (or None) is True.

    p_iate_se : Boolean (or None), optional
        Standard errors of IATEs will be estimated.
        Default (or None) is False.

    p_iate_m_ate : Boolean (or None), optional
        IATEs minus ATE will be estimated.
        Default (or None) is False.

    p_qiate : Boolean (or None), optional
        QIATEs will be estimated.
        Default (or None) is False.

    p_qiate_se : Boolean (or None), optional
        Standard errors of QIATEs will be estimated.
        Default (or None) is False.

    p_qiate_m_mqiate : Boolean (or None), optional
        QIATEs minus median of QIATEs will be estimated.
        Default (or None) is False.

    p_qiate_m_opp : Boolean (or None), optional.
       QIATE(x, q) - QIATE(x, 1-q) will be estimated (q denotes quantil level,
       q < 0.5),
       Default is False.

    p_qiate_no_of_quantiles : Integer (or None), optional
        Number of quantiles used for QIATE.
        Default (or None) is 99.

    p_qiate_smooth : Boolean (or None), optional
        Smooth estimated QIATEs using kernel smoothing.
        Default is True.

    p_qiate_smooth_bandwidth : Integer or Float (or None), optional
        Multiplier applied to default bandwidth used for kernel smoothing
        of QIATE.
        Default (or None) is 1.

    p_qiate_bias_adjust : Boolean (or None), optional
        Bias correction procedure for QIATEs based on simulations.
        Default is True.
    If p_qiate_bias_adjust is True, P_IATE_SE is set to True as well.

    p_qiate_bias_adjust_draws : Integer or Float (or None), optional
        Number of random draws used in computing the bias adjustment.
        Default is 1000.

    p_ci_level : Float (or None), optional
        Confidence level for bounds used in plots.
        Default (or None) is 0.95.

    p_cond_var : Boolean (or None), optional
        True : Conditional mean & variances are used.
        False : Variance estimation uses :math:`wy_i = w_i \\times y_i`
        directly.
        Default (or None) is True.

    p_knn : Boolean (or None), optional
        True : k-NN estimation. False: Nadaraya-Watson estimation.
        Nadaray-Watson estimation gives a better approximaton of the
        variance, but k-NN is much faster, in particular for larger datasets.
        Default (or None) is True.

    p_knn_min_k : Integer (or None), optional
        Minimum number of neighbours k-nn estimation.
        Default (or None) is 10.

    p_nw_bandw : Float (or None), optional
        Bandwidth for nw estimation: Multiplier of Silverman's optimal
        bandwidth.
        Default (or None) is 1.

    p_nw_kern : Integer (or None), optional
        Kernel for Nadaraya-Watson estimation.
        1 : Epanechikov.
        2 : Normal pdf.
        Default (or None) is 1.

    p_max_weight_share : Float (or None), optional
        Truncation of extreme weights. Maximum share of any weight, 0 <,
        <= 1. Enforced by trimming excess weights and renormalisation for
        each (BG,G,I,CBG)ATE separately. Because of renormalisation, the
        final weights could be somewhat above this threshold.
        Default (or None) is 0.05.

    p_cluster_std : Boolean (or None), optional
        Clustered standard errors. Always True if gen_panel_data is True.
        Default (or None) is False.

    p_se_boot_ate : Integer or Boolean (or None), optional
        Bootstrap of standard errors for ATE. Specify either a Boolean (if
        True, number of bootstrap replications will be set to 199) or an
        integer corresponding to the number of bootstrap replications (this
        implies True).
        None : 199 replications p_cluster_std is True, and False otherwise.
        Default is None.

    p_se_boot_gate : Integer or Boolean (or None), optional
        Bootstrap of standard errors for GATE. Specify either a Boolean (if
        True, number of bootstrap replications will be set to 199) or an
        integer corresponding to the number of bootstrap replications (this
        implies True).
        None : 199 replications p_cluster_std is True, and False otherwise.
        Default is None.

    p_se_boot_iate : Integer or Boolean (or None), optional
        Bootstrap of standard errors for IATE. Specify either a Boolean (if
        True, number of bootstrap replications will be set to 199) or an
        integer corresponding to the number of bootstrap replications (this
        implies True).
        None : 199 replications p_cluster_std is True, and False otherwise.
        Default is None.

    p_se_boot_qiate : Integer or Boolean (or None), optional
        Bootstrap of standard errors for QIATE. Specify either a Boolean (if
        True, number of bootstrap replications will be set to 199) or an
        integer corresponding to the number of bootstrap replications (this
        implies True).
        None : 199 replications p_cluster_std is True, and False otherwise.
        Default is None.

    p_bt_yes : Boolean (or None), optional
        ATE based balancing test based on weights. Relevance of this test
        in its current implementation is not fully clear.
        Default (or None) is True.

    p_choice_based_sampling : Boolean (or None), optional
        Choice based sampling to speed up programme if treatment groups
        have very different sizes.
        Default (or None) is False.

    p_choice_based_probs : List of Floats (or None), optional
        Choice based sampling:  Sampling probabilities to be specified.
        These weights are used for (G,B,CB)ATEs only. Treatment information
        must be available in the prediction data.
        Default is None.

    p_ate_no_se_only : Boolean (or None),optional
        Computes only the ATE without standard errors.
        Default (or None) is False.

    post_est_stats : Boolean (or None), optional
        Descriptive Analyses of IATEs (p_iate must be True).
        Default (or None) is True.

    post_relative_to_first_group_only : Boolean (or None), optional
        Descriptive Analyses of IATEs: Use only effects relative to
        treatment with lowest treatment value.
        Default (or None) is True.

    post_bin_corr_yes : Boolean (or None), optional
        Descriptive Analyses of IATEs: Checking the binary correlations of
        predictions with features.
        Default (or None) is True.

    post_bin_corr_threshold : Float, optional
        Descriptive Analyses of IATEs: Minimum threshhold of absolute
        correlation to be displayed.
        Default (or None) is 0.1.

    post_kmeans_yes : Boolean (or None), optional
        Descriptive Analyses of IATEs: Using k-means clustering to analyse
        patterns in the estimated effects.
        Default (or None) is True.

    post_kmeans_single : Boolean (or None), optional
        If True (and post_kmeans_yes is True), clustering is also with respect
        to all single effects. If False (and post_kmeans_yes is True),
        clustering is only with respect to all relevant IATEs jointly.
        Default (or None) is False.

    post_kmeans_no_of_groups : Integer or List or Tuple (or None), optional
        Descriptive Analyses of IATEs: Number of clusters to be built in
        k-means.
        None : List of 5 values: [a, b, c, d, e]; c = 5 to 10;
        depending on number of observations; c<7: a=c-2, b=c-1, d=c+1,
        e=c+2, else a=c-4, b=c-2, d=c+2, e=c+4.
        Default is None.

    post_kmeans_max_tries : Integer (or None), optional
        Descriptive Analyses of IATEs: Maximum number of iterations of
        k-means to achive convergence.
        Default (or None) is 1000.

    post_kmeans_replications : Integer (or None), optional
        Descriptive Analyses of IATEs: Number of replications with random
        start centers to avoid local extrema.
        Default (or None) is 10.

    post_kmeans_min_size_share : Float (or None).
        Smallest share observations for cluster size allowed in % (0-33).
        Default (None) is 1 (%).

    post_random_forest_vi : Boolean (or None), optional
        Descriptive Analyses of IATEs: Variable importance measure of
        random forest used to learn factors influencing IATEs.
        Default (or None) is True.

    post_plots : Boolean (or None), optional
        Descriptive Analyses of IATEs: Plots of estimated treatment
        effects.
        Default (or None) is True.

    post_tree : Boolean (or None), optional
        Regression trees (honest and standard) of Depth 2 to 5
        are estimated to describe IATES(x).
        Default (or None) is True.

    p_knn_const : Boolean (or None), optional
        Multiplier of default number of observation used in moving
        average of :meth:`~ModifiedCausalForest.analyse` method.
        Default (or None) is 1.

    _int_cuda : Boolean (or None), optional
        Use CUDA based GPU if CUDA-compatible GPU is available on hardware
        (experimental). Default (or None) is False.

    _int_descriptive_stats : Boolean (or None), optional
        Print descriptive stats if _int_with_output is True.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_show_plots : Boolean (or None), optional
        Execute show() command if _int_with_output is True.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_dpi : Integer (or None), optional
        dpi in plots.
        Default (or None) is 500.
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
        discretisation to speed up programme.
        Default is None.
        Internal variable, change default only if you know what you do.

    _int_max_save_values : Integer (or None), optional
        Save value of features in table only if less than
        _int_max_save_values different values.
        Default (or None) is 50.
        Internal variable, change default only if you know what you do.

    _int_max_obs_training : Integer (or None), optional
        Upper limit for sample size. If actual number is larger than this
        number, then the respective data will be randomly reduced to the
        specified upper limit.
        Training method: Reducing observations for training increases MSE
        and thus should be avoided. Default is infinity.
        Internal variable, change default only if you know what you do.

    _int_max_obs_prediction : Integer (or None), optional
        Upper limit for sample size. If actual number is larger than this
        number, then the respective data will be randomly reduced to the
        specified upper limit.
        Prediction method: Reducing observations for prediction does not
        much affect MSE. It may reduce detectable heterogeneity, but may also
        dramatically reduce computation time. Default is 250'000.
        Internal variable, change default only if you know what you do.

    _int_max_obs_kmeans :  Integer (or None), optional
        Upper limit for sample size. If actual number is larger than this
        number, then the respective data will be randomly reduced to the
        specified upper limit.
        kmeans in analyse method: Reducing observations may reduce detectable
        heterogeneity, but also reduces computation time. Default is 200'000.
        Internal variable, change default only if you know what you do.

    _int_max_obs_post_rel_graphs :  Integer (or None), optional
        Upper limit for sample size. If actual number is larger than this
        number, then the respective data will be randomly reduced to the
        specified upper limit. Figures show the relation of IATEs and features
        (note that the built-in non-parametric regression is computationally
        intensive).
        Default is 50'000.
        Internal variable, change default only if you know what you do.

    _int_mp_ray_del : Tuple of strings (or None), optional
        'refs' : Delete references to object store.
        'rest' : Delete all other objects of Ray task.
        'none' : Delete no objects.
        These 3 options can be combined.
        Default (or None) is ('refs',).
        Internal variable, change default only if you know what you do.

    _int_mp_ray_objstore_multiplier : Float (or None), optional
        Changes internal default values for size of Ray object store. Change to
        1 if programme crashes because object store is full. Only relevant
        if _int_mp_ray_shutdown is True.
        Default (or None) is 1.
        Internal variable, change default only if you know what you do.

    _int_mp_ray_shutdown : Boolean (or None), optional
        When computing the mcf repeatedly like in Monte Carlo studies,
        setting _int_mp_ray_shutdown to True may be a good idea.
        None: False if obs < 100000, True otherwise.
        Default is None.
        Internal variable, change default only if you know what you do.

    _int_mp_vim_type : Integer (or None), optional
        Type of multiprocessing when computing variable importance
        statistics:
        1 : Variable based (fast, lots of memory).
        2 : Bootstrap based (slower, less memory).
        None: 1 if obs < 20000, 2 otherwise.
        Default is None.
        Internal variable, change default only if you know what you do.

    _int_iate_chunk_size : Integer or None, optional
        Number of IATEs that are estimated in a single ray worker.
        Default is number of prediction observations / workers.
        If programme crashes in second part of IATE because of excess memory
        consumption, reduce _int_iate_chunk_size.

    _int_mp_weights_tree_batch : Integer (or None), optional
        Number of batches to split data in weight computation for variable
        importance statistics: The smaller the number of batches, the faster
        the programme and the more memory is needed.
        None : Automatically determined.
        Default is None.
        Internal variable, change default only if you know what you do.

    _int_mp_weights_type : Integer (or None), optional
        Type of multiprocessing when computing weights:
        '1': Groups-of-obs based (fast, lots of memory).
        '2' :Tree based (takes forever, less memory).
        Value of 2 will be internally changed to 1 if multiprocessing.
        Default (or None) is 1.
        Internal variable, change default only if you know what you do.

     _int_obs_bigdata : Integer or None, optional
         If number of training observations is larger than this number, the
         following happens during training:
         (i) Number of workers is halved in local centering.
         (ii) Ray is explicitely shut down.
         (iii) The number of workers used is reduced to 75% of default.
         (iv) The data type for some numpy arrays is reduced from float64 to float32.
         Default is 1'000'000.

    _int_output_no_new_dir : Boolean (or None), optional
        Do not create a new directory when the path already exists.
        Default (or None) is False.

    _int_report : Boolean (or None), optional
        Provide information for McfOptPolReports to construct informative
        reports.
        Default (or None) is True.

    _int_return_iate_sp : Boolean (or None), optional
        Return all data with predictions despite _int_with_output is False
        (useful for cross-validation and simulations).
        Default (or None) is False.
        Internal variable, change default only if you know what you do.

    _int_replication : Boolean (or None), optional
        If True all scikit-learn based computations will NOT use multi-
        processing.
        Default (or None) is False.

    _int_seed_sample_split : Integer (or None), optional
        Seeding is redone when building forest.
        Default (or None) is 67567885.
        Internal variable, change default only if you know what you do.

    _int_share_forest_sample : Float (or None), optional
        Share of sample used build forest.
        Default (or None) is 0.5.
        Internal variable, change default only if you know what you do.

    _int_verbose :  Boolean (or None), optional
        Additional output about running of mcf if _int_with_output is True.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_weight_as_sparse :  Boolean (or None), optional
        Save weights matrix as sparse matrix.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_weight_as_sparse_splits : Integer (or None), optional
        Compute sparse weight matrix in several chuncks.
        None : (Rows of prediction data * rows of Fill_y data)/(number of training splits * 25'000 * 25'000))
        Default is None.
        Internal variable, change default only if you know what you do.

    _int_with_output : Boolean (or None), optional
        Print output on txt file and/or console.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_del_forest : Boolean (or None), optional
        Delete forests from instance. If True, less memory is needed, but
        the trained instance of the class cannot be reused when calling
        :meth:`~ModifiedCausalForest.predict` with the same instance again,
        i.e. the forest has to be retrained when applied again.
        Default (or None) is False.

    _int_keep_w0 : Boolean (or None), optional.
        Keep all zeros weights when computing standard errors (slows down
        computation and may lead to undesirable behaviour).
        Default is False.

   Attributes
   ----------

    version : String
        Version of mcf module used to create the instance.

    <NOT-ON-API>

    blind_dict : Dictionary
        Parameters to compute (partially) blinded IATEs.

    cf_dict : Dictionary
        Parameters used in training the forest (directly).

    cs_dict : Dictionary
        Parameters used in common support adjustments.

    ct_dict : Dictionary
        Parameters used in dealing with continuous treatments.

    data_train_dict : Dictionary

    dc_dict : Dictionary
        Parameters used in data cleaning.

    fs_dict : Dictionary
        Parameters used in feature selection.

    forest : List
        List of lists containing the estimated causal forest.

    gen_dict : Dictionary
        General parameters used in various parts of the programme.

    int_dict : Dictionary
        Internal parameters used in various parts of the class.

    iv_mcf : Dictionary
        Internal instances of instrumental mcf containing for first stage
        and reduced form.

    lc_dict : Dictionary
        Parameters used in local centering.

    p_dict : Dictionary
        Parameters used in prediction method.

    post_dict : Dictionary
        Parameters used in analyse method.

    report :
        Provides information for McfOptPolReports to construct reports.

    sens_dict : Dictionary
        Parameters used in sensitivity method.

    time_strings : String
        Detailed information on how long the different methods needed.

    var_dict : Dictionary
        Variable names.

    var_x_type : Dictionary
        Types of covariates (internal).

    var_x_values : Dictionary
        Values of covariates (internal).

    </NOT-ON-API>

    """

    def __init__(
            self,
            var_d_name=None, var_id_name=None, var_iv_name=None,
            var_w_name=None,
            var_x_name_always_in_ord=None, var_x_name_always_in_unord=None,
            var_x_name_balance_test_ord=None,
            var_x_name_balance_test_unord=None, var_x_name_remain_ord=None,
            var_x_name_remain_unord=None, var_x_name_ord=None,
            var_x_name_unord=None, var_y_name=None, var_y_tree_name=None,
            var_z_name_cont=None, var_z_name_ord=None, var_z_name_unord=None,
            cf_alpha_reg_grid=1, cf_alpha_reg_max=0.15, cf_alpha_reg_min=0.05,
            cf_boot=1000, cf_chunks_maxsize=None, cf_compare_only_to_zero=False,
            cf_n_min_grid=1, cf_n_min_max=None, cf_n_min_min=None,
            cf_n_min_treat=None, cf_nn_main_diag_only=False, cf_m_grid=1,
            cf_m_random_poisson=True, cf_m_share_max=0.6, cf_m_share_min=0.1,
            cf_match_nn_prog_score=True, cf_mce_vart=1,
            cf_random_thresholds=None, cf_p_diff_penalty=None,
            cf_penalty_type='mse_d',
            cf_subsample_factor_eval=None, cf_subsample_factor_forest=1,
            cf_tune_all=False, cf_vi_oob_yes=False,
            cs_adjust_limits=None, cs_detect_const_vars_stop=True,
            cs_max_del_train=0.5, cs_min_p=0.01, cs_quantil=1, cs_type=1,
            ct_grid_dr=100, ct_grid_nn=10, ct_grid_w=10,
            dc_check_perfectcorr=True, dc_clean_data=True, dc_min_dummy_obs=10,
            dc_screen_covariates=True, fs_rf_threshold=1, fs_other_sample=True,
            fs_other_sample_share=0.33, fs_yes=False,
            gen_d_type='discrete', gen_iate_eff=False, gen_panel_data=False,
            gen_mp_parallel=None, gen_outfiletext=None, gen_outpath=None,
            gen_output_type=2, gen_panel_in_rf=True, gen_weighted=False,
            lc_cs_cv=True, lc_cs_cv_k=None, lc_cs_share=0.25,
            lc_estimator='RandomForest', lc_yes=True, lc_uncenter_po=True,
            p_ate_no_se_only=False, p_atet=False, p_bgate=False,
            p_bgate_sample_share=None, p_bt_yes=True, p_cbgate=False,
            p_choice_based_sampling=False, p_choice_based_probs=None,
            p_ci_level=0.95, p_cluster_std=False, p_cond_var=True,
            p_gates_minus_previous=False, p_gates_smooth=True,
            p_gates_smooth_bandwidth=1, p_gates_smooth_no_evalu_points=50,
            p_gates_no_evalu_points=50, p_gatet=False,
            p_iate=True, p_iate_se=False, p_iate_m_ate=False,
            p_iv_aggregation_method=('local', 'global',),
            p_knn=True, p_knn_const=1, p_knn_min_k=10,
            p_nw_bandw=1, p_nw_kern=1,
            p_max_cats_z_vars=None, p_max_weight_share=0.05,
            p_qiate=False, p_qiate_se=False, p_qiate_m_mqiate=False,
            p_qiate_m_opp=False,
            p_qiate_no_of_quantiles=99, p_qiate_smooth=True,
            p_qiate_smooth_bandwidth=1,
            p_qiate_bias_adjust=True,
            p_se_boot_ate=None, p_se_boot_gate=None,
            p_se_boot_iate=None, p_se_boot_qiate=None,
            var_x_name_balance_bgate=None, var_cluster_name=None,
            post_bin_corr_threshold=0.1, post_bin_corr_yes=True,
            post_est_stats=True, post_kmeans_no_of_groups=None,
            post_kmeans_max_tries=1000, post_kmeans_min_size_share=None,
            post_kmeans_replications=10, post_kmeans_single=False,
            post_kmeans_yes=True, post_random_forest_vi=True,
            post_relative_to_first_group_only=True, post_plots=True,
            post_tree=True,
            _int_cuda=False, _int_del_forest=False,
            _int_descriptive_stats=True, _int_dpi=500, _int_fontsize=2,
            _int_iate_chunk_size=None,
            _int_keep_w0=False, _int_no_filled_plot=20,
            _int_max_cats_cont_vars=None, _int_max_save_values=50,
            _int_max_obs_training=float('inf'), _int_max_obs_prediction=250000,
            _int_max_obs_kmeans=200000, _int_max_obs_post_rel_graphs=50000,
            _int_mp_ray_del=('refs',), _int_mp_ray_objstore_multiplier=1,
            _int_mp_ray_shutdown=None, _int_mp_vim_type=None,
            _int_mp_weights_tree_batch=None, _int_mp_weights_type=1,
            _int_obs_bigdata=1000000,
            _int_output_no_new_dir=False, _int_red_largest_group_train=False,
            _int_replication=False, _int_report=True, _int_return_iate_sp=False,
            _int_seed_sample_split=67567885, _int_share_forest_sample=0.5,
            _int_show_plots=True, _int_verbose=True,
            _int_weight_as_sparse=True, _int_weight_as_sparse_splits=None,
            _int_with_output=True
            ):

        self.__version__ = '0.8.0'

        self.int_dict = mcf_init.int_init(
            cuda=_int_cuda, cython=False,  # Cython turned off for now
            del_forest=_int_del_forest,
            descriptive_stats=_int_descriptive_stats, dpi=_int_dpi,
            fontsize=_int_fontsize,
            max_save_values=_int_max_save_values, mp_ray_del=_int_mp_ray_del,
            mp_ray_objstore_multiplier=_int_mp_ray_objstore_multiplier,
            mp_weights_tree_batch=_int_mp_weights_tree_batch,
            mp_weights_type=_int_mp_weights_type,
            no_filled_plot=_int_no_filled_plot,
            return_iate_sp=_int_return_iate_sp, replication=_int_replication,
            seed_sample_split=_int_seed_sample_split,
            share_forest_sample=_int_share_forest_sample,
            show_plots=_int_show_plots, verbose=_int_verbose,
            weight_as_sparse=_int_weight_as_sparse, keep_w0=_int_keep_w0,
            with_output=_int_with_output, mp_ray_shutdown=_int_mp_ray_shutdown,
            mp_vim_type=_int_mp_vim_type,
            obs_bigdata=_int_obs_bigdata,
            output_no_new_dir=_int_output_no_new_dir, report=_int_report,
            weight_as_sparse_splits=_int_weight_as_sparse_splits,
            max_cats_cont_vars=_int_max_cats_cont_vars,
            iate_chunk_size=_int_iate_chunk_size,
            max_obs_training=_int_max_obs_training,
            max_obs_prediction=_int_max_obs_prediction,
            max_obs_kmeans=_int_max_obs_kmeans,
            max_obs_post_rel_graphs=_int_max_obs_post_rel_graphs,
            p_ate_no_se_only=p_ate_no_se_only
            )
        gen_dict = mcf_init.gen_init(
            self.int_dict,
            d_type=gen_d_type, iate_eff=gen_iate_eff,
            mp_parallel=gen_mp_parallel, outfiletext=gen_outfiletext,
            outpath=gen_outpath, output_type=gen_output_type,
            weighted=gen_weighted, panel_data=gen_panel_data,
            panel_in_rf=gen_panel_in_rf)
        self.dc_dict = mcf_init.dc_init(
            check_perfectcorr=dc_check_perfectcorr,
            clean_data=dc_clean_data, min_dummy_obs=dc_min_dummy_obs,
            screen_covariates=dc_screen_covariates)
        self.ct_dict = {'grid_dr': ct_grid_dr, 'grid_nn': ct_grid_nn,
                        'grid_w': ct_grid_w}
        self.fs_dict = mcf_init.fs_init(
            rf_threshold=fs_rf_threshold, other_sample=fs_other_sample,
            other_sample_share=fs_other_sample_share, yes=fs_yes)
        self.cs_dict = mcf_init.cs_init(
            gen_dict,
            max_del_train=cs_max_del_train, min_p=cs_min_p, quantil=cs_quantil,
            type_=cs_type, adjust_limits=cs_adjust_limits,
            detect_const_vars_stop=cs_detect_const_vars_stop)
        self.lc_dict = mcf_init.lc_init(
            cs_cv=lc_cs_cv, cs_cv_k=lc_cs_cv_k, cs_share=lc_cs_share,
            undo_iate=lc_uncenter_po, yes=lc_yes, estimator=lc_estimator)
        self.cf_dict = mcf_init.cf_init(
            alpha_reg_grid=cf_alpha_reg_grid, alpha_reg_max=cf_alpha_reg_max,
            alpha_reg_min=cf_alpha_reg_min, boot=cf_boot,
            chunks_maxsize=cf_chunks_maxsize,
            compare_only_to_zero=cf_compare_only_to_zero,
            nn_main_diag_only=cf_nn_main_diag_only, m_grid=cf_m_grid,
            m_share_max=cf_m_share_max, m_share_min=cf_m_share_min,
            m_random_poisson=cf_m_random_poisson,
            match_nn_prog_score=cf_match_nn_prog_score, mce_vart=cf_mce_vart,
            vi_oob_yes=cf_vi_oob_yes, n_min_grid=cf_n_min_grid,
            n_min_max=cf_n_min_max, n_min_min=cf_n_min_min,
            n_min_treat=cf_n_min_treat,
            p_diff_penalty=cf_p_diff_penalty, penalty_type=cf_penalty_type,
            subsample_factor_eval=cf_subsample_factor_eval,
            subsample_factor_forest=cf_subsample_factor_forest,
            tune_all=cf_tune_all,
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
            gatet=p_gatet, gate_no_evalu_points=p_gates_no_evalu_points,
            bgate_sample_share=p_bgate_sample_share,
            iate=p_iate, iate_se=p_iate_se, iate_m_ate=p_iate_m_ate, knn=p_knn,
            knn_const=p_knn_const, knn_min_k=p_knn_min_k, nw_bandw=p_nw_bandw,
            nw_kern=p_nw_kern, max_cats_z_vars=p_max_cats_z_vars,
            max_weight_share=p_max_weight_share,
            se_boot_ate=p_se_boot_ate, se_boot_gate=p_se_boot_gate,
            se_boot_iate=p_se_boot_iate,
            qiate=p_qiate, qiate_se=p_qiate_se,
            qiate_m_mqiate=p_qiate_m_mqiate, qiate_m_opp=p_qiate_m_opp,
            qiate_no_of_quantiles=p_qiate_no_of_quantiles,
            se_boot_qiate=p_se_boot_qiate, qiate_smooth=p_qiate_smooth,
            qiate_smooth_bandwidth=p_qiate_smooth_bandwidth,
            qiate_bias_adjust=p_qiate_bias_adjust,
            iv_aggregation_method=p_iv_aggregation_method)
        self.post_dict = mcf_init.post_init(
            p_dict,
            bin_corr_threshold=post_bin_corr_threshold,
            bin_corr_yes=post_bin_corr_yes, est_stats=post_est_stats,
            kmeans_no_of_groups=post_kmeans_no_of_groups,
            kmeans_max_tries=post_kmeans_max_tries,
            kmeans_replications=post_kmeans_replications,
            kmeans_yes=post_kmeans_yes, kmeans_single=post_kmeans_single,
            kmeans_min_size_share=post_kmeans_min_size_share,
            random_forest_vi=post_random_forest_vi,
            relative_to_first_group_only=post_relative_to_first_group_only,
            plots=post_plots, tree=post_tree)
        self.var_dict, self.gen_dict, self.p_dict = mcf_init.var_init(
            gen_dict, self.fs_dict, p_dict,
            x_name_balance_bgate=var_x_name_balance_bgate,
            cluster_name=var_cluster_name,
            d_name=var_d_name, id_name=var_id_name, w_name=var_w_name,
            iv_name=var_iv_name,
            x_name_balance_test_ord=var_x_name_balance_test_ord,
            x_name_balance_test_unord=var_x_name_balance_test_unord,
            x_name_always_in_ord=var_x_name_always_in_ord,
            x_name_always_in_unord=var_x_name_always_in_unord,
            x_name_remain_ord=var_x_name_remain_ord,
            x_name_remain_unord=var_x_name_remain_unord,
            x_name_ord=var_x_name_ord, x_name_unord=var_x_name_unord,
            y_name=var_y_name, y_tree_name=var_y_tree_name,
            z_name_cont=var_z_name_cont, z_name_ord=var_z_name_ord,
            z_name_unord=var_z_name_unord)
        self.blind_dict = self.sens_dict = None
        self.data_train_dict = self.var_x_type = self.var_x_values = None
        self.forest, self.time_strings = None, {}
        self.report = {'predict_list': [],   # Needed for multiple predicts
                       'analyse_list': []
                       }
        self.iv_mcf = {'firststage': None, 'reducedform': None}
        self.predict_done = False
        self.predict_iv_done = False
        self.predict_different_allocations_done = False

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
        results : Dictionary.
            Contains the results. This dictionary has the following structure:
            'tree_df' : DataFrame
                Dataset used to build the forest.
            'fill_y_df' : DataFrame
                Dataset used to populate the forest with outcomes.
            'common_support_probabilities_tree': pd.DataFrame containing
                treatment probabilities for all treatments,
                the identifier of the observation, and a dummy variable
                indicating whether the observation is inside or outside the
                common support. This is for the data used to build the trees.
                None if _int_with_output is False.
            'common_support_probabilities_fill_y': pd.DataFrame containing
                treatment probabilities for all treatments, the identifier of
                the observation, and a dummy variable indicating
                whether the observation is inside or outside the common support.
                This is for the data used to fill the trees with outcome values.
                None if _int_with_output is False.
            'path_output' : Pathlib object
                Location of directory in which output is saved.

        """
        results = train_main(self, data_df)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results

    def train_iv(self, data_df):
        """
        Train the IV modified causal forest on the training data.

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the causal forest. It must contain information
            about outcomes, treatment, and features.

        Returns
        -------
        results : Dictionary.
            Contains the results. This dictionary has the following structure:
            'tree_df' : DataFrame
                Dataset used to build the forest.
            'fill_y_df' : DataFrame
                Dataset used to populate the forest with outcomes.
            'common_support_probabilities_tree': pd.DataFrame containing
                treatment probabilities for all treatments, the identifier of
                the observation, and a dummy variable indicating
                whether the observation is inside or outside the common support.
                This is for the data used to build the trees.
                None if _int_with_output is False.
            'common_support_probabilities_fill_y': pd.DataFrame containing
                treatment probabilities for all treatments, the identifier of
                the observation, and a dummy variable indicating
                whether the observation is inside or outside the common support.
                This is for the data used to fill the trees with outcome values.
                None if _int_with_output is False.
            'path_output' : Pathlib object
                Location of directory in which output is saved.

        """
        results = train_iv_main(self, data_df)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results

    def predict(self, data_df):
        """
        Compute all effects.

        meth:`~ModifiedCausalForest.train` method must be run beforehand.

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the predictions. It must contain information
            about features (and treatment if effects for treatment specific
            subpopulations are desired as well).

        Returns
        -------
        results : Dictionary.
            Contains the results. This dictionary has the following structure:
            'ate': ATE, 'ate_se': Standard error of ATE,
            'ate_effect_list': List of names of estimated effects,
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
            'qiate': QIATE, 'qiate_se': Standard error of QIATEs,
            'qiate_diff': QIATE minus QIATE at median,
            'qiate_diff_se': Standard error of QIATE minus QIATE at median,
            'iate_eff': (More) Efficient IATE (IATE estimated twice and
            averaged where role of tree_building and tree_filling
            sample is exchanged),
            'iate_data_df': DataFrame with IATEs,
            'iate_names_dic': Dictionary containing names of IATEs,
            'bala': Effects of balancing tests,
            'bala_se': Standard error of effects of balancing tests,
            'bala_effect_list': Names of effects of balancing tests.
            'common_support_probabilities' : pd.DataFrame containing treatment
            probabilities for all treatments, the identifier of the observation,
            and a dummy variable indicating whether the observation is inside or
            outside the common support. None if _int_with_output is False.
            'path_output': Pathlib object, location of directory in which output
            is saved.

        """
        self.predict_done = True
        results = predict_main(self, data_df)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results

    def predict_different_allocations(self,
                                      data_df,
                                      allocations_df: bool = None
                                      ):
        """
        Predict average potential outcomes for different allocations.

        meth:`~ModifiedCausalForest.train` method must be run beforehand. The
        details of this methods are described in the working paper by
        Busshoff and Lechner (2025).

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the predictions. It must contain information
            about features (and treatment if effects for treatment specific
            subpopulations are desired as well).

        allocations_df : Dataframe or None, optional
            Different allocations which are to be evaluated. The length of this
            dataframe must be the same as the length of data_df.
            Default is None.

        Returns
        -------
        results : Dictionary
            Results. This dictionary has the following structure:
            'ate': Average treatment effects
            'ate_se': Standard error of average treatment effects
            'ate_effect_list': List with name with estiamted effects
            'alloc_df': Dataframe with value and variance of value for all
                        allocations investigated.
            'outpath' : Pathlib object. Location of directory in which output
                        is saved.

        """
        self.predict_different_allocations_done = True
        results, self.gen_dict['outpath'] = predict_different_allocations_main(
            self, data_df, allocations_df)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results

    def predict_iv(self, data_df):
        """
        Compute all effects for instrument mcf (possibly in 2 differnt ways).

        :meth:`~ModifiedCausalForest.train_iv` method must be run beforehand.

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the predictions. It must contain information
            about features (and treatment if effects for treatment specific
            subpopulations are desired as well).

        Returns
        -------
        results_global : Dictionary.
            Contains the results. This dictionary has the following structure:
            'ate': LATE, 'ate_se': Standard error of LATE,
            'ate_effect_list': List of names of estimated effects,
            'ate_1st': ATE 1st stage, 'ate_1st_se': Standard error of ATE (1st)
            'ate 1st_effect_list': List of names of estimated effects (1st),
            'ate_redf': ATE reduced form, 'ate_redf_se': Standard error of ATE
            of reduced form,
            'ate redf_effect_list': List of names of estimated effects (red.f.),
            'gate': LGATE, 'gate_se': SE of LGATE,
            'gate_diff': LGATE minus LATE,
            'gate_diff_se': Standard error of LGATE minus LATE,
            'cbgate': LCBGATE (all covariates balanced),
            'cbgate_se': Standard error of LCBGATE,
            'cbgate_diff': LCBGATE minus LATE,
            'cbgate_diff_se': Standard error of LCBGATE minus LATE,
            'bgate': LBGATE (only prespecified covariates balanced),
            'bgate_se': Standard error of LBGATE,
            'bgate_diff': LBGATE minus LATE,
            'bgate_diff_se': Standard errror of LBGATE minus LATE,
            'gate_names_values': Dictionary: Order of gates parameters
            and name and values of LGATE effects.
            'iate': LIATE, 'iate_se': Standard error of LIATE,
            'iate_1st': IATE (1st stage), 'iate_1st_se': Standard error of
            IATE (1st stage),
            'iate_redf': IATE (reduced form), 'iate_redf_se': Standard error of
            IATE (reduced form),
            'iate_eff': (More) Efficient LIATE (LIATE estimated twice and
            averaged where role of tree_building and tree_filling
            sample is exchanged),
            iate_1st_eff': (More) Efficient IATE (1st stage),
            iate_redf_eff': (More) Efficient IATE (reduced form),
            'iate_data_df': DataFrame with LIATEs,
            'iate_1st_data_df': DataFrame with IATEs (1st stage),
            'iate_redf_data_df': DataFrame with IATEs (reduced form),
            'iate_names_dic': Dictionary containing names of LIATEs,
            'iate_1st_names_dic': Dictionary containing names of IATEs (1st),
            'iate_redf_dic': Dictionary containing names of LIATEs (red.f.),
            'qiate': QLIATE, 'qiate_se': Standard error of QLIATE,
            'bala_1st': Effects of balancing tests (1st stage),
            'bala_1st_se': Standard error of effects of balancing tests (1st),
            'bala_1st_effect_list': Names of effects of balancing tests (1st),
            'bala_redf': Effects of balancing tests (reduced form),
            'bala_redf_se': Standard error of effects of balancing tests (red.),
            'bala_redf_effect_list': Names of effects of balancing tests (red.).
            'common_support_probabilities': pd.DataFrame containing treatment
            probabilities for all treatments, the identifier of the observation,
            and a dummy variable indicating whether the observation is inside or
            outside the common support. None if _int_with_output is False.
            'path_output': Pathlib object, location of directory in which output
            is saved.

            It is empty if the IV estimation method 'global' has not been
            used.

        results_local : Dictionary.
            Same content as results_wald.
            It is empty if the IV estimation method 'local' has not been
            used.

        """
        self.predict_iv_done = True
        # Reduce sample size to upper limit
        data_df, rnd_reduce, txt_red = check_reduce_dataframe(
            data_df, title='Prediction',
            max_obs=self.int_dict['max_obs_prediction'],
            seed=124535, ignore_index=True)
        if rnd_reduce and self.int_dict['with_output']:
            print_mcf(self.gen_dict, txt_red, summary=True)

        results_global, results_local = predict_iv_main(self, data_df)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results_global, results_local

    def analyse(self, results):
        """
        Analyse estimated IATEs with various descriptive tools.

        Parameters
        ----------
        results : Dictionary
            Contains estimation results. This dictionary must have the same
            structure as the one returned from the
            :meth:`~ModifiedCausalForest.predict` method.

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
        results_plus_cluster = analyse_main(self, results)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results_plus_cluster

    def blinder_iates(
        self, data_df, blind_var_x_protected_name=None,
        blind_var_x_policy_name=None, blind_var_x_unrestricted_name=None,
        blind_weights_of_blind=None, blind_obs_ref_data=50,
            blind_seed=123456):
        """
        Compute IATEs that causally depend less on protected variables.

        WARNING
        This method is deprecated and will be removed in future
        versions. Use the method fairscores of the OptimalPolicy class instead.

        Parameters
        ----------
        data_df : DataFrame.
            Contains data needed to
            :meth:`~ModifiedCausalForest.predict` the various adjusted IATES.

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

        var_x_policy_ord_name : List of strings
            Ordered variables to be used to build the decision rules.

        var_x_policy_unord_name : List of strings.
            Unordered variables to be used to build the decision rules.

        var_x_blind_ord_name : List of strings
            Ordered variables to be used to blind potential outcomes.

        var_x_blind_unord_name : List of strings.
            Unordered variables to be used to blind potential outcomes.

        outpath : Pathlib object
            Location of directory in which output is saved.

        """
        print('This method of reducing dependence on protected '
              'variables is deprecated. Use the method '
              'fairscores of the OptimalPolicy class instead.')

        (blinded_dic, data_on_support_df, var_x_policy_ord_name,
         var_x_policy_unord_name, var_x_blind_ord_name,
         var_x_blind_unord_name, self.gen_dict['outpath']
         ) = blinder_iates_main(
             self,
             data_df, blind_var_x_protected_name=blind_var_x_protected_name,
             blind_var_x_policy_name=blind_var_x_policy_name,
             blind_var_x_unrestricted_name=blind_var_x_unrestricted_name,
             blind_weights_of_blind=blind_weights_of_blind,
             blind_obs_ref_data=blind_obs_ref_data, blind_seed=blind_seed)

        return (blinded_dic, data_on_support_df, var_x_policy_ord_name,
                var_x_policy_unord_name, var_x_blind_ord_name,
                var_x_blind_unord_name, self.gen_dict['outpath'])

    def sensitivity(
                    self,
                    train_df,
                    predict_df: None = None,
                    results: None = None,
                    sens_cbgate: None = None,
                    sens_bgate: None = None,
                    sens_gate: None = None,
                    sens_iate: None = None,
                    sens_iate_se: None = None,
                    sens_scenarios: None = None,
                    sens_cv_k: None = None,
                    sens_replications: int = 2,
                    sens_reference_population: None = None,
                    ):
        """
        Compute simulation-based sensitivity indicators.
    
        Parameters
        ----------
        train_df : DataFrame
            Data with real outcomes, treatments, and covariates. Data will be
            transformed to compute sensitivity indicators.
    
        predict_df : DataFrame, optional
            Prediction data to compute all effects for. This data will not be
            changed in the computation process. Only covariate information is
            used from this dataset. If None, ``train_df`` will be used.
    
        results : dict, optional
            Output dictionary from :meth:`~ModifiedCausalForest.predict`.
            If it contains estimated IATEs, they are used for the no-effect
            (basic) scenario and compared to those in the dictionary.
            Otherwise, passing it has no effect.
    
        sens_cbgate : bool, optional
            If True, compute CBGATEs for sensitivity analysis. Default is False.
    
        sens_bgate : bool, optional
            If True, compute BGATEs for sensitivity analysis. Default is False.
    
        sens_gate : bool, optional
            If True, compute GATEs for sensitivity analysis. Default is False.
    
        sens_iate : bool, optional
            If True, compute IATEs for sensitivity analysis.
            If ``results`` contains IATEs, default is True, otherwise False.
    
        sens_iate_se : bool, optional
            If True, compute standard errors of IATEs for sensitivity analysis.
            Default is False.
    
        sens_scenarios : list or tuple of str, optional
            Different scenarios considered. Default is ('basic',).
            - 'basic': Use estimated treatment probabilities for simulations
              (no confounding).
    
        sens_cv_k : int, optional
            Number of folds in cross-validation. Default is 5.
    
        sens_replications : int, optional
            Number of replications for simulating placebo treatments.
            Default is 2.
    
        sens_reference_population : int or float, optional
            Treatment status of the reference population used by the
            sensitivity analysis. Default is the treatment with the most
            observed observations.
    
        Returns
        -------
        results_avg : dict
            Same structure as :meth:`~ModifiedCausalForest.predict`, but
            averaged over replications (if applicable).
        """
        results_avg = sensitivity_main(
            self, train_df, predict_df=predict_df, results=results,
            sens_cbgate=sens_cbgate, sens_bgate=sens_bgate, sens_gate=sens_gate,
            sens_iate=sens_iate, sens_iate_se=sens_iate_se,
            sens_scenarios=sens_scenarios, sens_cv_k=sens_cv_k,
            sens_replications=sens_replications,
            sens_reference_population=sens_reference_population)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results_avg
