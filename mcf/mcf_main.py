from pathlib import Path

from mcf.mcf_feature_selection import FsCfg
from mcf.mcf_general import check_reduce_dataframe
from mcf.mcf_inf_for_alloc import predict_different_allocations_main
from mcf.mcf_init_change_keywords import change_keywords
from mcf.mcf_init import CtGrid, DCCfg, GenCfg, GenTvCfg, IntCfg, VarCfg
from mcf.mcf_init_predict import PBiasAdjustmentCfg, PCfg, PostCfg, LowMemCfg
from mcf.mcf_init_train import CfCfg, CsCfg, LcCfg
from mcf.mcf_init_update_helper import var_helper
from mcf.mcf_init_values_cfg import inconsistencies, inconsistencies_train, inconsistencies_sens
from mcf.mcf_iv import train_iv_main, predict_iv_main
from mcf.mcf_print_stats import print_mcf
from mcf.mcf_sensitivity import sensitivity_main
from mcf.mcf_unconfound import train_main, predict_main, analyse_main
from mcf.mcfoptp_parallel_backend_ray_classical import check_ray_shutdown


class ModifiedCausalForest:
    """
    Estimation of treatment effects with the Modified Causal Forest.

    Parameters
    ----------
    var_y_name : String, list or tuple of strings (or None), optional
        Names of outcome variables required in data passed to :meth:`~ModifiedCausalForest.train`
        or :meth:`~ModifiedCausalForest.train_iv`. At initialization, at least one outcome name
        must be supplied through 'var_y_name' or 'var_y_tree_name'.
        If 'var_y_tree_name' is None, the first outcome is used for tree building. Explicit tree
        outcomes are appended to the outcome list if they are not already included.
        Default is None; in that case, the outcomes come from 'var_y_tree_name'.

    var_d_name : String, list or tuple of strings (or None), optional
        Name of the treatment variable. Required at initialization; None raises ValueError.
        The named column is required in training data. It can also be supplied in prediction data.
        For discrete treatments, training recodes the main treatment to consecutive integers.
        Codes start at zero. When treatment information is used by predict(), supply the original
        training labels. The stored mapping is reused; unseen labels raise ValueError.
        If the number of versions > 1, this is a list with the treatment version as second variable.
        Note that the value of versions are conditional on the main treatment. In other words,
        version 2 of treatment 1 and version 2 of treatment 2 lead to different potential outcomes.
            IMPORTANT: Main treatment must always be the first element in the treatments list. This
            variable is also used for the programme to determine if there are treatment
            versions at all. Only one element in list/tuple or string: No treatment versions.
            Two elements in list/tuple: 1st element is main treatment, 2nd element is subtreatment.

    var_x_name_ord : String, list or tuple of strings (or None), optional
        Names of ordered forest covariates, including dummy variables. At least one ordered or
        unordered covariate must be supplied through these feature lists or 'var_z_name_cont',
        'var_z_name_ord' or 'var_z_name_unord'. Heterogeneity variables are added automatically.
        If 'p_ate_no_se_only' is True, heterogeneity variables are ignored, so at least one feature
        must be specified through 'var_x_name_ord' or 'var_x_name_unord'.
        Default is None.

    var_x_name_unord : String, list or tuple of strings (or None), optional
        Names of unordered forest covariates. The joint requirement for at least one feature is
        described under 'var_x_name_ord'. Variables from 'var_z_name_unord' are added automatically
        unless 'p_ate_no_se_only' is True.
        Training requires at least three distinct values per unordered covariate. Variables with
        fewer values raise ValueError; specify binary covariates as ordered variables.
        Prediction rows with categories absent from training are removed.
        Default is None.

    var_x_name_balance_bgate : String, list or tuple of strings (or None), optional
        Variables to balance the GATEs on. Only relevant if p_bgate is True. The distribution of
        these variables is kept constant when a BGATE is computed. None: Use the other heterogeneity
        variables (var_z_...) (if there are any) for balancing. Default is None.

    var_x_name_ba : String, list or tuple of strings (or None), optional
        Names of forest covariates included directly in the bias-adjustment regressions.
        Used only when both 'p_ba' and 'p_ba_use_x' are True; otherwise, the names are ignored.
        When both are True, a nonempty selection is required, even if propensity or prognostic
        scores are also enabled. An empty selection or names outside the ordered and unordered
        forest covariates raise ValueError during initialization.
        The variable set is fixed during training. In 'new_keywords', only the same set is
        accepted when both 'p_ba' and 'p_ba_use_x' are True after applying the updates.
        Otherwise, the supplied names are ignored and the trained selection is retained.
        Changing the set used for bias adjustment requires retraining.
        Default is None.

    var_x_name_balance_test_ord : String, list or tuple of strings (or None), optional
        Names of ordered variables used in balancing tests when ``'p_bt_yes'`` is True.
        Both balancing-variable lists also enter the initial training descriptives by treatment
        when output and descriptive statistics are enabled, even if ``'p_bt_yes'`` is False.
        If neither use is enabled, the lists are ignored. In prediction data, these columns are
        required only for an enabled balancing test, unless another active role also needs them.
        Default is None.

    var_x_name_balance_test_unord : String, list or tuple of strings (or None), optional
        Names of unordered variables used in balancing tests when ``'p_bt_yes'`` is True.
        These variables also enter the initial training descriptives described under
        ``'var_x_name_balance_test_ord'``. Later descriptive stages can include other variables.
        Default is None.

    var_x_name_always_in_ord : String, list or tuple of strings (or None), optional
        Names of ordered variables in the always-in set for the initial candidate draw.
        Together with var_x_name_always_in_unord, these variables have priority for splitting.
        They are added to the forest covariates and protected from feature selection and covariate
        screening. They need not also appear in 'var_x_name_ord' or 'var_x_name_unord'. The separate
        initial feature requirement under 'var_x_name_ord' must still be met.
        If the candidate count is at most the size of this set, draw that many variables from
        it uniformly without replacement. Otherwise include the whole set and draw the
        remaining candidates from the other variables. This also applies to singleton draws
        and to fixed candidate counts.
        Only relevant for :meth:`~ModifiedCausalForest.train`. Default is None.

    var_x_name_always_in_unord : String, list or tuple of strings (or None), optional
        Names of unordered variables in the always-in set for the initial candidate draw.
        Combine them with var_x_name_always_in_ord. The automatic inclusion, removal protection,
        initial feature requirement and priority rule described there apply to both lists.
        Only relevant for :meth:`~ModifiedCausalForest.train`. Default is None.

    var_x_name_remain_ord : String, list or tuple of strings (or None), optional
        Names of ordered forest covariates protected from removal by feature selection and by
        covariate screening ('dc_screen_covariates'). Listing a name here does not add it to the
        forest covariates. Ignored if both 'fs_yes' and 'dc_screen_covariates' are False.
        Default is None.

    var_x_name_remain_unord : String, list or tuple of strings (or None), optional
        Names of unordered forest covariates protected from removal by feature selection and by
        covariate screening ('dc_screen_covariates'). Listing a name here does not add it to the
        forest covariates. Ignored if both 'fs_yes' and 'dc_screen_covariates' are False.
        Default is None.

    var_x_name_tv : String, list or tuple of strings (or None), optional
        Names of forest covariates included directly in treatment-version regressions.
        Used when 'var_d_name' specifies a main treatment and a version column; otherwise ignored.
        When used, every supplied variable must belong to the covariates used to build the forest.
        None supplies no direct covariates. If 'gen_tv_addscores' is also False, no covariates are
        added to the version regressions. Interactions follow 'gen_tv_specification'.
        Default is None.

    var_cluster_name : String, list or tuple of strings (or None), optional
        Name of the cluster or panel identifier, given as a string or a one-element list or tuple.
        Required when ``'p_cluster_std'`` or ``'gen_panel_data'`` is True.
        Used for cluster-robust inference and, with ``'gen_panel_in_rf'``, for sampling clusters
        during tree construction.
        Default is None.

    var_id_name : String, list or tuple of strings (or None), optional
        Name of the identifier column, supplied as a string or a one-element list or tuple.
        The name is converted to lowercase. None selects 'id'.
        With 'dc_clean_data' enabled, an existing column with this name is kept. If it is absent,
        a column with consecutive integers starting at zero is added before cleaning removes rows.
        With 'dc_clean_data' disabled, supply the identifier column in the input data.
        Default is None.

    var_iv_name : String, list or tuple of strings (or None), optional
        Name of one binary instrument, supplied as a string or a one-element list or tuple.
        Required by :meth:`~ModifiedCausalForest.train_iv`; None raises ValueError during training.
        Instrument values must be 0 or 1. Fractional or other nonbinary values raise ValueError
        before conversion to integers.
        Default is None.

    var_w_name : String, list or tuple of strings (or None), optional
        Name of the sampling-weight variable, given as a string or a one-element list or tuple.
        Required at initialization when 'gen_weighted' is True. The named column must then be
        present in both training and prediction data.
        When 'gen_weighted' is False, the supplied name is ignored.
        Default is None.

    var_z_name_cont : String, list or tuple of strings (or None), optional
        Names of ordered variables with many values to define causal heterogeneity. They will be
        discretized and (dependening ``'p_gates_smooth'``) also treated as continuous. If not
        already included in ``'var_x_name_ord'``, they will be added to the list of features.
        Default is None.

    var_z_name_ord : String, list or tuple of strings (or None), optional
        Names of ordered variables with not so many values to define causal heterogeneity. If not
        already included in ``'var_x_name_ord'``, they will be added to the list of features.
        Default is None.

    var_z_name_unord : String, list or tuple of strings (or None), optional
        Names of unordered variables with relatively few values used to define causal
        heterogeneity. Variables not already included in ``'var_x_name_unord'`` are
        added to the unordered features.
        The default is None.

    var_y_tree_name : String, list or tuple of strings (or None), optional
        Outcome used to build trees. Supply one name as a string or a one-element list or tuple.
        If not already in 'var_y_name', it is added to the outcome list.
        If None, the first outcome in 'var_y_name' is used.
        The resolved selection must contain exactly one outcome; otherwise initialization raises
        ValueError. Use 'var_y_name' to request estimation for multiple outcomes.
        In treatment-version ridge regressions, this outcome also selects the common penalty.
        Version-adjusted weights are shared across the requested outcomes; regression residuals
        for inference are computed separately for each outcome. With local centering, the
        corresponding centered outcome is used for both tree building and penalty selection.
        Default is None.

    cf_alpha_reg_grid : Integer or float (or None), optional
        Number of candidate values for the minimum share required in each daughter leaf.
        None or values below 1 use 1; other numeric values are rounded to the nearest integer.
        If ``'cf_tune_all'`` is True, request at least 3 values before checking the bounds.
        With one value, use the mean of the normalized bounds (0.10 with the default bounds).
        Otherwise, use an equally spaced grid including both bounds and select the value by
        out-of-bag evaluation of the objective function. If the normalized minimum is at least
        the maximum, use only the minimum.
        Default (or None) is 1.

    cf_alpha_reg_max : Float (or None), optional
        Upper bound for the minimum share required in each daughter leaf.
        None or values outside 0 <= value < 0.5 use 0.15. A bound below the normalized
        ``'cf_alpha_reg_min'`` is raised to that minimum.
        Default (or None) is 0.15.

    cf_alpha_reg_min : Float (or None), optional
        Lower bound for the minimum share required in each daughter leaf.
        None or values outside 0 <= value < 0.5 use 0.05.
        Default (or None) is 0.05.

    cf_alpha_reg_type : String (or None), optional
        Alpha-regularity rule used to determine admissible splits.
        'pooled' applies alpha regularity to all observation independent of their treatment.
        'treatment' requires each daughter to contain at least
        max(cf_n_min_treat, ceil(alpha * parent treatment count)) observations of each treatment.
        These are unweighted counts; the total minimum leaf size also applies.
        If no admissible candidate exists, splitting stops.
        Explicit 'treatment' is available for discrete treatments only.
        Default is None: 'treatment' for discrete treatments and 'pooled' for continuous treatments.
        Explicit 'treatment' with continuous treatments raises an error.

    cf_boot : Integer or float (or None), optional
        Number of trees in each candidate causal forest. Also used for auxiliary forests,
        including common-support estimation and prognostic-score matching.
        None or values below 1 use 1000; other numeric values are rounded to the nearest integer.
        Default (or None) is 1000.

    cf_chunks_maxsize : Integer or float (or None), optional
        Target maximum size of a forest chunk, counting construction and outcome-sample rows.
        Larger samples are split randomly into approximately equal chunks. A forest is trained in
        each chunk, and the resulting effect estimates are averaged.
        Numeric values of at least 100 are rounded to the nearest integer. None and values below 100
        select the automatic limit:

        .. math::

            \\operatorname{round}\\left(B + \\frac{\\max(N-B,0)^{0.8}}{K-1}\\right)

        B is 100'000 if ``'_int_low_memory_predict'`` is False and 250'000 if it is True.
        N is the effective training sample size used during initialization, adjusted for separate
        samples used for feature selection, common-support estimation and local centering.
        K is the number of treatments.
        The actual number of chunks is computed from the rows available for forest training after
        preprocessing. If that count is at most the chosen limit, one forest chunk is used.
        Default is None.

    cf_compare_only_to_zero : Boolean (or None), optional
        If True, restrict pairwise terms in the forest objective to comparisons involving the
        first treatment (usually the control group). This affects the MCE terms when
        ``'cf_mce_vart'`` is 1 or 3 and the effect-heterogeneity criterion when it is 2.
        Treatment-specific MSE terms are retained for all treatments. With two treatments or
        ``'cf_mce_vart'`` set to 0, this option has no effect.
        Default (or None) is False.

    cf_n_min_grid : Integer or float (or None), optional
        Requested number of candidate minimum leaf sizes. None or values below 1 select 1; other
        values are rounded. If ``'cf_tune_all'`` is True, the requested count is at least 3.
        With one candidate, use the rounded average of the adjusted ``'cf_n_min_min'`` and
        ``'cf_n_min_max'`` bounds. Otherwise, use an equally spaced grid between those bounds,
        rounded to integers. Duplicate values are removed, so the final count can be smaller.
        With multiple candidates, the value is chosen using the out-of-bag objective function.
        Default (or None) is 1.

    cf_n_min_max : Integer or float (or None), optional
        Initial upper bound of the grid of minimum leaf sizes. None or values below 1 use:

        .. math::

            \\operatorname{round}\\left[K\\max\\left(\\sqrt{m}/10,2\\right)\\right]

        Here m is the estimated number of observations from the smallest treatment group in a
        subsample used to build a tree, accounting for chunking, the share for tree construction,
        tree subsampling and separate samples for feature selection, common-support estimation and
        local centering. K is the number of treatments.
        Explicit numeric values of at least 1 are rounded. The upper bound is raised to at least the
        lower bound. For discrete treatments, both bounds are at least ``'cf_n_min_treat'`` times K.
        The final leaf sizes follow the grid rule described under ``'cf_n_min_grid'``.
        Default is None.

    cf_n_min_min : Integer or float (or None), optional
        Initial lower bound of the grid of minimum leaf sizes. None or values below 1 use:

        .. math::

            \\operatorname{round}\\left[K\\max\\left(m^{0.4}/10,1.5\\right)\\right]

        Here m and K are defined under ``'cf_n_min_max'``. Explicit numeric values of at least 1 are
        rounded. For discrete treatments, the bound is at least ``'cf_n_min_treat'`` times K.
        The final leaf sizes follow the grid rule described under ``'cf_n_min_grid'``.
        Default is None.

    cf_n_min_treat : Integer or float (or None), optional
        Minimum observations per treatment in a leaf during tree construction for discrete
        treatments. Ignored for continuous treatments. This parameter is not tuned on a grid.
        Higher values reduce the risk of missing treatment observations when filling the leaves.
        Explicit numeric values of at least 1 are rounded. None or values below 1 select:

        .. math::

            \\operatorname{round}\\left[\\max\\left(\\frac{L+U}{20K},1\\right)\\right]

        L and U are the initialized ``'cf_n_min_min'`` and ``'cf_n_min_max'`` bounds, with U at
        least L, before imposing the per-treatment requirement or constructing the final grid.
        K is the number of treatments. Each candidate total minimum leaf size is at least K times
        the resolved per-treatment minimum.
        Default is None.

    cf_match_nn_prog_score : Boolean (or None), optional
        Choice of variables for nearest neighbour matching, used when ``'cf_mce_vart'`` is 1 or 3.
        True : Use one prognostic score per treatment group if the covariate matrix has at least
        twice as many columns as treatment groups used for matching. Otherwise, use covariates.
        Scores predict ``'var_y_tree_name'`` using random forests fitted in each treatment group.
        If local centering is enabled, the scores use the outcome before centering.
        False : Use covariates directly.
        Unordered covariates are dummy-coded before counting columns or estimating the scores.
        Both choices use Mahalanobis matching on the selected variables.
        Default (or None) is True.

    cf_nn_main_diag_only : Boolean (or None), optional
        Covariance matrix used for nearest neighbour matching on covariates or prognostic scores.
        True : Use only its main diagonal, ignoring covariances between matching variables.
        False : Use the full matrix, subject to the existing singularity safeguards.
        Relevant when ``'cf_mce_vart'`` is 1 or 3.
        Default (or None) is False.

    cf_m_grid : Integer or float (or None), optional
        Requested number of nominal candidate-variable counts to compare using the out-of-bag
        objective. None or values below 1 select 1; other values are rounded.
        With one value, use the rounded average of the counts derived from ``'cf_m_share_min'``
        and ``'cf_m_share_max'``. With more values, use a logarithmically spaced grid between
        those counts, rounded to integers and with duplicates removed.
        If ``'cf_tune_all'`` is True, request at least 3 values; equal bounds produce one value.
        Default (or None) is 1.

    cf_m_random_poisson : Boolean (or None), optional
        Number of candidate variables used at each new split of a tree.
        True: Draw 1 + Poisson(m - 1) for every positive nominal count m.
        The cf_m_share parameters determine m. Cap the draw at the available variable count.
        Before capping, the expected candidate count is m.
        False: Use the fixed nominal count m, capped at the available variable count.
        Always-in priority applies to both choices: if the candidate count does not exceed the
        always-in count, draw only from that set without replacement. Otherwise include all
        always-in variables and draw the remaining candidates from the other variables.
        Default (or None) is True.

    cf_m_share_max : Float (or None), optional
        Upper share for the grid of nominal candidate-variable counts. None or values outside
        (0, 1] use 0.6. Multiply the share by the number of available covariates and round.
        Raise the resulting count to at least the count derived from ``'cf_m_share_min'``.
        Actual split-variable counts follow ``'cf_m_random_poisson'``.
        Default (or None) is 0.6.

    cf_m_share_min : Float (or None), optional
        Lower share for the grid of nominal candidate-variable counts. None or values outside
        (0, 1] use 0.1. Multiply the share by the number of available covariates, round, and
        retain at least one variable. Actual split-variable counts follow ``'cf_m_random_poisson'``.
        Default (or None) is 0.1.

    cf_mce_vart : Integer (or None), optional
        Splitting rule for tree building:
        0 : mse's of regression only considered.
        1 : mse+mce criterion (default).
        2 : -var(effect): heterogeneity maximising splitting rule of Wager & Athey (2018).
        3 : randomly switching between outcome-mse+mce criterion & penalty functions.
        Default (or None) is 1.

    cf_mce_type : String (or None), optional
        Definition of MCE when cf_mce_vart is 1 or 3.
        'covariance': Covariance of matched outcomes, centered at their matched-sample means.
        'cross_error': Mean product of matched-outcome errors, centered at the observed
        treatment-specific leaf means, as in Lechner and Mareckova (2024).
        Default (or None) is 'cross_error'.

    cf_p_diff_penalty : Float (or None), optional
        Penalty function (depends on the value of `mce_vart`).

        `mce_vart == 0`
            Irrelevant (no penalty).
    
        `mce_vart == 1`
            Multiplier of penalty (in terms of `var(y)`).
            0 : No penalty.
            None :
    
            .. math::
    
                \\frac{2 \\times (\\text{n} \\times \\text{subsam\\_share})^{0.9}}{\\text{n} \\times
                \\text{subsam\\_share}} \\times \\sqrt{\\frac{\\text{no\\_of\\_treatments} \\times (\\text{no\\_of\\_treatments} - 1)}{2}}  
    
        `mce_vart == 2`
            Multiplier of penalty (in terms of MSE(y) value function without splits).
            0 : No penalty.
            None : 100 times the default multiplier for ``'cf_mce_vart'`` == 1.
    
        `mce_vart == 3`
            Probability of using p-score (0-1). None : 0.5. Increase value if balancing tests
            indicate problems. 
        
        Default is None.
    
    cf_penalty_type : String (or None), optional
        Type of penalty function.
        'mse_d':  MSE of treatment prediction in daughter leaf (new since 0.7.0)
        'diff_d': Penalty as squared leaf difference (as in Lechner, 2018)
        Note that an important advantage of ``'mse_d'`` that it can also be used for tuning (due to
        its computation, this is not possible for 'diff_d').
        Default (or None) is ``'mse_d'``.
        Approximately constant observed outcomes in parent leaf stop splitting unless the node
        objective uses a treatment penalty: a positive effective multiplier for the additive
        criterion, or selection of the treatment criterion under random switching.
        The outcome-constancy comparison and its tolerances are unchanged when this stopping
        condition applies.

    cf_random_thresholds : Integer (or None), optional
        Random thresholds for ordered variables in the initial split search. Draw new thresholds
        at each node. Fewer thresholds reduce computation but can affect the selected split.
        If the initial search finds no admissible split, retry all available variables and all
        supported thresholds before stopping. The fallback retains the existing categorical
        ordering, treatment minima, and alpha restrictions.
        0 : Use all supported thresholds in the initial search.
        > 0 : Limit the initial random threshold sample for each ordered variable to this count.
        None or a negative integer : Use :math:`\\operatorname{round}(4+n^{0.2})`.
        Here n is the estimated effective training sample size per forest chunk, including both
        construction and filling observations. It accounts for separate samples used for feature
        selection, common-support estimation and local centering.
        Default is None.

    cf_subsample_factor_forest : Float (or None), optional
        Multiplier f for the per-tree construction subsample share S. None or nonpositive values
        use f = 1. For n, the estimated effective training sample size per forest chunk:

        .. math::

            B = \\min\\left(\\frac{4(n/2)^{0.85}}{n},0.67\\right),\\qquad
            S = \\max\\left(\\min(fB,0.67),\\frac{2\\sqrt{n/2}}{n}\\right).

        The value n accounts for separate samples used for feature selection, common-support
        estimation and local centering. It includes both construction and filling observations.
        For individual sampling, draw round(S times the available construction-row count) rows
        without replacement. With panel sampling, apply S to the cluster count instead and include
        all construction rows from the selected clusters.
        Default (or None) is 1.

    cf_subsample_factor_eval : Float or Boolean (or None), optional
        Scale of the per-tree filling sample. False requests all available filling rows.
        None or True resolves to factor 2. A numeric factor below 0.01 also requests all rows.
        Other numeric factors must not be NaN.
        With cf_subsample_power_eval > 0.85, factor 2 gives the calibrated default scale;
        factor 1 halves the uncapped filling count. This rule is independent of
        cf_subsample_factor_forest. With cf_subsample_power_eval=0.85, the factor instead
        multiplies the construction share, reproducing the filling rule of version 0.10.0.
        Smaller filling samples can reduce computation and storage at the cost of additional
        subsampling variability and potentially more empty treatment cells. Default is None.

    cf_subsample_power_eval : Float (or None), optional
        Filling-sample exponent. None resolves to 0.90. Valid values are 0.85 through 1.
        The construction exponent remains 0.85. Setting this parameter to 0.85 selects the
        previous construction-share rule, including its existing bounds and rounding.
        For a larger exponent b and resolved filling factor f, the filling count is

        .. math::

            s = \\min\\left(n_f,\\max\\left(1,\\operatorname{round}
                \\left[\\frac{f}{2}4^{(1-b)/0.15}n_f^b\\right]\\right)\\right).

        Here n_f is the actual available filling-sample size in the current forest chunk
        and estimation round. The default b=0.90 and f=2 gives scale 4**(2/3), about 2.519842,
        preserving the default full-filling threshold of 0.10.0 with equal construction and
        filling halves. Filling rows are sampled without replacement when the count is smaller
        than n_f. Explicit full filling takes precedence over this calculation. An exponent
        of 1 with factor 2 also uses every filling row. The exponent ordering alone does not
        establish the paper's stronger normality rate restriction or its other assumptions.
        Older saved configurations missing this parameter retain the previous rule (0.85).
        Default is None.

    cf_tune_all : Boolean (or None), optional
        Enable grid tuning of the candidate-variable count, minimum leaf size and minimum share
        required in each daughter leaf. If True, raise each requested grid size to at least 3:
        ``'cf_m_grid'``, ``'cf_n_min_grid'`` and ``'cf_alpha_reg_grid'``.
        Larger requested grid sizes are retained. The final grids can contain fewer values when
        bounds coincide or integer rounding produces duplicates.
        Default (or None) is False.

    cf_vi_oob_yes : Boolean (or None), optional
        Compute causal-forest variable importance by permuting covariates and evaluating the
        resulting change in the out-of-bag objective. Results are reported as percentage changes
        relative to the unpermuted objective.
        Computation requires ``'_int_with_output'`` to be True and uses only the first forest
        chunk in the regular estimation round, before any exchange of construction and outcome
        samples for efficient estimation.
        Default (or None) is False.

    cs_type : Integer (or None), optional
        Common-support method based on random-forest treatment-probability estimates.
        With train_iv() and predict_iv(), use instrument probabilities instead. For IV estimation,
        the probabilities and groups mentioned in the 'cs_' options refer to instrument values.
        0 : No common-support adjustment.
        1 : Intersect the treatment-specific probability ranges, using minima/maxima or quantiles
        controlled by ``'cs_quantil'`` and adjusted by ``'cs_adjust_limits'``. The first treatment's
        probability is excluded from this check.
        2 : Require every treatment probability to lie between ``'cs_min_p'`` and 1 - cs_min_p,
        including both bounds.
        Observations outside the bounds are removed from training and prediction data.
        With ``'lc_cs_cv'`` True, use cross-fitted predictions for the construction sample and
        average the fold models' predictions for the filling and prediction samples. Otherwise,
        fit the probability model on a separate sample controlled by ``'lc_cs_share'``.
        Values outside 0, 1 and 2 use 1. Continuous treatments always use 0.
        Default (or None) is 1.

    cs_adjust_limits : Float (or None), optional
        Relax the probability bounds when ``'cs_type'`` is 1. For adjustment a, multiply lower
        bounds by 1 - a and upper bounds by 1 + a, then clip both to [0, 1].
        The final bounds are the intersection of these adjusted treatment-specific ranges.
        If None, use (number of treatments - 2) * 0.05, giving no adjustment for two treatments.
        A negative value raises ValueError during training initialization, even if support
        adjustment is disabled or ``'cs_type'`` is 2.
        Default is None.

    cs_max_del_train : Float (or None), optional
        Maximum share of observations removed for lack of common support during training.
        Check the construction and filling samples separately, after setting aside any sample
        used for common-support estimation. If either deletion share exceeds this value, raise
        ValueError. Equality is allowed.
        The check applies when ``'cs_type'`` is positive, independently of output settings.
        It does not limit deletions from prediction data.
        None or values outside 0 < value <= 1 use 0.5.
        Default (or None) is 0.5.

    cs_min_p : Float (or None), optional
        Probability bound used only when ``'cs_type'`` is 2. Remove an observation if any
        treatment probability is below cs_min_p or above 1 - cs_min_p; equality is retained.
        None or values outside 0 <= cs_min_p <= 0.5 use 0.01.
        Default (or None) is 0.01.

    cs_quantil : Float (or None), optional
        Quantile level q for common-support bounds when ``'cs_type'`` is 1.
        q = 1 uses the minimum and maximum within each treatment group. For q < 1, use
        the (1 - q) quantile as the lower bound and the q quantile as the upper bound.
        Bounds are adjusted by ``'cs_adjust_limits'`` and intersected across treatment groups.
        None or values outside [0, 1] use 1. Values below 0.5 reverse the two quantile levels;
        without sufficient adjustment, this can leave no observations on support.
        Default (or None) is 1.

    cs_detect_const_vars_stop : Boolean (or None), optional
        If True and ``'cs_type'`` is positive, check covariates for almost zero variation within
        each treatment group. The check runs separately on the tree-construction and tree-filling
        samples before common-support trimming.
        A within-group sample standard deviation below 1e-8 raises NoVariationInTreatmentState.
        Review the affected covariates and observations before adjusting the data or specification.
        False disables this preliminary check; probability-based trimming follows ``'cs_type'``.
        Default (or None) is True.

    ct_grid_dr : Integer (or None), optional
        Requested number of quantile points for the positive part of the continuous treatment
        in the dose-response discretization. A zero point is added to the quantile grid.
        Positive Python integers are used; other inputs use 100. The requested count is raised
        to at least ``'ct_grid_w'`` after both values have been normalized.
        This grid also sets interpolation density between weight-grid points; the final
        dose-response evaluation grid can have a different number of points.
        Default (or None) is 100.

    ct_grid_nn : Integer (or None), optional
        Number of quantile points for the positive part of the continuous treatment, used for
        nearest-neighbour matching. A zero point is added, giving one additional grid entry.
        Positive Python integers are used; other inputs use 10.
        Default (or None) is 10.

    ct_grid_w : Integer (or None), optional
        Number of quantile points for the positive part of the continuous treatment, used for
        weight estimation. A zero point is added, giving one additional grid entry.
        Positive Python integers are used; other inputs use 10.
        Default (or None) is 10.

    dc_clean_data : Boolean (or None), optional
        If True, keep the columns needed for training or prediction and remove rows with missing
        values in those retained columns. Covariate screening follows 'dc_screen_covariates'.
        Feature preprocessing runs first and rejects missing or nonnumeric values in any supplied
        column, including unused columns, regardless of this setting. Main-treatment labels are
        recoded before this validation when treatment information is used.
        Remove or impute missing values and remove or numerically encode other nonnumeric columns
        before calling a training or prediction method; this option does not bypass validation.
        Default (or None) is True.

    dc_check_perfectcorr : Boolean (or None), optional
        If True, mark a feature for removal during covariate screening when
        its absolute Pearson correlation with an earlier feature in the
        screening order is greater than 0.999.
        Both positive and negative correlations are considered.
        Default (or None) is True.

    dc_min_dummy_obs : Integer or float (or None), optional
        During covariate screening, mark features with exactly two distinct
        observed values for removal if either value occurs fewer times than
        this threshold. The values need not be coded as 0 and 1.
        None or values below 1 use 10; other values are rounded to an integer
        using Python's round().
        Set to 1 to disable this filter. Setting 0 uses the default threshold.
        Default (or None) is 10.

    dc_screen_covariates : Boolean (or None), optional
        Screen and clean covariates.
        Default (or None) is True.

    fs_yes : Boolean (or None), optional
        If True, select features before building the causal forest using auxiliary random forests.
        With train(), their targets are the treatment and the outcome in 'var_y_tree_name'.
        With train_iv(), selection uses the first-stage problem: the outcome target is the
        treatment in 'var_d_name', and the treatment target is the instrument in 'var_iv_name'.
        The selected features are used for both the first-stage and reduced-form forests.
        In either case, 'y_relevant' in 'fs_rel_vi_keep_if' omits the treatment-target model.
        Measure importance by the loss in test-sample prediction score after
        permuting a feature; dummy columns of unordered features move together.
        Regression uses R2. Classification uses negative MSE between class
        indicators and predicted probabilities if fs_mse_for_classifier is
        True, and accuracy otherwise.
        Apply fs_rel_vi_threshold_y and fs_rel_vi_threshold_d according to
        fs_rel_vi_keep_if. The default 'y_or_d_relevant' removes a feature
        only if its importance is at or below the respective threshold for
        both the outcome and the treatment.
        Variables in var_x_name_always_in_ord, var_x_name_always_in_unord,
        var_x_name_remain_ord and var_x_name_remain_unord are protected,
        as are variables defining GATE groups or used for BGATE balancing.
        Remove at most one feature per iteration and refit before considering
        another removal. Stop when no feature qualifies or at most two
        original features remain.
        Default (or None) is False.

    fs_rel_vi_threshold_y : Integer or float (or None), optional
        Threshold for outcome feature importance. The deletion rule combines
        this with fs_rel_vi_threshold_d according to fs_rel_vi_keep_if.
        Values in (0, 1] are used directly; values in (1, 100] are divided
        by 100. Thus 0.05 and 5 both give 0.05, while 1 remains 1.
        None, nonpositive values and values above 100 use 0.
        Default (or None) is 0.

    fs_rel_vi_threshold_d : Integer or float (or None), optional
        Threshold for treatment feature importance, with the same input
        conversion as fs_rel_vi_threshold_y. Combined with the outcome threshold
        according to fs_rel_vi_keep_if; ignored for 'y_relevant'.
        Default (or None) is 0.

    fs_rel_vi_keep_if : String (or None), optional
        Define when an unprotected feature qualifies for removal:
        'y_relevant': Outcome importance is at or below fs_rel_vi_threshold_y.
        The treatment model is not estimated.
        'y_or_d_relevant': Both outcome and treatment importance are at or
        below fs_rel_vi_threshold_y and fs_rel_vi_threshold_d, respectively.
        'y_and_d_relevant': Either outcome or treatment importance is at or
        below its respective threshold.
        None or an unrecognized string uses 'y_or_d_relevant'.
        Default is 'y_or_d_relevant'.

    fs_mse_for_classifier : Boolean (or None), optional
        Score used for classifiers during feature selection.
        If True, use negative mean squared error between one-hot class indicators and predicted
        class probabilities, averaged over observations and classes.
        Include all classes found in training or test data. A class absent from training has
        predicted probability zero; its test observations still contribute their squared error.
        If False or None, use classification accuracy. Regression models use R2.
        Default is True.

    fs_other_sample : Boolean (or None), optional
        True : Random sample from training data used. These observations will not be used for the
        causal forest.
        False : Use the same sample as used for causal forest estimation.
        Default (or None) is True.

    fs_other_sample_share : Float (or None), optional
        Share of training observations reserved when feature selection runs
        with fs_other_sample=True.
        Use a value greater than 0 and at most 0.5. None, negative values
        and values above 0.5 use 0.33.
        A value of 0 passes initialization but raises ValueError when the
        separate sample for feature selection is drawn.
        To reuse the same observations for feature selection and subsequent
        estimation, set fs_other_sample=False.
        Default (or None) is 0.33.

    gen_d_type : String (or None), optional
        Type of treatment. ``'discrete'``: Discrete treatment. ``'continuous'``: Continuous
        treatment.
        Default (or None) is ``'discrete'``.

    gen_ate_eff : Boolean (or None), optional
        Compute ATEs from two rounds with the roles of the tree-building
        and tree-filling samples exchanged, and average the estimates.
        Conservative inference averages the two variance estimates.
        During initialization, gen_gate_eff=True, gen_iate_eff=True or
        gen_qiate_eff=True also enables this option, even if gen_ate_eff
        is False or None.
        Default (or None) is False unless enabled by one of these flags.

    gen_gate_eff : Boolean (or None), optional
        Compute more efficient GATEs / BGATEs / CBGATEs (effects are estimated twice and averaged
        where the role of ``'tree_building'`` and ``'tree_filling'`` sample is exchanged;
        X-fitting). Conservative inference is provided for these parameters by averaging the
        variances of the estimators.
        Default (or None) is False.

    gen_iate_eff : Boolean (or None), optional
        Estimate IATEs in two rounds, exchanging the tree-building and tree-filling samples, and
        average the two estimates. Requested IATE-minus-ATE deviations are also averaged over
        both rounds.
        If standard errors are requested, the variance estimate from the regular round is
        multiplied by 1/2; no separate variance estimate is computed in the additional round.
        Default (or None) is False.

    gen_qiate_eff : Boolean (or None), optional
        Estimate QIATEs in two rounds, exchanging the tree-building and tree-filling samples,
        and average the estimates. Requested QIATE variances are averaged over both rounds.
        Each round ranks individual effects from its own forest. Individual predictions and any
        variances needed for rank adjustment are computed even when 'gen_iate_eff' is False.
        The 'gen_iate_eff' setting separately controls averaging of the reported IATEs.
        Default (or None) is False.

    gen_mp_parallel : Integer or float (or None), optional
        Requested number of workers for CPU parallel computations.
        Python integers and floats at most 1.5 select one worker; larger
        values are rounded to the nearest integer with round().
        None or a nonnumeric value selects 80% of logical CPU cores,
        rounded to the nearest integer.
        For backend selection, see _int_mp_backend and _int_mp_use_old_ray.
        On Windows with '_int_mp_backend' set to 'joblib', initialization rejects worker counts
        above 63 with ValueError. If '_int_mp_use_old_ray' is False, counts from 61 to 63 are also
        rejected, with RuntimeError.
        During training initialization, if the sample has more than
        _int_obs_bigdata observations and the current worker count exceeds 1,
        replace that count with int(0.75 * current_count).
        This applies to automatic and explicitly requested counts.
        Default is None.

    gen_outfiletext : String (or None), optional
        Base name for the detailed '<name>.txt' and condensed '<name>_Summary.txt' output files.
        With '_int_with_output' enabled, initialization deletes existing files at these paths,
        including when 'gen_output_type' is 0. See '_int_output_no_new_dir' for directory reuse.
        Default (or None) is 'txtFileWithOutput'.

    gen_outpath : String or Pathlib object (or None), optional
        Directory for text output and figures.
        With None, the base directory is 'output' below the current working directory.
        Missing directories are created. Directory reuse and numeric
        suffixes are controlled by _int_output_no_new_dir.
        Only used when _int_with_output is enabled. Otherwise the stored output path is None.
        Default is None.

    gen_output_type : Integer or float (or None), optional
        Destination of standard text output: 0 for terminal, 1 for files, 2 for both.
        Numeric values are converted with int(round(value)); the result must be 0, 1 or 2.
        Other rounded values raise TypeError. Halfway values round to the nearest even integer.
        '_int_with_output' set to False disables standard text output.
        At initialization, the default (or None) is 2. None is not allowed in 'new_keywords'.

    gen_panel_data : Boolean (or None), optional
        If True, treat observations as clustered using ``'var_cluster_name'``, which is required.
        This enables ``'p_cluster_std'`` even if that option is False, and enables cluster sampling
        within trees unless ``'gen_panel_in_rf'`` is False.
        Default (or None) is False.

    gen_panel_in_rf : Boolean (or None), optional
        If True, draw each tree's construction subsample by selecting clusters without replacement.
        Include all available construction rows of each selected cluster within the current chunk.
        If False, sample individual rows. The initial construction/outcome split and forest
        chunking operate on rows in either case.
        Default (or None) is True when ``'gen_panel_data'`` is True. Otherwise, this option is
        forced to False, even when explicitly set to True.

    gen_weighted : Boolean (or None), optional
        Use of sampling weights to be provided in ``'var_w_name'``.
        Default (or None) is False.

    gen_tv_estimator : String (or None), optional
        Regression estimator for treatment-version adjustment within each main treatment.
        'ols': Weighted least squares using the forest weights for the effect being estimated.
        'ridge': Weighted ridge regression with a penalty selected by cross-validation.
        'gen_tv_specification' controls the regressor design; 'gen_tv_cv_k' controls ridge CV folds.
        Treatment versions require discrete treatments and are incompatible with sampling weights
        ('gen_weighted') and cluster-robust inference ('p_cluster_std', including 'gen_panel_data').
        Initialization raises ValueError for these combinations, as it does for 'p_qiate' or 'p_ba'.
        Default (or None) is 'ridge'.
    
    gen_tv_addscores : Boolean (or None), optional
       Add the scores selected by 'gen_tv_score_type' to the OLS or ridge treatment-version
       regressions to account for nonlinearities. See 'gen_tv_score_type' for the predictors
       used to fit these scores.
       Default (or None) is False.
           
    gen_tv_specification : String (or None), optional
        Defines how covariates enter the treatment-version regressions within each main treatment.
        Possible options are ``'interacted'`` or ``'separable'``.
        ``'interacted'``: Include version dummies and their interactions with the covariates,
        allowing both intercepts and covariate slopes to differ between versions.
        ``'separable'``: Include version dummies and covariates without interactions, allowing
        version-specific intercepts but imposing common covariate slopes.
        Default (or None) is 'interacted'.

    gen_tv_cv_k : Integer (or None), optional
        Number of folds used to select the ridge penalty in treatment-version estimation.
        Used only when ``'gen_tv_estimator'`` is 'ridge'.
        When treatment versions are enabled, an explicit value must be a Python integer greater
        than one. Other values raise ValueError; floats are not rounded.
        Default (or None) depends on the size of the training sample (N):
        N < 100'000: 5;  100'000 <= N < 250'000: 4; 250'000 <= N < 500'000: 3; 500'000 <= N: 2.

    gen_tv_min_subtreat : Integer (or None), optional
        Minimum observations per treatment version, within its main treatment, for version
        regression. Counts use the current effect calculation's forest weights and include only
        observations whose weight exceeds the numerical zero tolerance.
        Below this threshold, retain the main-treatment weights for the version's potential outcome.
        Must be a positive Python integer.
        Default (or None) is 10.

    gen_tv_penalize_version: Boolean or list or tuple of Booleans (or None), optional
        Determines whether the coefficients of version dummies are penalized in a particular main
        treatment. Only relevant if ``'gen_tv_estimator'`` == 'ridge' is used.
        A Boolean or a one-element list or tuple applies the same setting to every main treatment.
        Otherwise, a list or tuple must contain one Boolean per main treatment.
        True: Coefficients of the version dummies in the version ridge regression are (also)
        penalized. Could be useful, when there are very many treatment versions.
        False: Only coefficients of covariates are penalized (including treatment covariate
        interactions).
        Default (or None) is False.

    gen_tv_score_estimator: String or None, optional
        The estimator used for prognostic scores. Possible choices are scikit-learn's regression
        methods 'RandomForest', 'RandomForestNminl5', 'RandomForestNminls5', 'SupportVectorMachine',
        'SupportVectorMachineC2', 'SupportVectorMachineC4', 'AdaBoost', 'AdaBoost100',
        'AdaBoost200', 'GradBoost', 'GradBoostDepth6', 'GradBoostDepth12', 'NeuralNet',
        'NeuralNetLarge', 'NeuralNetLarger',
        'Mean' ('Mean' is included for the case when other estimators are not useful). If set to
        'automatic', the estimator with the lowest cross-validated mean squared error (MSE) is
        selected separately for each treatment or treatment version. The number of folds is set
        by ``'gen_tv_score_cv_k'`` and capped at the number of observations in the relevant group.
        Propensity scores use 'RandomForestClassifier'. Prognostic scores use it instead of the
        selected regression method when the score-training sample has fewer than 10 distinct
        outcome values. This count is taken before splitting that sample by treatment or version.
        A group with fewer than three observations uses its mean outcome instead.
        Default (or None) is 'RandomForest'.

    gen_tv_score_type : String, list or tuple of strings (or None), optional
        Defines the propensity and/or prognostic scores to include for versions and/or main
        treatments. Possible options are 'prop_version', 'prog_version', 'prop_main', 'prog_main'.
        A single score or any combination of scores can be included.
        Main scores are trained on all forest covariates.
        With 'gen_tv_score_version_local' set to False, version scores use only the covariates
        selected for version estimation, excluding main scores.
        With True, version scores also use any selected main scores as predictors.
        Only relevant if ``'gen_tv_addscores'`` is True.
        Default (or None) is ('prop_version', 'prop_main').

    gen_tv_score_version_local : bool or None, optional
        If True, fit version-specific scores separately for each aggregate effect. In this mode,
        IATE version regressions omit version-specific scores; no global scores replace them.
        If False, fit version-specific score models on the tree-building sample and use their
        predictions in both aggregate-effect and IATE version regressions.
        This option does not change the use of main-treatment scores.
        Only relevant if 'gen_tv_addscores' is True and 'gen_tv_score_type' includes 'prop_version'
        or 'prog_version'.
        Default (or None) is False.

    gen_tv_score_cv_k : Integer or float (or None), optional
        Requested fold count for automatic prognostic regression selection and local score
        cross-fitting. Regression selection caps the count at the observations in each treatment
        or version group. For global scores, only automatic prognostic regression selection uses
        this setting; propensity and classifier-based prognostic scores do not use folds.
        With 'gen_tv_score_version_local' set to True, also controls stratified cross-fitting of
        local propensity and prognostic scores, capped at the size of the smallest version group.
        If any version has fewer than two observations, cross-fitting is skipped.
        Numeric values are rounded to the nearest integer. None or values rounding to fewer than
        two folds select the default, which depends on the size of the training sample (N):
        N < 100'000: 5;  100'000 <= N < 250'000: 4; 250'000 <= N < 500'000: 3; 500'000 <= N: 2.

    lc_yes : Boolean (or None), optional
        If True, subtract predictions of each outcome from its observed values before forest
        training. Predictions use the available covariates, excluding the treatment variable.
        With train_iv(), also center the binary treatment for the first stage. Outcome centering
        for the reduced form is fitted separately; the instrument itself is not centered.
        The regression method is specified by ``'lc_estimator'`` (default 'RandomForest').
        Automatic method selection is used only when ``'lc_estimator'`` is 'automatic'.
        If ``'lc_cs_cv'`` is True, cross-fitting is used with the fold count controlled by
        ``'lc_cs_cv_k'``. Otherwise, the centering models are fitted on a separate sample with
        share ``'lc_cs_share'``.
        Default (or None) is True.

    lc_estimator : String (or None), optional
        The estimator used for local centering. Possible choices are
        scikit-learn's regression methods 'RandomForest', 'RandomForestNminl5',
        'RandomForestNminls5', 'SupportVectorMachine', 'SupportVectorMachineC2',
        'SupportVectorMachineC4', 'AdaBoost', 'AdaBoost100', 'AdaBoost200',
        'GradBoost', 'GradBoostDepth6', 'GradBoostDepth12', 'NeuralNet', 'NeuralNetLarge',
        'NeuralNetLarger', 'Mean'.
        If set to ``'automatic'``, the estimator with the lowest out-of-sample mean squared error
        (MSE) is selected. Whether this selection is based on cross-validation or a test sample is
        governed by the keyword ``'lc_cs_cv'``. 'Mean' is included for the cases when none of the
        methods have explanatory power.
        Default (or None) is 'RandomForest'.

    lc_uncenter_po : Boolean (or None), optional
        Add local-centering predictions back to potential outcomes in the output.
        The IATE DataFrame receives additional '_un_lc_pot' columns only for the outcome selected by
        'var_y_tree_name'. Existing centered potential-outcome columns and treatment effects remain.
        The ATE text output also reports restored average potential outcomes for every outcome.
        When the reported estimate averages reversed rounds, restoration uses the average
        of their centering predictions; a regular-round estimate uses its own predictions.
        Forced to False when 'lc_yes' is False. Default (or None) is True otherwise.

    lc_cs_cv : Boolean (or None), optional
        Data to be used for local centering & common support adjustment.
        True : Crossvalidation. For local centering with reversed estimation rounds,
        fit/cross-fit on each round's construction sample and predict its filling sample.
        False : Random sample not to be used for forest building. With reversed rounds,
        local-centering model selection and fitting both use this independent sample.
        Default (or None) is True.

    lc_cs_cv_k : Integer or float (or None), optional
        Number of folds for local centering and common-support estimation if ``'lc_cs_cv'`` is True.
        Numeric values are rounded to the nearest integer. None or values rounding to fewer than
        two folds select the default, which depends on the size of the training sample (N):
        N < 100'000: 5;  100'000 <= N < 250'000: 4; 250'000 <= N < 500'000: 3; 500'000 <= N: 2.

    lc_cs_share : Float (or None), optional
        Share reserved for fitting common-support and local-centering models when 'lc_cs_cv' is
        False. Each active procedure reserves this share from the observations available at its
        stage. Reserved observations are excluded from subsequent forest estimation.
        If both procedures run, common support reserves its sample first; local centering then
        reserves this share from the remaining data.
        In train_iv(), first-stage centering reserves this share from the construction sample.
        Reduced-form centering then reserves this share from the remaining construction sample
        and from the filling sample. Thus IV local centering reduces construction data twice.
        Values with 0.0999 < value < 0.9001 are accepted; other numeric values use 0.25.
        Default (or None) is 0.25.

    p_atet : Boolean (or None), optional
        Compute ATEs for specific observed treatment groups. Available for discrete treatments only.
        Prediction data must contain the treatment column. With treatment versions, both columns
        listed in 'var_d_name' must be present.
        If either 'p_atet' or 'p_gatet' is enabled, feature preprocessing requires every main
        treatment group stored during training to occur in the prediction sample. A sample
        containing only a subset raises RuntimeError. This check follows any prediction-row cap
        and precedes common-support trimming.
        Prediction sets both 'p_atet' and 'p_gatet' to False for continuous treatments or if any
        required treatment column is missing.
        Default (or None) is False.

    p_gates_minus_previous : Boolean (or None), optional
        Select the reference for differences reported with GATE, BGATE and CBGATE estimates.
        True compares consecutive evaluation points in their evaluation order.
        The first point has no preceding comparison. False uses the corresponding ATE as reference
        (LATE for IV estimation). Smoothed evaluation points need not be observed feature values.
        Differences use group weights before applying 'p_max_weight_share'.
        This setting controls the '_diff' results and their corresponding standard errors.
        Default (or None) is False.

    p_gates_no_evalu_points : Integer or float (or None), optional
        Fallback number of quantile evaluation points for BGATE and CBGATE features without stored
        evaluation values. Existing evaluation values are reused unchanged.
        Probabilities are equally spaced from 0.01 to 0.99. The count is capped at the number of
        available observations, and duplicate quantile values are removed.
        Discretization is controlled by 'p_max_cats_z_vars'; smoothed evaluation points are
        controlled by 'p_gates_smooth_no_evalu_points'.
        At initialization, None or values < 2 select 50; other numeric values are rounded.
        Updates via new_keywords are rounded and must be between 2 and 100000 inclusive;
        None is not allowed.
        Default (or None) is 50.

    p_gates_smooth : Boolean (or None), optional
        Add smoothed estimates for continuous heterogeneity features alongside estimates based on
        their discretized versions. This applies to GATE, BGATE and CBGATE estimation.
        Evaluation points are controlled by 'p_gates_smooth_no_evalu_points'; smoothing uses an
        Epanechnikov kernel with bandwidth scaled by 'p_gates_smooth_bandwidth'.
        False disables the additional smoothed estimates.
        Default (or None) is True.

    p_gates_smooth_bandwidth : Float (or None), optional
        Multiplier for the rule-of-thumb bandwidth used to smooth GATE, BGATE and CBGATE estimates
        over continuous features when 'p_gates_smooth' is True. Larger values increase smoothing.
        At initialization, None or values <= 0 select 1. Updates via new_keywords require a
        positive numeric value; None is not allowed.
        Default (or None) is 1.

    p_gates_smooth_no_evalu_points : Integer or float (or None), optional
        Maximum number of evaluation points for smoothed GATE, BGATE and CBGATE estimates over
        continuous features when 'p_gates_smooth' is True.
        If the number of distinct observed values is at most this limit, all such values are used.
        Otherwise, sample quantiles at equally spaced probabilities from 0.01 to 0.99 are used.
        Duplicate quantile values are removed, so the resulting number of points can be smaller.
        At initialization, None or values < 2 select 50; other numeric values are rounded.
        Updates via new_keywords are rounded and must be between 2 and 100000 inclusive;
        None is not allowed.
        Default (or None) is 50.

    p_gatet : Boolean (or None), optional
        Compute GATEs for specific observed treatment groups. The treatment-type and prediction-data
        requirements, including automatic disabling, are the same as for 'p_atet'.
        Requires at least one variable in 'var_z_name_ord', 'var_z_name_unord' or 'var_z_name_cont'.
        At initialization, True also enables 'p_atet', even if it was set to False. If no
        heterogeneity variables are supplied, 'p_gatet' is then disabled but 'p_atet' stays enabled.
        'p_ate_no_se_only' overrides both options and disables them.
        Default (or None) is False.

    p_bgate : Boolean (or None), optional
        Estimate a GATE balanced in the features selected by ``'var_x_name_balance_bgate'``.
        At initialization, this option is forced to False if no heterogeneity variables are given.
        With exactly one heterogeneity variable, it is also forced to False if the balancing list
        is omitted or contains only that variable.
        Default (or None) is False.

    p_cbgate : Boolean (or None), optional
        Estimate CBGATEs, balancing all other features for each heterogeneity variable.
        Requires at least one variable in 'var_z_name_ord', 'var_z_name_unord' or 'var_z_name_cont'.
        Disabled at initialization if none are supplied or 'p_ate_no_se_only' is True.
        Reference-sample subsampling follows 'p_bgate_sample_share'.
        Default (or None) is False.

    p_bgate_sample_share : Float (or None), optional
        Controls random subsampling of the reference data for BGATE and CBGATE estimation.
        During prediction setup, None or values <= 0 select 1 if n < 1000; otherwise:

        .. math::

            \\frac{1000 + (n - 1000)^{3/4}}{n}

        Here n is the prediction sample size when the setting is initialized.
        For a variable with m evaluation points, the base reference-sample size is
        min(N, max(50, floor(N * value / m))), where N is the number of observations available
        when constructing the reference sample. Thus, at least 50 observations are used if
        available; for N <= 50, all observations are used.
        A smaller base sample is drawn without replacement using a fixed random seed. The same
        base sample is reused across all m evaluation points, giving m times its size in
        reference rows. The minimum of 50 can override the requested sampling fraction.
        Updates via new_keywords require 0 < value <= 1; None is not allowed.
        Default is None.

    p_max_cats_z_vars : Integer or float (or None), optional
        Maximum number of groups for discretizing features specified in 'var_z_name_cont'.
        During training setup, None or values < 1 select round(n_train_eff ** 0.3), where
        n_train_eff is the effective training sample size per forest chunk at initialization.
        Explicit values >= 1 are rounded to the nearest integer.
        Features with at most this many distinct training values retain their original values.
        Otherwise, quantiles of the training data define bins, represented by their training means.
        Duplicate quantile boundaries can reduce the number of bins.
        Updates via new_keywords are not supported.
        Default is None.

    p_iate : Boolean (or None), optional
        IATEs will be estimated.
        Default (or None) is True.

    p_iate_se : Boolean (or None), optional
        Standard errors of IATEs will be estimated.
        Default (or None) is False.

    p_iate_m_ate : Boolean (or None), optional
        Estimate IATE deviations from ATE using differences between their estimation weights.
        The calculation uses IATE and ATE weights saved before capping by 'p_max_weight_share'.
        It can therefore differ from subtracting the separately reported IATE and ATE estimates.
        Requires '_int_low_memory_predict' set to False. Standard errors follow 'p_iate_se'.
        Disabled at initialization if IATE estimation is off or 'p_ate_no_se_only' is True.
        Default (or None) is False.

    p_qiate : Boolean (or None), optional
        Estimate QIATEs for discrete treatments. This enables ``'p_iate'`` at initialization.
        Requires ``'_int_low_memory_predict'`` and ``'p_ba'`` to be False. Treatment versions are
        unsupported, and :meth:`~ModifiedCausalForest.predict_iv` does not return QIATE estimates.
        QIATE-specific rank adjustment is available through ``'p_qiate_bias_adjust'``.
        Default (or None) is False.

    p_qiate_se : Boolean (or None), optional
        Estimate standard errors when ``'p_qiate'`` is True.
        Unless ``'p_ate_no_se_only'`` is True, enabling this option also sets ``'p_iate_se'`` to
        True at initialization, even if ``'p_iate_se'`` was False.
        This option does not enable QIATE estimation itself.
        Default (or None) is False.

    p_qiate_m_mqiate : Boolean (or None), optional
        Estimate QIATE(q) - QIATE(0.5) for each requested quantile q when ``'p_qiate'`` is True.
        The reference at q = 0.5 is computed separately, including when the quantile grid does
        not contain 0.5. Standard errors of the differences follow ``'p_qiate_se'``.
        Default (or None) is False.

    p_qiate_m_opp : Boolean (or None), optional
        Estimate QIATE(q) - QIATE(1 - q) on the full grid selected by ``'p_qiate_no_of_quantiles'``.
        Each quantile is paired with its reflected level 1 - q, including quantiles above 0.5.
        Used when ``'p_qiate'`` is True. Standard errors of the differences follow ``'p_qiate_se'``.
        Default (or None) is False.

    p_qiate_no_of_quantiles : Integer (or None), optional
        Number Q of QIATE quantile levels. Python integers of at least 10 are used; None and
        other inputs select 99 at initialization.
        Quantile levels are (j + 0.5) / Q for j = 0, ..., Q - 1; 0.5 is included only for odd Q.
        In ``'new_keywords'``, numeric values are rounded and must give an integer of at least 10.
        Default (or None) is 99.

    p_qiate_smooth : Boolean (or None), optional
        If True, aggregate forest weights using an Epanechnikov kernel centered at the requested
        quantile of fractional IATE ranks. Ranking is specific to each treatment comparison and
        outcome and incorporates ``'p_qiate_bias_adjust'`` when enabled.
        If False, use the prediction observation whose rank is closest to that quantile. If
        smoothing leaves no kernel weight above the numerical tolerance, use the same fallback.
        Default (or None) is True.

    p_qiate_smooth_bandwidth : Integer or float (or None), optional
        Multiplier for the rule-of-thumb bandwidth used when ``'p_qiate_smooth'`` is True.
        The base bandwidth uses fractional IATE ranks for the first treatment comparison and
        first outcome; the resulting bandwidth is shared across all comparisons and outcomes.
        At initialization, None or nonpositive values use 1. Updates through ``'new_keywords'``
        require a positive numeric value.
        Default (or None) is 1.

    p_qiate_bias_adjust : Boolean (or None), optional
        Adjust the IATE ranks used to construct QIATE weights using normal quantiles and IATE
        standard errors. The original IATE estimates are located in the distribution of corrected
        values to determine the adjusted ranks.
        At initialization, True also enables 'p_iate_se'. Missing variances needed for ranking are
        computed separately, including in the additional round and after disabling reported IATE
        standard errors through 'new_keywords'.
        Used when 'p_qiate' is True. Default (or None) is False.

    p_iv_aggregation_method : String or list/tuple of strings (or None), optional
        Defines the method used to obtain aggregated IV effects.
        Pass 'local' or 'global' as a string to select one method.
        A list or tuple must contain both 'local' and 'global' and no other values.
        Single-method lists and tuples, such as ['local'] or ('global',), are rejected.

        `local` : LIATEs will be computed and aggregated to 
                  obtain LGATEs, LBGATEs, LATEs, etc. This estimator 
                  is internally consistent.
                  
        `global` : LATEs will be directly computed as the ratio of
                   reduced form and first–stage predictions. This estimator
                   estimator is not necessarily internally consistent.
            
        Group effects are computed only when 'local' is selected. Requested LIATEs are
        available with either selection and always use local first-stage scaling.

        For the differences in assumptions and properties of the two approaches see
        Lechner and Mareckova (2025).
        Default (or None) is `('local', 'global',)`.

    p_ci_level : Float (or None), optional
        Confidence level for pointwise, two-sided confidence intervals in effect plots and
        exported plot data. Bounds use a normal critical value and the estimated standard errors.
        At initialization, accept 0.5 < value < 0.99999999; None or other numeric values use 0.95.
        In ``'new_keywords'``, a numeric value with 0 < value <= 0.9999999999 is required;
        invalid updates raise an error.
        Default (or None) at initialization is 0.95.

    p_cond_var : Boolean (or None), optional
        Select the analytic variance estimator used when standard errors are requested.
        True uses outcome means and variances conditional on the estimation weights. The smoother
        is selected by 'p_knn'. False bases the variance estimate on weighted outcomes directly.
        Bootstrap and regression-residual variance calculations bypass this choice.
        Default (or None) is True.

    p_knn : Boolean (or None), optional
        Select the smoothing method for conditional moments in analytic variance estimation.
        Used only in the analytic variance branch with 'p_cond_var' set to True.
        True uses moving averages over observations sorted by weight, with window size controlled
        by 'p_knn_const' and 'p_knn_min_k'. False uses Nadaraya-Watson kernel regression on weights,
        with bandwidth scaled by 'p_nw_bandw' and kernel selected by 'p_nw_kern'.
        Default (or None) is True.

    p_knn_min_k : Integer (or None), optional
        Minimum moving-average window for conditional means and variances of the outcomes.
        Used in analytic variance estimation when 'p_cond_var' and 'p_knn' are True.
        The minimum is applied after capping the window from 'p_knn_const' at half the available
        observations. If the resulting window reaches the sample size, all observations are used.
        At initialization, None or values <= 0 select 10. Updates in new_keywords must round to > 0.
        Default (or None) is 10.

    p_nw_kern : Integer or float (or None), optional
        Kernel for Nadaraya-Watson smoothing of conditional moments in analytic variance estimation.
        Used when 'p_cond_var' is True and 'p_knn' is False.
        1 : Epanechnikov kernel.
        2 : Gaussian (normal) kernel.
        The selected kernel also determines the rule-of-thumb bandwidth, scaled by 'p_nw_bandw'.
        At initialization, a numeric value of 2 selects the Gaussian kernel; other numeric values
        select 1. Updates via 'new_keywords' round numeric values and require a result of 1 or 2;
        other rounded values raise TypeError. None is not allowed in 'new_keywords'.
        Default (or None) is 1.

    p_nw_bandw : Float (or None), optional
        Positive multiplier of the rule-of-thumb bandwidth for Nadaraya-Watson smoothing.
        Used in analytic variance estimation when 'p_cond_var' is True and 'p_knn' is False.
        Values above 1 increase smoothing; values between 0 and 1 reduce smoothing.
        At initialization, None or values <= 0 select 1. Updates in new_keywords require values > 0.
        Default (or None) is 1.

    p_max_weight_share : Float (or None), optional
        Threshold for truncating extreme estimation weights before renormalisation.
        Values strictly between 0 and 1 enable truncation; values >= 1 disable it.
        For nonnegative weights normalized to sum to one, truncation caps weights at this value.
        If negative weights are possible, truncation caps absolute weights relative to their total
        absolute weight. Renormalisation can raise final weight shares above the threshold.
        At initialization, None or values <= 0 select 0.05; zero does not disable truncation.
        Updates via new_keywords are not supported.
        Default (or None) is 0.05.

    p_cluster_std : Boolean (or None), optional
        Clustered standard errors. Always True if ``'gen_panel_data'`` is True.
        Default (or None) is False.

    p_se_boot_ate : Integer, float or Boolean (or None), optional
        Number of bootstrap replications for ATE standard errors.
        At initialization, True or 0 < value < 49 selects 199 replications. Values >= 49 are
        rounded to the nearest integer; False or values <= 0 disable bootstrapping.
        None selects 199 replications when clustered standard errors are enabled; otherwise False.
        If 'p_ate_no_se_only' is True at initialization, bootstrapping is disabled.
        With new_keywords, True selects 99 replications and False disables bootstrapping.
        Numeric values must exceed 5 and are rounded; other numeric values raise ValueError.
        None is not allowed in new_keywords. Default is None at initialization.

    p_se_boot_gate : Integer, float or Boolean (or None), optional
        Number of bootstrap replications for GATE standard errors.
        Uses the same initialization and new_keywords rules as 'p_se_boot_ate'.
        Default is None.

    p_se_boot_iate : Integer, float or Boolean (or None), optional
        Number of bootstrap replications for IATE standard errors.
        Uses the same initialization and new_keywords rules as 'p_se_boot_ate'.
        Default is None.

    p_se_boot_qiate : Integer, float or Boolean (or None), optional
        Number of bootstrap replications for QIATE standard errors.
        Uses the same initialization and new_keywords rules as 'p_se_boot_ate'.
        Default is None.
 
    p_bt_yes : Boolean (or None), optional
        ATE-based balancing test using forest weights. Relevance of this test in its current
        implementation is not fully clear.
        At initialization, the test is disabled if both 'var_x_name_balance_test_ord' and
        'var_x_name_balance_test_unord' are empty, or if 'p_ate_no_se_only' is True.
        Default (or None) is True, subject to these conditions.

    p_choice_based_sampling : Boolean (or None), optional
        Enable treatment-specific reweighting of prediction observations for ATE, GATE, BGATE and
        CBGATE aggregation using 'p_choice_based_probs'. This option does not draw a subsample.
        Requires discrete treatments and treatment information in the prediction data.
        Default (or None) is False.

    p_choice_based_probs : List or tuple of floats (or None), optional
        Relative multipliers applied to prediction observations according to their observed
        treatment when 'p_choice_based_sampling' is True. Supply one finite, positive value for each
        treatment, in the model's treatment order. Values are normalized to have mean one.
        Values multiply aggregation weights directly; they are not inverted automatically.
        Default is None; explicit values are required when 'p_choice_based_sampling' is True.

    p_ate_no_se_only : Boolean (or None), optional
        Compute only ATEs without standard errors, disabling ATETs, GATEs, BGATEs, CBGATEs,
        IATEs, QIATEs and balancing tests.
        Applies at initialization and when changed through 'new_keywords'. In an update, True
        overrides other effect flags regardless of their order.
        Default (or None) is False.

    p_ba : Boolean (or None), optional
        If True, use bias adjustment. It cannot be combined with continuous treatments, treatment
        versions, clustered standard errors (including 'gen_panel_data'), sampling weights or QIATE
        estimation. These combinations raise ValueError during initialization.
        When enabled, at least one of 'p_ba_use_prop_score', 'p_ba_use_prog_score' or 'p_ba_use_x'
        must also be enabled. Disabling all three raises ValueError during initialization.
        Prediction updates must retain at least one source unless they also disable 'p_ba'.
        Default (or None) is False.

    p_ba_adj_method : String (or None), optional
        Type of adjustment method used. Possible methods are ``'zeros'``,
        ``'train_obs'``, ``'weighted_train_obs'``.
        Default (or None) is ``'train_obs'``.
    
        This defines how to evaluate the estimated regressions in the adjustment procedures:
    
        ``'zeros'`` : The values of the (centered) covariates are set to zero.
    
        ``'train_obs'`` : Evaluate at the empirical distribution of the centered regressors in
        the regular round's full outcome sample (used to fill the trees), pooling treatments.
        This reference is computed before forest chunking and reused in the additional round.
        The adjustment regression is linear, so averaging over these centered regressors has
        the same target as 'zeros', apart from numerical rounding.
    
        ``'weighted_train_obs'`` : Evaluate on the outcome-sample observations represented by the
        forest weights in the current forest chunk and estimation round. Weight their regressors
        by the forest weights across treatments. This makes the evaluation distribution local while
        removing treatment-control differences in the regressor distribution within leaves.

    p_ba_use_prop_score : Boolean (or None), optional
        If True, propensity score is used as regressor. Propensity is estimated with random forest
        classifier of scikit-learn.
        Default is True.

    p_ba_use_prog_score : Boolean (or None), optional
        If True, estimate one prognostic score per treatment arm using the outcome specified by
        'var_y_tree_name'. With local centering, use that outcome's original, uncentered values.
        The same set of scores enters the bias-adjustment regressions for all outcomes.
        See 'p_ba_estimator' for the estimation method.
        Default (or None) is True.

    p_ba_estimator : String (or None), optional
        Regression estimator for the prognostic scores, used if ``'p_ba_use_prog_score'`` is True.
        If the outcome has fewer than 10 distinct values in the sample used to fit the scores,
        a random forest classifier is used instead, regardless of this setting.
        Possible choices are scikit-learn's regression methods 'RandomForest', 'RandomForestNminl5',
        'RandomForestNminls5', 'SupportVectorMachine', 'SupportVectorMachineC2',
        'SupportVectorMachineC4', 'AdaBoost', 'AdaBoost100', 'AdaBoost200', 'GradBoost',
        'GradBoostDepth6', 'GradBoostDepth12', 'NeuralNet', 'NeuralNetLarge',
        'NeuralNetLarger', 'Mean'.
        If set to 'automatic', the estimator with the lowest cross-validated mean squared error
        (MSE) is selected separately within each treatment arm, using ``'p_ba_cv_k'`` folds.
        'Mean' is included for the cases when none of the methods have explanatory power.
        Default (or None) is 'RandomForest'.

    p_ba_cv_k : Integer or float (or None), optional
        Number of folds for ridge-penalty selection and automatic prognostic regression selection
        in bias adjustment. Automatic prognostic selection caps the count at the number of
        score-training observations in each treatment group; a single observation uses its outcome
        as the predicted mean. Classifier-based prognostic scores do not use cross-validation.
        Numeric values are rounded to the nearest integer. None or values rounding to fewer than
        two folds select the default, which depends on the size of the training sample (N):
        N < 100'000: 5;  100'000 <= N < 250'000: 4; 250'000 <= N < 500'000: 3; 500'000 <= N: 2.

    p_ba_use_x : Boolean (or None), optional
        If True, use variables specified in 'var_x_name_ba' as regressors.
        Enabling this through 'new_keywords' requires covariates prepared during training with
        this option enabled. Otherwise, bias-adjustment prediction raises ValueError.
        Default (or None) is False.

    p_ba_ridge : Boolean (or None), optional
        If True, use weighted ridge regression for bias adjustment; otherwise use weighted OLS.
        Ridge selects its penalty by cross-validation using the first outcome in the resolved
        'var_y_name' list. With local centering, its centered values are used.
        Selection is repeated for each set of forest weights being adjusted. The resulting
        adjusted weights are shared across all outcomes. The intercept is not penalized.
        Default (or None) is True.

    p_ba_pos_weights_only : Boolean (or None), optional
        If True, bias adjustment constrains weights to be nonnegative; zero weights are allowed.
        Only relevant if ``'p_ba'`` is True.
        Default (or None) is False.

    post_est_stats : Boolean (or None), optional
        Enable the descriptive IATE analyses performed by :meth:`~ModifiedCausalForest.analyse`.
        Individual ``'post_'`` options select correlations, plots, k-means, random forests and
        regression trees. Supply prediction results containing IATE data.
        Calling ``analyse()`` requires this option, ``'_int_with_output'`` and
        ``'_int_return_iate_sp'`` to be True; otherwise, the method raises ValueError.
        Default (or None) is True. Forced to False at initialization if IATE estimation is disabled.

    post_relative_to_first_group_only : Boolean (or None), optional
        Select comparisons for descriptive IATE analyses. True uses only effects relative to the
        treatment with the lowest value. False includes all estimated comparisons for discrete
        treatments. For continuous treatments, the restriction applies regardless of this setting.
        Applies to correlations, plots, k-means clustering, random forests and regression trees.
        Default (or None) is True.

    post_bin_corr_yes : Boolean (or None), optional
        Report pairwise Pearson correlations in :meth:`~ModifiedCausalForest.analyse`: IATEs with
        features and, for discrete treatments, IATEs with other IATEs and potential outcomes.
        ``'post_bin_corr_threshold'`` filters feature-IATE correlations only.
        Default (or None) is True.

    post_bin_corr_threshold : Float (or None), optional
        Display a feature-IATE correlation only if its absolute value exceeds this threshold.
        For discrete treatments, correlations between effects and with potential outcomes are
        not filtered. Requires 'post_bin_corr_yes' to be True. A value of 0.1 means 10%.
        At initialization, values in [0, 1] are accepted; None or values outside this range use 0.1.
        Numeric updates via new_keywords require 0 < value <= 1; zero is rejected.
        None is not allowed in new_keywords. Default is 0.1 at initialization.

    post_kmeans_yes : Boolean (or None), optional
        Use k-means clustering in :meth:`~ModifiedCausalForest.analyse` to group observations by
        their IATEs, with k-means++ initialization. Joint clustering uses selected effect columns in
        their original units, without standardization, so their relative scales affect the result.
        Comparisons follow 'post_relative_to_first_group_only'.
        Default (or None) is True.

    post_kmeans_single : Boolean (or None), optional
        If True (and ``'post_kmeans_yes'`` is True), clustering is also with respect to all single
        effects. If False (and ``'post_kmeans_yes'`` is True), clustering is only with respect to
        all relevant IATEs jointly.
        Default (or None) is False.

    post_kmeans_no_of_groups : Integer, float, list or tuple (or None), optional
        Candidate cluster counts for k-means. A scalar or one-element sequence requests one count;
        a longer sequence compares candidates and retains the highest average silhouette score.
        Values must be finite numbers that round to integers of at least 2. Boolean values and
        empty sequences are rejected. Each count is capped at the number of distinct effect
        profiles and one fewer than the observations used for clustering. Small clusters
        may be merged, reducing the final count. If no candidate has a valid silhouette score,
        analyse() raises ValueError, e.g. with fewer than three observations or constant effects.
        None selects an automatic grid based on prediction sample size n: c = 5 if n < 10000,
        c = 10 if n > 100000, and c = 5 + round(n / 20000) otherwise.
        The grid is [c-2, c-1, c, c+1, c+2] if c < 7, or [c-4, c-2, c, c+2, c+4] otherwise.
        Scalars, lists and tuples have the same meaning in new_keywords for predict(), predict_iv()
        and analyse(). None is not allowed in new_keywords. Default is None at initialization.

    post_kmeans_max_tries : Integer or float (or None), optional
        Maximum number of iterations per k-means initialization.
        At initialization, numeric values are truncated to an integer, with a minimum of 10.
        Numeric updates via new_keywords are rounded and must be between 1 and 10000 inclusive;
        values outside this range raise ValueError. None is not allowed in new_keywords.
        Default (or None) is 1000.

    post_kmeans_replications : Integer or float (or None), optional
        Number of k-means runs with different k-means++ initializations for each candidate count.
        At initialization, None or negative values select 10 runs. Nonnegative values are rounded
        to the nearest integer, with a minimum of 1.
        Numeric updates via new_keywords are rounded and must be between 1 and 1000 inclusive;
        values outside this range raise ValueError. None is not allowed in new_keywords.
        Default (or None) is 10.

    post_kmeans_min_size_share : Float (or None), optional
        Minimum cluster size as a percentage of the observations used for clustering.
        Clusters below this threshold are repeatedly merged with the nearest remaining cluster,
        updating centroids after each merge. This may reduce the final number of clusters.
        A candidate with fewer than two clusters is excluded from silhouette-based selection.
        At initialization, 0 < value < 33 is accepted; None or values outside this range use 1.
        Numeric updates via new_keywords require 0 < value <= 33. None is not allowed.
        Default (or None) is 1 (%).

    post_random_forest_vi : Boolean (or None), optional
        Fit auxiliary random forests to describe estimated IATEs in
        :meth:`~ModifiedCausalForest.analyse`. Report out-of-bag R2 and permutation importance,
        measured as the loss in test-set R2 when a feature is shuffled. The importance calculation
        uses a separate forest with a random 80/20 training/test split of the analysis sample.
        Unordered features are shuffled jointly with their dummy variables.
        These statistics describe associations with estimated effects.
        Default (or None) is True.

    post_plots : Boolean (or None), optional
        Save figures from :meth:`~ModifiedCausalForest.analyse`, including sorted effects,
        densities, IATE-feature relationships and regression trees when ``'post_tree'`` is True.
        Display figures only if ``'_int_show_plots'`` is also True.
        False disables these figure outputs; sorted-effect and density calculations and their CSV
        exports still run. Regression-tree fitting remains controlled by ``'post_tree'``.
        Default (or None) is True.

    post_tree : Boolean (or None), optional
        Fit regression trees in :meth:`~ModifiedCausalForest.analyse` to describe selected IATEs.
        Trees have maximum depths 2, 3, 4 and 5; they may be shallower.
        For each depth, the standard tree is fitted on a construction subsample. The honest variant
        keeps its splits and updates terminal-leaf predictions using a separate filling subsample.
        Leaves without filling observations retain their original predictions.
        Both variants report R-squared on the same separate 25% test sample.
        Default (or None) is True.

    p_knn_const : Integer or float (or None), optional
        Multiplier for the moving-average window in k-NN estimation of conditional means and
        variances, used for analytic standard errors when 'p_cond_var' and 'p_knn' are True.
        Also controls smoothing of sorted IATE plots in :meth:`~ModifiedCausalForest.analyse`.
        For plot smoothing, the rounded window is at least one observation.
        Default (or None) is 1. Negative values passed to the constructor also use 1.

    _int_cuda : Boolean (or None), optional
        Request experimental CUDA acceleration for supported computations, including matching and
        IATE estimation. Requires importable PyTorch with CUDA available; otherwise a warning is
        issued and CUDA is disabled.
        The IATE GPU path additionally requires '_int_low_memory_predict' and 'p_ba' to be False
        and the selected IATE worker count to be below 16. Otherwise, IATEs use the CPU.
        Default (or None) is False.

    _int_descriptive_stats : Boolean (or None), optional
        Print descriptive stats if ``'_int_with_output'`` is True.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_show_plots : Boolean (or None), optional
        Execute show() command if ``'_int_with_output'`` is True.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_dpi : Integer or float (or None), optional
        Resolution passed to figure-saving functions, in dots per inch.
        None or numeric values below 10 select 500; other numeric values are rounded to an integer.
        Default (or None) is 500.
        Internal variable, change default only if you know what you do.

    _int_fontsize : Integer or float (or None), optional
        Font-size index used for plot legends and titles. Indices 1 through 7 select, respectively,
        'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large' and 'xx-large'.
        Numeric inputs strictly between 0.5 and 7.5 are rounded to an index; other inputs use 2.
        Default (or None) is 2 ('x-small').
        Internal variable, change default only if you know what you do.

    _int_no_filled_plot : Integer or float (or None), optional
        In GATE, BGATE and CBGATE plots for discrete treatments, connect adjacent estimates and
        confidence limits when an ordered variable has more evaluation values than this threshold.
        In that case, smoothed estimates also produce a plot with filled confidence bands.
        The ordinary plot is still saved.
        None or values below 5 select 20; other numeric values are rounded to an integer.
        Default (or None) is 20.
        Internal variable, change default only if you know what you do.

    _int_max_cats_cont_vars : Integer or float (or None), optional
        Optional quantile discretization of ordered covariates. None or values below 1 disable it;
        other numeric values are rounded to a target number g of bins before training.
        A covariate is discretized only if it has more than g + 2 distinct training values.
        Duplicate quantile boundaries can reduce the number of bins. Each bin is represented by
        its training mean. Prediction reuses the training bins and means; values outside the
        training range use the corresponding outer bin.
        Default is None.
        Internal variable, change default only if you know what you do.

    _int_max_save_values : Integer or float (or None), optional
        Store an ordered feature's distinct values in the internal dictionary only if their count
        is strictly below this threshold; otherwise store an empty list.
        Unordered features retain all category values regardless of this setting.
        BGATE and CBGATE use ``'p_gates_no_evalu_points'`` for fallback quantile grids when no
        evaluation values are available.
        Default (or None) is 50.
        Internal variable, change default only if you know what you do.

    _int_max_obs_training : Integer (or None), optional
        Maximum number of input training rows. If exceeded, draw this many rows without replacement
        before data cleaning and other preprocessing, using a fixed random seed.
        Integer values greater than 100 set the limit. None or values at most 100 impose no limit.
        Default (or None) is no limit.
        Internal variable, change default only if you know what you do.

    _int_max_obs_prediction : Integer (or None), optional
        Maximum number of input prediction rows. If exceeded, draw this many rows without
        replacement before preprocessing, using a fixed random seed. Predictions and aggregate
        effects use the retained sample, subject to subsequent data cleaning and trimming.
        Integer values greater than 100 set the limit. None or values at most 100 use the default.
        Default is 250'000 when ``'_int_low_memory_predict'`` is False and 1'000'000 otherwise.
        Internal variable, change default only if you know what you do.

    _int_low_memory_predict : Boolean (or None), optional
        If True, the memory footprint of the prediction step
        will be drastically reduced (and computational speed significantly increased) by not keeping
        the full weight matrix. This will allow for deeper forests and more prediction data points
        that can reasonably be used when training data is very large. Therefore, the defaults for
        _int_max_obs_prediction and cf_chunks_maxsize are larger when ``'_int_low_memory_predict'``
        is True. As of now, this option is incompatible with instrumental variable estimation and
        the estimation of QIATEs. ``'p_iate_m_ate'`` must also be set to False
        if ``'_int_low_memory_predict'`` is True.
        Default is True.
        Internal variable, change default only if you know what you do.

    _int_low_memory_max_chunksize : Integer or float (or None), optional
        Target number of prediction rows per chunk when '_int_low_memory_predict' is True.
        The current sample is split into ceil(n_prediction / value) approximately equal chunks.
        None selects 1000 if n_training <= 10000, else max(1000 - sqrt(n_training - 10000), 10).
        n_training is the training sample size after applying '_int_max_obs_training'.
        Explicit finite values >= 1 are retained, including values below 10 or above 1000.
        Values in [0.1, 1) also select automatic sizing. With low-memory prediction enabled,
        nonnumeric inputs and values below 0.1 raise ValueError.
        Default is None. Internal variable, change default only if you know what you do.

    _int_max_obs_kmeans : Integer (or None), optional
        Maximum number of prediction rows used for k-means in :meth:`~ModifiedCausalForest.analyse`.
        If exceeded, sample this many rows without replacement using a fixed seed. The returned
        ``'iate_data_df'`` with cluster labels contains only the retained rows.
        Integer values greater than 100 set the limit. None or values at most 100 use 200'000.
        Default (or None) is 200'000.
        Internal variable, change default only if you know what you do.

    _int_max_obs_post_rel_graphs : Integer (or None), optional
        Maximum number of rows per IATE-feature graph in :meth:`~ModifiedCausalForest.analyse`.
        Used when ``'post_plots'`` is True. If exceeded, sample this many rows without replacement
        using a fixed seed. The graphs, their CSV data and any LOWESS fits use the retained rows.
        Integer values greater than 100 set the limit. None or values at most 100 use 50'000.
        Default (or None) is 50'000.
        Internal variable, change default only if you know what you do.

    _int_mp_use_old_ray : Boolean (or None), optional
        Use old implementation of ray.
        Default is False.
        Internal variable, change default only if you know what you do.
        
    _int_mp_backend : String (or None), optional
        Backend for parallel execution when _int_mp_use_old_ray is False.
        Options are 'ray', 'joblib' and 'sequential'.
        Default is None, resolved during training initialization.
        None selects 'joblib' if Ray cannot be imported, or on Windows when
        the adjusted training sample size (cf_cfg.n_train) is at most 250,000.
        Otherwise, None selects 'ray'.
        This sample size accounts for data reserved for feature selection,
        common support and local centering.
        On Windows, explicitly selecting 'joblib' raises ValueError during
        initialization if the initial worker count exceeds 63.
        Internal variable, change default only if you know what you do.

    _int_mp_batches : Integer, float or string (or None), optional
        Requested maximum number of task batches where batching is used.
        Numeric values must be at least 1; floats are truncated to integers.
        'automatic' or None selects batch sizes from the task count, worker
        count, backend and computation. Other strings are not accepted.
        Minimum batch sizes depend on the worker count, so the actual number
        of batches can be smaller than requested.
        Only relevant if _int_mp_use_old_ray is False.
        Weight computations that stream results through executor.map_iter
        bypass this setting.
        Default is 'automatic'.
        Internal variable, change default only if you know what you do.

    _int_mp_memmap_min_bytes : Integer or float (or None), optional
        Threshold in bytes for memory-mapping eligible shared NumPy arrays
        with the 'joblib' backend when _int_mp_use_old_ray is False.
        An array is mapped when its nbytes is at least this threshold.
        Arrays with dtype object are excluded.
        Nonnegative Python integers and floats are accepted; floats are
        truncated to integers. None, negative values and other input types
        use 4 * 1024 * 1024 bytes.
        A value of 0 removes the size threshold for eligible arrays.
        Default is 4 * 1024 * 1024 bytes (4 MiB).
        Internal variable, change default only if you know what you do.

    _int_mp_memmap_dir : str or Path object (or None), optional
        Base directory for temporary files used by the 'joblib' backend.
        Only relevant if _int_mp_use_old_ray is False.
        Strings are converted to Path objects; Path objects are kept.
        None and other input types use Path.cwd() / 'joblibtemp' at
        initialization.
        Each executor creates a unique subdirectory beneath this directory
        and attempts to remove that subdirectory at shutdown.
        The base directory is removed only if the executor created it and
        it is empty. A directory that already existed is retained.
        Cleanup failures issue warnings and may leave temporary files.
        Default is Path.cwd() / 'joblibtemp', evaluated at module import.
        Internal variable, change default only if you know what you do.

    _int_mp_ray_del : String, tuple or list of strings (or None), optional
        Controls selected local references in legacy Ray cleanup calls.
        A string is treated as a one-item tuple; lists are converted to tuples.
        At most two entries are accepted: 'refs', 'rest' or 'none'.
        'refs' replaces the supplied shared-object references with None.
        'rest' replaces the supplied result list and completed-task reference
        list with None; it does not delete all objects created by a task.
        Use ('refs', 'rest') to select both actions.
        ('none',) or an empty tuple selects neither action.
        'none' does not override 'refs' or 'rest' when combined with them.
        This releases only those local references; other references can keep
        the underlying objects alive.
        Default (or None) is ('refs',).
        Internal variable, change default only if you know what you do.

    _int_mp_ray_objstore_multiplier : Integer or float (or None), optional
        Multiplier for explicit Ray object-store memory requests computed
        during training and prediction initialization.
        Positive values scale these requests before final caps based on
        available memory are applied.
        None or a negative value uses 1; 0 skips scaling.
        Requests set to None use Ray's default sizing and are not scaled.
        The multiplier applies independently of _int_mp_ray_shutdown.
        Memory requests affect Ray initialization; an already running Ray
        instance is not resized.
        Initialization may fall back to Ray's default object-store size if
        explicit memory requests fail.
        Default (or None) is 1.
        Internal variable, change default only if you know what you do.

    _int_mp_ray_shutdown : Boolean (or None), optional
        If True, shut down initialized Ray at legacy cleanup points when
        _int_mp_use_old_ray is True and the configured worker count exceeds 1.
        During training initialization, None becomes False if the effective
        sample size per chunk (cf_cfg.n_train_eff) is below 150,000, and True
        otherwise.
        If still None during prediction initialization, use the same cutoff
        with sqrt(n_pred * cf_cfg.n_train_eff), where n_pred is the number of
        prediction observations.
        An already resolved Boolean is retained by later updates.
        This flag does not control shutdowns by the new Ray executor or by
        restart and error-recovery logic.
        Default is None.
        Internal variable, change default only if you know what you do.

    _int_iate_chunk_size : Integer or float (or None), optional
        Target number of prediction observations per task in IATE 2/2
        (effect computation), for legacy Ray and the newer backends.
        None, nonpositive values and nonnumeric inputs select automatic
        sizing: ceil(n_prediction / number_of_workers).
        This default also applies with low-memory prediction and is
        independent of _int_low_memory_max_chunksize.
        A value of 1 uses one observation per task. Values above 1 divide
        observations into ceil(n_prediction / value) balanced chunks.
        Positive values below 1 are not supported by the parallel branch.
        Ignored by the direct single-worker branch.
        Default is None.

    _int_mp_weights_tree_batch : Integer, float or Boolean (or None), optional
        Number of forest batches in the tree-batching branch of OOB
        variable-importance evaluation.
        Used only in get_oob_mcf when no_mp=False and more than one worker
        is selected. Tasks parallelized over variables or groups bypass it.
        False and True become 1. Other numeric values above 0.5 are rounded
        with round(); all remaining inputs, including None, select automatic
        sizing.
        A normalized value of 1 disables batching. Larger values specify
        the batch count, which must not exceed the number of trees.
        Automatic sizing uses forest size, available memory and worker count.
        Splitting is skipped if the automatic count is at least the number
        of trees.
        Default is None.
        Internal variable, change default only if you know what you do.

    _int_mp_weights_type : Integer or float (or None), optional
        Parallelization strategy for standard forest-weight computation.
        1: Distribute groups of prediction observations across workers.
        2: Process prediction observations sequentially and distribute the
        trees across workers for each observation.
        Supported by both legacy Ray and the newer parallel backends.
        Ignored when weight computation uses only one worker, including
        within each chunk of low-memory prediction.
        A value equal to 2 (including 2.0) selects strategy 2; other values,
        including None, select strategy 1.
        Default is 1.
        Internal variable, change default only if you know what you do.

    _int_obs_bigdata : Integer or float (or None), optional
        Observation threshold for sample-size-dependent memory reductions.
        Numeric values above 10 are retained without rounding; None and
        other values use 1_000_000.
        During training, if the input sample exceeds this threshold and the
        current worker count exceeds 1, gen_mp_parallel is replaced by
        int(0.75 * gen_mp_parallel), for automatic and explicit settings.
        Local-centering estimator selection halves its worker limit again
        if its input sample exceeds the threshold (rounded down, minimum 1).
        The internal big-data flag is enabled when the adjusted forest
        training sample size exceeds this threshold. That size accounts for
        feature selection and holdout shares.
        Selected arrays use float32 instead of float64 during training and
        prediction. The cutoff depends on the operation and its input size;
        some switch at the threshold, others only above it.
        Default (or None) is 1_000_000.

    _int_output_no_new_dir : Boolean (or None), optional
        During initialization, True reuses an existing output directory even if it is nonempty.
        With False or None, reuse an empty directory; for a nonempty one, search for an unused or
        empty directory with a numeric suffix. Missing directories are created in either case.
        Existing text and summary files with the configured names are deleted during initialization.
        A 'new_keywords' dictionary (including {}) passed to predict(), predict_iv() or analyse()
        triggers the empty-directory or numeric-suffix search regardless of this flag. The selected
        paths persist on the instance.
        Only relevant when output is enabled. Default (or None) is False.
        Internal variable, change default only if you know what you do.

    _int_return_iate_sp : Boolean (or None), optional
        Return processed prediction data with individual-effect results
        in 'iate_data_df', including when _int_with_output is False.
        For predict_iv(), this also controls the corresponding first-stage
        and reduced-form DataFrames.
        Requires IATE estimation to be enabled. This flag does not enable
        IATE estimation or suppress the separately returned effect arrays.
        Potential outcomes and requested standard errors and effects
        relative to the average effect are included in these DataFrames.
        During initialization, output enabled via _int_with_output forces
        this flag to True, even if False or None is supplied.
        With output disabled, only True enables it; p_ate_no_se_only=True forces it to False.
        Default argument is False.
        Internal variable, change default only if you know what you do.

    _int_replication : Boolean (or None), optional
        When True, use n_jobs=1 for the scikit-learn random forests used in
        feature selection, common-support estimation, local centering and
        propensity-score estimation for sensitivity analysis.
        This does not disable multiprocessing globally: matching and
        post-estimation random forests still use gen_mp_parallel.
        MCF forest construction and weight computation also retain their usual worker settings.
        Only the Boolean value True enables this option.
        Default (or None) is False.

    _int_memory_print : Boolean (or None), optional
        Print system RAM statistics (total, used, available and free) at
        selected stages of training and prediction.
        Requires output and verbose output to be enabled through
        _int_with_output and _int_verbose.
        Output follows gen_output_type. File output is written to the
        extensive text file.
        Only the Boolean value True enables this option.
        Default (or None) is False.

    _int_share_forest_sample : Float (or None), optional
        Share of the training sample allocated to filling trees with
        outcomes in the regular estimation round. The remaining share is
        used to build the tree structures.
        The split is made after data cleaning and feature selection, before
        common-support trimming and local centering.
        With efficient estimation enabled, the additional round exchanges the roles of the two
        samples. This value also affects automatically selected minimum leaf sizes.
        Values must be strictly between 0.01 and 0.99; None and numeric
        values outside this interval are replaced by 0.5.
        Default (or None) is 0.5.
        Internal variable, change default only if you know what you do.

    _int_verbose :  Boolean (or None), optional
        Additional output about running of mcf if ``'_int_with_output'`` is True.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_weight_as_sparse : Boolean (or None), optional
        Store forest weights in sparse matrices.
        Must be True for :meth:`~ModifiedCausalForest.train_iv`; False raises ValueError there.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_weight_as_sparse_splits : Integer or None, optional
        Number of prediction-data chunks used to construct sparse weights.
        Relevant only when _int_weight_as_sparse is True. Use a positive
        Python integer; 1 disables this extra splitting.
        With multiple chunks, sparse weights are temporarily saved to disk,
        then reloaded and combined into full matrices for the current call.
        Values that are not Python int instances are treated as None.
        During initialization, None becomes 1 if _int_low_memory_predict
        is enabled. Otherwise, None is resolved when weights are first
        computed: use 1 if n_pred < 5000, and
        max(1, round(n_fill * n_pred / (n_folds * 25000**2))) otherwise.
        Here, n_fill is the current forest's filling-sample size, n_pred is
        the number of prediction rows in the current call, and n_folds is
        the number of training-data chunks.
        Default is None.
        Internal variable, change default only if you know what you do.

    _int_with_output : Boolean (or None), optional
        Enable standard text output and output-dependent diagnostics,
        figures and report information. Text output follows gen_output_type.
        False disables verbose output and leaves output paths unset
        during initialization.
        It also suppresses returned common-support probability tables
        and the diagnostics requested by cf_vi_oob_yes.
        During initialization, enabled output forces _int_return_iate_sp
        to True. With output disabled, that option controls IATE DataFrame return.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_del_forest : Boolean (or None), optional
        Delete forests from instance. If True, less memory is needed, but
        the trained instance of the class cannot be reused when calling
        :meth:`~ModifiedCausalForest.predict` with the same instance again,
        i.e. the forest has to be retrained when applied again.
        Default (or None) is False.

    _int_keep_w0 : Boolean (or None), optional.
        Keep all zeros weights when computing standard errors (slows down computation and may
        lead to undesirable behaviour).
        Default is False.

    Attributes
    ----------   
    __version__ : String
        Version of mcf module used to create the instance.

    <NOT-ON-API>

    cf_cfg : Instance of CfCfg dataclass
        Parameters used in training the forest (directly).

    cs_cfg : Instance of CsCfg dataclass
        Parameters used in common support adjustments.

    ct_cfg :  Instance of CtGrid dataclass
        Parameters used in dealing with continuous treatments.

    data_train_dict : Dictionary

    dc_cfg : Instance of DCCfg dataclass
        Parameters used in data cleaning.

    fs_cfg : Instance of FsCfg dataclass
        Parameters used in feature selection.

    forest : List
        List of lists containing the estimated causal forest.

    gen_cfg : Instance of GenCfg dataclass
        General parameters used in various parts of the programme.
    
    gen_tv_cfg : Instance of GenTvCfg dataclass
        General parameters used for version estimation in various parts of the programme.

    int_cfg : Dataclass
        Internal parameters used in various parts of the class.

    iv_mcf : Dictionary
        Internal instances of instrumental mcf containing for first stage and reduced form.

    low_mem_cfg : Instance of LowMemCfg dataclass
        Parameters used in the low memory version of the *predict* methods.

    lc_cfg : Instance of LcCfg dataclass
        Parameters used in local centering.

    p_cfg : Instance of PCfg dataclass
        Parameters used in prediction method.

    post_cfg : Instance of PostCfg dataclass
        Parameters used in analyse method.

    report : Dictionary
        Provides information for McfOptPolReports to construct reports.

    sens_cfg : Instance of SensCfg dataclass
        Parameters used in sensitivity method.

    time_strings : Dictionary
        Maps computation-step labels to stored timing summaries.

    var_cfg : Instance of VarCfg Dataclass
        Variable names.

    var_x_type : Dictionary
        Types of covariates (internal).

    var_x_values : Dictionary
        Values of covariates (internal).

    </NOT-ON-API>

    """

    def __init__(
            self, *,    # Keywords only
            var_cluster_name=None,
            var_d_name=None,
            var_id_name=None, var_iv_name=None,
            var_w_name=None,
            var_x_name_always_in_ord=None, var_x_name_always_in_unord=None,
            var_x_name_balance_test_ord=None, var_x_name_balance_test_unord=None,
            var_x_name_remain_ord=None, var_x_name_remain_unord=None,
            var_x_name_ord=None, var_x_name_unord=None, var_x_name_balance_bgate=None,
            var_x_name_ba=None, var_x_name_tv=None, var_y_name=None, var_y_tree_name=None,
            var_z_name_cont=None, var_z_name_ord=None, var_z_name_unord=None,
            cf_alpha_reg_grid=1, cf_alpha_reg_max=0.15, cf_alpha_reg_min=0.05,
            cf_alpha_reg_type=None,
            cf_boot=1_000, cf_chunks_maxsize=None, cf_compare_only_to_zero=False,
            cf_n_min_grid=1, cf_n_min_max=None, cf_n_min_min=None, cf_n_min_treat=None,
            cf_nn_main_diag_only=False, cf_m_grid=1, cf_m_random_poisson=True,
            cf_m_share_max=0.6, cf_m_share_min=0.1, cf_match_nn_prog_score=True,
            cf_mce_type='cross_error', cf_mce_vart=1,
            cf_random_thresholds=None, cf_p_diff_penalty=None, cf_penalty_type='mse_d',
            cf_subsample_factor_eval=None, cf_subsample_factor_forest=1,
            cf_subsample_power_eval: float | int | None = None, cf_tune_all=False,
            cf_vi_oob_yes=False,
            cs_adjust_limits=None, cs_detect_const_vars_stop=True, cs_max_del_train=0.5,
            cs_min_p=0.01, cs_quantil=1, cs_type=1,
            ct_grid_dr=100, ct_grid_nn=10, ct_grid_w=10,
            dc_check_perfectcorr=True, dc_clean_data=True, dc_min_dummy_obs=10,
            dc_screen_covariates=True,
            fs_rel_vi_threshold_y=0, fs_rel_vi_threshold_d=0, fs_other_sample=True,
            fs_other_sample_share=0.33, fs_yes=False, fs_rel_vi_keep_if='y_or_d_relevant',
            fs_mse_for_classifier=True,
            gen_ate_eff=False, gen_d_type='discrete', gen_gate_eff=False, gen_iate_eff=False,
            gen_mp_parallel=None, gen_qiate_eff=False, gen_outfiletext=None, gen_outpath=None,
            gen_output_type=2, gen_panel_data=False, gen_panel_in_rf=True, gen_weighted=False,
            gen_tv_estimator='ridge', gen_tv_cv_k=None, gen_tv_addscores=False,
            gen_tv_score_estimator='RandomForest', gen_tv_score_cv_k=None,
            gen_tv_penalize_version=False, gen_tv_min_subtreat=10,
            gen_tv_specification='interacted',
            gen_tv_score_type=('prop_version', 'prop_main',),
            gen_tv_score_version_local=False,
            lc_cs_cv=True, lc_cs_cv_k=None, lc_cs_share=0.25, lc_estimator='RandomForest',
            lc_yes=True, lc_uncenter_po=True,
            p_ate_no_se_only=False, p_atet=False, p_bgate=False, p_ba=False,
            p_ba_adj_method='train_obs', p_ba_ridge=True, p_ba_pos_weights_only=False,
            p_ba_use_prop_score=True, p_ba_use_prog_score=True, p_ba_use_x=False,
            p_ba_estimator='RandomForest', p_ba_cv_k=None, p_bgate_sample_share=None,
            p_bt_yes=True, p_cbgate=False, p_choice_based_sampling=False, p_choice_based_probs=None,
            p_ci_level=0.95, p_cluster_std=False, p_cond_var=True, p_gates_minus_previous=False,
            p_gates_smooth=True, p_gates_smooth_bandwidth=1, p_gates_smooth_no_evalu_points=50,
            p_gates_no_evalu_points=50, p_gatet=False, p_iate=True, p_iate_se=False,
            p_iate_m_ate=False, p_iv_aggregation_method=('local', 'global',), p_knn=True,
            p_knn_const=1, p_knn_min_k=10, p_nw_bandw=1, p_nw_kern=1, p_max_cats_z_vars=None,
            p_max_weight_share=0.05, p_qiate=False, p_qiate_se=False, p_qiate_m_mqiate=False,
            p_qiate_m_opp=False, p_qiate_no_of_quantiles=99, p_qiate_smooth=True,
            p_qiate_smooth_bandwidth=1, p_qiate_bias_adjust=False, p_se_boot_ate=None,
            p_se_boot_gate=None, p_se_boot_iate=None, p_se_boot_qiate=None,
            post_bin_corr_threshold=0.1, post_bin_corr_yes=True, post_est_stats=True,
            post_kmeans_no_of_groups=None, post_kmeans_max_tries=1_000,
            post_kmeans_min_size_share=None, post_kmeans_replications=10, post_kmeans_single=False,
            post_kmeans_yes=True, post_random_forest_vi=True,
            post_relative_to_first_group_only=True, post_plots=True, post_tree=True,
            _int_cuda=False, _int_del_forest=False, _int_descriptive_stats=True, _int_dpi=500,
            _int_fontsize=2, _int_iate_chunk_size=None, _int_keep_w0=False,
            _int_low_memory_predict=True, _int_low_memory_max_chunksize=None,
            _int_max_cats_cont_vars=None, _int_max_save_values=50,  _int_no_filled_plot=20,
            _int_max_obs_training=float('inf'), _int_max_obs_prediction=None,
            _int_max_obs_kmeans=200_000, _int_max_obs_post_rel_graphs=50_000,
            _int_memory_print=False, _int_mp_backend=None, _int_mp_batches='automatic',
            _int_mp_memmap_min_bytes=4*1024*1024, _int_mp_memmap_dir=Path.cwd() / 'joblibtemp',
            _int_mp_ray_del=('refs',), _int_mp_ray_objstore_multiplier=1, _int_mp_ray_shutdown=None,
            _int_mp_use_old_ray=False, _int_mp_weights_tree_batch=None,
            _int_mp_weights_type=1,
            _int_obs_bigdata=1_000_000, _int_output_no_new_dir=False,
            _int_replication=False, _int_return_iate_sp=False,
            _int_share_forest_sample=0.5, _int_show_plots=True, _int_verbose=True,
            _int_weight_as_sparse=True, _int_weight_as_sparse_splits=None, _int_with_output=True,
            ):

        self.__version__ = '0.11.0'

        self.int_cfg = IntCfg.from_args(cuda=_int_cuda, cython=False,  # Cython turned off for now
                                        del_forest=_int_del_forest,
                                        descriptive_stats=_int_descriptive_stats, dpi=_int_dpi,
                                        fontsize=_int_fontsize,
                                        iate_chunk_size=_int_iate_chunk_size,
                                        keep_w0=_int_keep_w0,
                                        low_memory_predict= _int_low_memory_predict,
                                        max_save_values=_int_max_save_values,
                                        max_obs_training=_int_max_obs_training,
                                        max_obs_prediction=_int_max_obs_prediction,
                                        max_obs_kmeans=_int_max_obs_kmeans,
                                        max_obs_post_rel_graphs=_int_max_obs_post_rel_graphs,
                                        max_cats_cont_vars=_int_max_cats_cont_vars,
                                        memory_print=_int_memory_print,
                                        mp_use_old_ray=_int_mp_use_old_ray,
                                        mp_backend=_int_mp_backend,
                                        mp_batches=_int_mp_batches,
                                        mp_memmap_min_bytes=_int_mp_memmap_min_bytes,
                                        mp_memmap_dir=_int_mp_memmap_dir,
                                        mp_ray_del=_int_mp_ray_del,
                                        mp_ray_objstore_multiplier=_int_mp_ray_objstore_multiplier,
                                        mp_ray_shutdown=_int_mp_ray_shutdown,
                                        mp_weights_tree_batch=_int_mp_weights_tree_batch,
                                        mp_weights_type=_int_mp_weights_type,
                                        no_filled_plot=_int_no_filled_plot,
                                        obs_bigdata=_int_obs_bigdata,
                                        output_no_new_dir=_int_output_no_new_dir,
                                        replication=_int_replication,
                                        share_forest_sample=_int_share_forest_sample,
                                        show_plots=_int_show_plots,
                                        weight_as_sparse=_int_weight_as_sparse,
                                        weight_as_sparse_splits=_int_weight_as_sparse_splits,
                                        )
        gen_cfg = GenCfg.from_args(self.int_cfg,
                                   ate_eff=gen_ate_eff,
                                   d_type=gen_d_type,
                                   gate_eff=gen_gate_eff,
                                   iate_eff=gen_iate_eff,
                                   mp_parallel=gen_mp_parallel,
                                   return_iate_sp=_int_return_iate_sp,
                                   p_ate_no_se_only=p_ate_no_se_only,
                                   outfiletext=gen_outfiletext, outpath=gen_outpath,
                                   output_type=gen_output_type,
                                   panel_data=gen_panel_data, panel_in_rf=gen_panel_in_rf,
                                   qiate_eff=gen_qiate_eff,
                                   verbose=_int_verbose,
                                   weighted=gen_weighted, with_output=_int_with_output,
                                   )
        self.dc_cfg = DCCfg.from_args(check_perfectcorr=dc_check_perfectcorr,
                                      clean_data=dc_clean_data,
                                      min_dummy_obs=dc_min_dummy_obs,
                                      screen_covariates=dc_screen_covariates,
                                      )
        self.ct_cfg = CtGrid.from_args(grid_dr=ct_grid_dr, grid_nn=ct_grid_nn, grid_w=ct_grid_w)
        self.fs_cfg = FsCfg.from_args(rel_vi_threshold_y=fs_rel_vi_threshold_y,
                                      rel_vi_threshold_d=fs_rel_vi_threshold_d,
                                      rel_vi_keep_if=fs_rel_vi_keep_if,
                                      other_sample=fs_other_sample,
                                      other_sample_share=fs_other_sample_share, yes=fs_yes,
                                      mse_for_classifier=fs_mse_for_classifier,
                                      )
        self.cs_cfg = CsCfg.from_args(gen_cfg,
                                      adjust_limits=cs_adjust_limits,
                                      detect_const_vars_stop=cs_detect_const_vars_stop,
                                      max_del_train=cs_max_del_train, min_p=cs_min_p,
                                      quantil=cs_quantil,
                                      type_=cs_type,
                                      )
        self.lc_cfg = LcCfg.from_args(cs_cv=lc_cs_cv, cs_cv_k=lc_cs_cv_k, cs_share=lc_cs_share,
                                      estimator=lc_estimator,
                                      undo_iate=lc_uncenter_po,
                                      yes=lc_yes,
                                      )
        self.cf_cfg = CfCfg.from_args(gen_cfg,
                                      alpha_reg_grid=cf_alpha_reg_grid,
                                      alpha_reg_max=cf_alpha_reg_max,
                                      alpha_reg_min=cf_alpha_reg_min,
                                      alpha_reg_type=cf_alpha_reg_type,
                                      boot=cf_boot,
                                      chunks_maxsize=cf_chunks_maxsize,
                                      compare_only_to_zero=cf_compare_only_to_zero,
                                      m_grid=cf_m_grid, m_share_max=cf_m_share_max,
                                      m_share_min=cf_m_share_min,
                                      m_random_poisson=cf_m_random_poisson,
                                      match_nn_prog_score=cf_match_nn_prog_score,
                                      mce_type=cf_mce_type,
                                      mce_vart=cf_mce_vart,
                                      nn_main_diag_only=cf_nn_main_diag_only,
                                      n_min_grid=cf_n_min_grid, n_min_max=cf_n_min_max,
                                      n_min_min=cf_n_min_min, n_min_treat=cf_n_min_treat,
                                      p_diff_penalty=cf_p_diff_penalty,
                                      penalty_type=cf_penalty_type,
                                      random_thresholds=cf_random_thresholds,
                                      subsample_factor_eval=cf_subsample_factor_eval,
                                      subsample_power_eval=cf_subsample_power_eval,
                                      subsample_factor_forest=cf_subsample_factor_forest,
                                      tune_all=cf_tune_all,
                                      vi_oob_yes=cf_vi_oob_yes,
                                      )
        p_cfg = PCfg.from_args(gen_cfg,
                               ate_no_se_only=p_ate_no_se_only, atet=p_atet,
                               bgate=p_bgate, bgate_sample_share=p_bgate_sample_share,
                               bt_yes=p_bt_yes,
                               cbgate=p_cbgate, choice_based_sampling=p_choice_based_sampling,
                               choice_based_probs=p_choice_based_probs, ci_level=p_ci_level,
                               cluster_std=p_cluster_std, cond_var=p_cond_var,
                               gates_minus_previous=p_gates_minus_previous,
                               gates_smooth=p_gates_smooth,
                               gates_smooth_bandwidth=p_gates_smooth_bandwidth,
                               gates_smooth_no_evalu_points=p_gates_smooth_no_evalu_points,
                               gatet=p_gatet, gate_no_evalu_points=p_gates_no_evalu_points,
                               iate=p_iate, iate_se=p_iate_se, iate_m_ate=p_iate_m_ate,
                               iv_aggregation_method=p_iv_aggregation_method,
                               knn=p_knn, knn_const=p_knn_const, knn_min_k=p_knn_min_k,
                               max_cats_z_vars=p_max_cats_z_vars,
                               max_weight_share=p_max_weight_share,
                               nw_bandw=p_nw_bandw, nw_kern=p_nw_kern,
                               qiate=p_qiate, qiate_se=p_qiate_se, qiate_m_mqiate=p_qiate_m_mqiate,
                               qiate_m_opp=p_qiate_m_opp,
                               qiate_no_of_quantiles=p_qiate_no_of_quantiles,
                               qiate_smooth_bandwidth=p_qiate_smooth_bandwidth,
                               qiate_bias_adjust=p_qiate_bias_adjust, qiate_smooth=p_qiate_smooth,
                               se_boot_ate=p_se_boot_ate, se_boot_gate=p_se_boot_gate,
                               se_boot_iate=p_se_boot_iate, se_boot_qiate=p_se_boot_qiate,
                               )
        self.p_ba_cfg = PBiasAdjustmentCfg.from_args(adj_method=p_ba_adj_method,
                                                     cv_k=p_ba_cv_k, clustering=p_cfg.cluster_std,
                                                     continuous=gen_cfg.d_type == 'continuous',
                                                     estimator=p_ba_estimator,
                                                     ridge=p_ba_ridge,
                                                     pos_weights_only=p_ba_pos_weights_only,
                                                     qiate=p_cfg.qiate,
                                                     use_prop_score=p_ba_use_prop_score,
                                                     use_prog_score=p_ba_use_prog_score,
                                                     use_x=p_ba_use_x, yes=p_ba,
                                                     weighted=gen_cfg.weighted,
                                                     )
        self.post_cfg = PostCfg.from_args(p_cfg,
                                          bin_corr_threshold=post_bin_corr_threshold,
                                          bin_corr_yes=post_bin_corr_yes,
                                          est_stats=post_est_stats,
                                          kmeans_no_of_groups=post_kmeans_no_of_groups,
                                          kmeans_max_tries=post_kmeans_max_tries,
                                          kmeans_replications=post_kmeans_replications,
                                          kmeans_yes=post_kmeans_yes,
                                          kmeans_single=post_kmeans_single,
                                          kmeans_min_size_share=post_kmeans_min_size_share,
                                          random_forest_vi=post_random_forest_vi,
                                          relative_to_first_group_only
                                              =post_relative_to_first_group_only,
                                          plots=post_plots,
                                          tree=post_tree,
                                          )
        var_cfg = VarCfg.from_args(gen_cfg=gen_cfg, p_cfg=p_cfg, fs_yes=self.fs_cfg.yes,
                                   screen_covariates=self.dc_cfg.screen_covariates,
                                   descriptive_stats=self.int_cfg.descriptive_stats,
                                   p_ba_yes=self.p_ba_cfg.yes, p_ba_use_x=self.p_ba_cfg.use_x,
                                   cluster_name=var_cluster_name,
                                   d_name=var_d_name,
                                   id_name=var_id_name, iv_name=var_iv_name,
                                   w_name=var_w_name,
                                   x_name_balance_test_ord=var_x_name_balance_test_ord,
                                   x_name_balance_bgate=var_x_name_balance_bgate,
                                   x_name_ba=var_x_name_ba,
                                   x_name_tv=var_x_name_tv,
                                   x_name_balance_test_unord=var_x_name_balance_test_unord,
                                   x_name_always_in_ord=var_x_name_always_in_ord,
                                   x_name_always_in_unord=var_x_name_always_in_unord,
                                   x_name_remain_ord=var_x_name_remain_ord,
                                   x_name_remain_unord=var_x_name_remain_unord,
                                   x_name_ord=var_x_name_ord, x_name_unord=var_x_name_unord,
                                   y_name=var_y_name, y_tree_name=var_y_tree_name,
                                   z_name_cont=var_z_name_cont, z_name_ord=var_z_name_ord,
                                   z_name_unord=var_z_name_unord,
                                   )
        self.gen_tv_cfg = GenTvCfg.from_args(clustering=p_cfg.cluster_std, cv_k=gen_tv_cv_k,
                                             continuous=gen_cfg.d_type == 'continuous',
                                             d_name=var_cfg.d_name,
                                             estimator=gen_tv_estimator,
                                             addscores=gen_tv_addscores,
                                             score_estimator=gen_tv_score_estimator,
                                             score_cv_k=gen_tv_score_cv_k,
                                             score_type=gen_tv_score_type,
                                             score_version_local=gen_tv_score_version_local,
                                             qiate=p_cfg.qiate,
                                             p_ba=self.p_ba_cfg.yes,
                                             penalize_version=gen_tv_penalize_version,
                                             specification=gen_tv_specification,
                                             tv_min_subtreat=gen_tv_min_subtreat,
                                             weighted=gen_cfg.weighted,
                                             boot=self.cf_cfg.boot,
                                             )
        # Hide this part of code in little helper
        self.gen_cfg, self.p_cfg, self.var_cfg = var_helper(var_cfg)

        self.low_mem_cfg = LowMemCfg.from_args(yes=self.int_cfg.low_memory_predict,
                                               max_chunksize=_int_low_memory_max_chunksize,
                                               )
        self.sens_cfg = self.data_train_dict = self.var_x_type = self.var_x_values = None
        self.forest, self.time_strings = None, {}
        self.report = {'predict_list': [], 'analyse_list': []}  # Needed for multiple predicts

        self.iv_mcf = {'firststage': None, 'reducedform': None}
        self.predict_done = False
        self.predict_iv_done = False
        self.predict_different_allocations_done = False
        self.instance_used_for_training = False
        self.instance_used_for_prediction = False

        inconsistencies(self)        # Checks for illegal parameter combinations.

    def train(self, data_df, exit_after_commonsupport=False):
        """
        Build the modified causal forest on the training data.

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the causal forest. It must contain information
            about outcomes, treatment, and features.
        
        exit_after_commonsupport : Boolean, optional
            If True, return the training-result dictionary after common-support processing.
            Data preparation and sample splitting still run, but local centering and
            causal forest training are skipped.
            To apply the fitted common-support rules to new data, call
            :meth:`~ModifiedCausalForest.predict` with 'exit_after_commonsupport=True'.
            If 'cs_type' is 0, common-support processing is skipped and the method
            still returns before forest training.
            The default is False.

        Returns
        -------
        results : Dictionary.
            Training samples and common-support information. The same keys are
            returned for both values of 'exit_after_commonsupport'.

            'tree_df' : DataFrame
                Processed sample for building the trees.
            'fill_y_df' : DataFrame
                Processed sample for filling the trees with outcome values.
            'common_support_probabilities_tree' : DataFrame or None
                Common-support table for the sample used to build the trees, before trimming.
                Contains treatment probabilities, the observation identifier, and
                'on_support' (1 if retained, 0 if excluded).
                None if 'cs_type' is 0 or '_int_with_output' is False.
            'common_support_probabilities_fill_y' : DataFrame or None
                Same information for the sample used to fill the trees, before trimming.
                None if 'cs_type' is 0 or '_int_with_output' is False.
            'path_output' : Path or None
                Directory in which output is saved.
                None if _int_with_output is False.
        """
        self.instance_used_for_training = True
        results = train_main(self, data_df, exit_after_commonsupport)

        if self.int_cfg.mp_use_old_ray:
            check_ray_shutdown(self.int_cfg.mp_ray_shutdown, self.gen_cfg.mp_parallel)

        return results

    def train_iv(self, data_df):
        """
        Train the IV modified causal forest on the training data.

        Parameters
        ----------
        data_df : DataFrame
            Training data containing the outcomes, treatment, instrument and features.
            The instrument must be specified by 'var_iv_name'.
            IV estimation supports a binary treatment and a binary instrument.
            Instrument values must be 0 or 1 before conversion to integers.
            The input must meet the validation requirements under 'dc_clean_data', even when
            that option is False.

        Returns
        -------
        results : Dictionary.
            Contains the results. This dictionary has the following structure:
            'tree_df' : DataFrame
                Processed sample used for both construction and filling of the first-stage forest,
                and for construction of the regular reduced-form forest.
            'fill_y_df' : DataFrame
                Processed sample used to fill the regular reduced-form forest with outcomes.
                An additional reduced-form round exchanges its role with 'tree_df'.
            'common_support_probabilities_tree': DataFrame (or None)
                Estimated probabilities of each instrument value for the sample used
                to build the trees, before common-support trimming, plus the observation
                identifier and 'on_support' (1 if retained, 0 if excluded).
                None if 'cs_type' is 0 or '_int_with_output' is False.
            'common_support_probabilities_fill_y': DataFrame (or None)
                Estimated probabilities of each instrument value for the sample used
                to fill the trees, before common-support trimming, plus the observation
                identifier and 'on_support' (1 if retained, 0 if excluded).
                None if 'cs_type' is 0 or '_int_with_output' is False.
            'path_output' : Path or None
                Directory in which output is saved.
                None if _int_with_output is False.

        """
        inconsistencies_train(self, iv=True)
        self.instance_used_for_training = True

        data_df, rnd_reduce, txt_red = check_reduce_dataframe(
            data_df, title='Training', max_obs=self.int_cfg.max_obs_training,
            seed=124535, ignore_index=True,
            )
        if rnd_reduce and self.gen_cfg.with_output:
            print_mcf(self.gen_cfg, txt_red, summary=True)

        results = train_iv_main(self, data_df)

        if self.int_cfg.mp_use_old_ray:
            check_ray_shutdown(self.int_cfg.mp_ray_shutdown, self.gen_cfg.mp_parallel)

        return results

    def predict(self, data_df, new_keywords=None, exit_after_commonsupport=False):
        """
        Compute all effects.

        meth:`~ModifiedCausalForest.train` method must be run beforehand.

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the predictions. It must contain information about features (and
            treatment if effects for treatment specific subpopulations are desired as well).

        new_keywords: Dictionary (or None). Default is None.
            Parameters of mcf instance to be changed. The keys in the dictionary are the parameters
            to be changed when running this method (and all methods that are run subsequently, like
            analyse or sensitivity), and the values corresponding to the keys are the new value
            (None is allowed only for ignored variable-name entries; see below).
            However, not all parameters can differ from those used during training. The following
            parameters can be changed, and thus specified as keys in this dictionary (some of these
            will however not influence the results of this method, but of other other used with the
            same instance):
            'cs_type' (can only be changed to 0, use only if prediction data are already on common
                       support);
            'gen_output_type';
            'var_x_name_balance_test_ord',  'var_x_name_balance_test_unord',
            'var_x_name_balance_bgate', 'var_x_name_ba', 'var_z_name_ord', 'var_z_name_unord',
            'p_ba', 'p_ba_adj_method', 'p_ba_pos_weights_only', 'p_ba_use_x', 'p_ba_use_prop_score',
            'p_ba_use_prog_score', 'p_ba_ridge', 'p_ate_no_se_only', 'p_atet', 'p_gatet',
            'p_bgate', 'p_cbgate', 'p_iate', 'p_iate_se', 'p_iate_m_ate', 'p_bgate_sample_share',
            'p_gates_minus_previous', 'p_gates_smooth_bandwidth', 'p_gates_smooth',
            'p_gates_smooth_no_evalu_points', 'p_gates_no_evalu_points',
            'p_qiate', 'p_qiate_se', 'p_qiate_m_mqiate', 'p_qiate_m_opp', 'p_qiate_no_of_quantiles',
            'p_qiate_smooth', 'p_qiate_smooth_bandwidth', 'p_qiate_bias_adjust', 'p_bt_yes',
            'p_choice_based_sampling', 'p_choice_based_probs', 'p_cond_var', 'p_knn', 'p_knn_const',
            'p_knn_min_k', 'p_nw_bandw', 'p_nw_kern', 'p_ci_level', 'p_se_boot_ate',
            'p_se_boot_gate', 'p_se_boot_iate', 'p_se_boot_qiate';
            'post_bin_corr_threshold', 'post_bin_corr_yes', 'post_est_stats', 'post_kmeans_yes',
            'post_kmeans_no_of_groups', 'post_kmeans_max_tries', 'post_kmeans_min_size_share',
            'post_kmeans_replications', 'post_kmeans_single', 'post_random_forest_vi',
            'post_relative_to_first_group_only', 'post_plots', 'post_tree'.
            For 'p_ba', 'p_ba_use_prop_score', 'p_ba_use_prog_score', 'p_qiate', 'p_qiate_se'
            and 'p_bt_yes', new_keywords accepts only False. Passing True raises ValueError, even
            if the flag is already enabled on the instance.
            The additional key 'no_treatment_versions' disables prediction for treatment versions,
            even if versions were used during training. Its value is currently ignored: any value
            other than None, including False, triggers this change. None raises ValueError.
            Omit the key to retain prediction for treatment versions.
            A dictionary, including {}, is accepted only before the first prediction on an
            instance already used for training. Otherwise, ValueError is raised.
            For repeated predictions with different settings, create copies
            of the trained instance with copy.deepcopy() before its first prediction.
            None skips this check.
            Updates to variable names require a list or tuple of strings, even for one variable.
            Names are casefolded and must match the feature lists stored in the trained instance.
            'var_x_name_balance_test_ord' and 'var_z_name_ord' require ordered features;
            'var_x_name_balance_test_unord' and 'var_z_name_unord' require unordered features.
            'var_x_name_balance_bgate' and 'var_x_name_ba' accept either feature type.
            Variable-name entries are checked only when their use is enabled after all updates.
            'var_x_name_ba' requires both 'p_ba' and 'p_ba_use_x'; an active selection must equal
            the trained set. Changing that set requires retraining.
            The balancing-test lists require 'p_bt_yes'. 'var_x_name_balance_bgate' requires
            'p_bgate' and enabled GATEs; CBGATEs do not use this balancing list.
            'p_ate_no_se_only' disables checks of grouping and balancing-variable updates.
            Inactive entries, including None, are ignored and the stored lists are retained.
            Dictionary order does not affect which variable-name entries are checked.
            For unordered features, supply original names; the internal '_prime' suffix is added.

        exit_after_commonsupport : Boolean, optional
            If True, stop after common-support processing without estimating effects.
            Return the processed prediction data in 'inputdata_on_support'.
            The default is False.

        Returns
        -------
        results : Dictionary.
            Contains the results. The descriptions of 'gate_diff', 'bgate_diff' and 'cbgate_diff'
            below assume 'p_gates_minus_previous' is False. If True, these entries compare each
            evaluation point with its predecessor; the first point has no preceding comparison.
            Their '_diff_se' entries contain the corresponding standard errors.
            Entries for GATEs, BGATEs, CBGATEs, QIATEs and balancing tests are None when that
            estimation is disabled, including their standard errors and effect differences.
            'gate_names_values' is None when all three GATE variants are disabled.
            The dictionary has the following structure:
            'ate': ATE, 'ate_se': Standard error of ATE. None if exit_after_commonsupport is True.
            'ate_effect_list': List of names of estimated effects. None if exit_after_commonsupport
                is True.
            'gate': GATE, 'gate_se': SE of GATE. None if exit_after_commonsupport is True.
            'gate_diff': GATE minus ATE. None if exit_after_commonsupport is True.
            'gate_diff_se': Standard error of GATE minus ATE. None if exit_after_commonsupport is
                True.
            'cbgate': cbGATE (all covariates balanced). None if exit_after_commonsupport is True.
            'cbgate_se': Standard error of CBGATE. None if exit_after_commonsupport is True.
            'cbgate_diff': CBGATE minus ATE. None if exit_after_commonsupport is True.
            'cbgate_diff_se': Standard error of CBGATE minus ATE. None if exit_after_commonsupport
                is True.
            'bgate': BGATE (only prespecified covariates balanced). None if exit_after_commonsupport
                is True.
            'bgate_se': Standard error of BGATE. None if exit_after_commonsupport is True.
            'bgate_diff': BGATE minus ATE. None if exit_after_commonsupport is True.
            'bgate_diff_se': Standard errror of BGATE minus ATE. None if exit_after_commonsupport
                is True.
            'gate_names_values': Dictionary: Order of gates parameters and name and values of GATE
                effects.
            None if exit_after_commonsupport is True.
            'qiate': QIATE, 'qiate_se': Standard error of QIATEs. None if exit_after_commonsupport
                is True.
            'qiate_mmed': QIATE(q) - QIATE(0.5).
                None if not computed or exit_after_commonsupport is True.
            'qiate_mmed_se': Standard errors of qiate_mmed.
                None if not computed or exit_after_commonsupport is True.
            'qiate_mopp': QIATE(q) - QIATE(1 - q).
                None if not computed or exit_after_commonsupport is True.
            'qiate_mopp_se': Standard errors of qiate_mopp.
                None if not computed or exit_after_commonsupport is True.
            'iate': NumPy array or None
                Individual effects with shape (N, Y, C, 2): retained prediction observations,
                outcomes, treatment comparisons, and effect type. The last axis contains IATEs
                at index 0 and IATE-minus-ATE deviations at index 1. Index 1 contains NaN when
                'p_iate_m_ate' is False.
                Available even when 'iate_data_df' is not returned. None if IATE estimation is
                disabled or exit_after_commonsupport is True.
            'iate_se': NumPy array or None
                Standard errors with the same shape and ordering as 'iate'. None when 'p_iate_se'
                is False, IATE estimation is disabled, or exit_after_commonsupport is True.
            'iate_eff': Boolean or None
                Resolved 'gen_iate_eff' flag; indicates whether reported IATEs use both rounds.
                Returned even when IATE estimation is disabled. None if exit_after_commonsupport
                is True.
            'iate_data_df': DataFrame or None
                Processed prediction data with potential outcomes and IATEs,
                plus requested standard errors and IATE-minus-ATE results.
                Contains only observations retained after data preparation
                and common-support trimming.
                None if exit_after_commonsupport is True, IATE estimation
                is disabled, or IATE DataFrame return is disabled.
                See _int_with_output and _int_return_iate_sp.
            'iate_names_dic': Tuple of two dictionaries or None
                Column-name metadata for potential outcomes, IATEs and
                requested standard errors and IATE-minus-ATE results.
                Index 0 includes names for all computed effect comparisons.
                Index 1 restricts effect names to comparisons against the
                first treatment value; potential-outcome names are unchanged.
                Entries for unrequested standard errors or IATE-minus-ATE results are None.
                Returned even when IATE DataFrame return is disabled.
                None if IATE estimation is disabled or exit_after_commonsupport is True.
            'bala': Effects of balancing tests. None if exit_after_commonsupport is True.
            'bala_se': Standard error of effects of balancing tests. None if
                exit_after_commonsupport is True.
            'bala_effect_list': Names of effects of balancing tests. None if
                exit_after_commonsupport is True.
            'common_support_probabilities': DataFrame containing treatment probabilities,
                the observation identifier, and an indicator for being on common support.
                None if _int_with_output is False or cs_type is 0.
            'path_output' : Path or None
                Directory in which output is saved.
                None if _int_with_output is False.
            'inputdata_on_support': Processed prediction data retained after common-support
                processing when exit_after_commonsupport is True; otherwise None.

        """
        if new_keywords is not None:
            change_keywords(self, 'unconfound', 'predict', new_keywords)

        self.predict_done = True
        self.instance_used_for_prediction = True

        results = predict_main(self, data_df, exit_after_commonsupport)
        if self.int_cfg.mp_use_old_ray:
            check_ray_shutdown(self.int_cfg.mp_ray_shutdown, self.gen_cfg.mp_parallel)

        return results

    def predict_different_allocations(self, data_df, allocations_df):
        """
        Predict average potential outcomes for different allocations.

        meth:`~ModifiedCausalForest.train` method must be run beforehand. The details of this
        methods are described in the working paper by Busshoff (2026).

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the predictions. It must contain information about features (and
            treatment if effects for treatment specific subpopulations are desired as well).

        allocations_df : DataFrame or None
            Required argument. Pass None to skip evaluating supplied allocations;
            'alloc_df' in the returned dictionary is then None.
            Otherwise, each column specifies an allocation, with one treatment
            code per observation. Match the number and row order of observations
            in data_df. Rows are matched by position; index labels are ignored.
            Use the trained model's treatment codes, after any recoding performed
            during training.

        Returns
        -------
        results : Dictionary
            Dictionary with the following entries:
            'ate': Array of average treatment effects comparing allocations in which
                everyone receives the same treatment.
            'ate_se': Array of standard errors corresponding to 'ate'.
            'ate_effect_list': Treatment comparisons in the order used by 'ate'.
            'alloc_df': DataFrame containing estimates and their variances for the
                supplied allocations and all pairwise differences between them.
                Rows are 'est' and 'var'; columns are named '<outcome>_<allocation>'
                or '<outcome>_<allocation1>_m_<allocation2>', where each difference
                is allocation1 minus allocation2.
                None if allocations_df is None.

            The output directory is stored in self.gen_cfg.outpath.

        Raises
        ------
        NotImplementedError
            If allocations_df is not None and any of the following applies:
            treatments are continuous, 'p_choice_based_sampling' is True,
            or 'gen_weighted' is True.

        """
        self.predict_different_allocations_done = True

        results, self.gen_cfg.outpath = predict_different_allocations_main(self,
                                                                           data_df, allocations_df
                                                                           )
        if self.int_cfg.mp_use_old_ray:
            check_ray_shutdown(self.int_cfg.mp_ray_shutdown, self.gen_cfg.mp_parallel)

        return results

    def predict_iv(self, data_df, new_keywords=None):
        """
        Compute all effects for instrument mcf (possibly in 2 differnt ways).

        :meth:`~ModifiedCausalForest.train_iv` method must be run beforehand.

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the predictions. It must contain information about features (and
            treatment if effects for treatment specific subpopulations are desired as well).

        new_keywords: Dictionary (or None). Default is None.
            Parameters of mcf instance to be changed. The keys in the dictionary are the parameters
            to be changed when running this method (and all methods that are run subsequently, like
            analyse or sensitivity), and the values corresponding to the keys are the new value
            (None is allowed only for ignored variable-name entries; see below).
            However, not all parameters can differ from those
            used during training. The following parameters can be changed, and thus specified as
            keys in this dictionary:
            'gen_output_type',
            'cs_type' (can only be changed to 0, use only if prediction data are already on common
                       support);
            'var_x_name_balance_test_ord',  'var_x_name_balance_test_unord',
            'var_x_name_balance_bgate', 'var_x_name_ba', 'var_z_name_ord', 'var_z_name_unord',
            'p_ba', 'p_ba_adj_method', 'p_ba_pos_weights_only', 'p_ba_use_x', 'p_ba_use_prop_score',
            'p_ba_use_prog_score', 'p_ba_ridge', 'p_ate_no_se_only', 'p_atet', 'p_gatet',
            'p_bgate', 'p_cbgate', 'p_iate', 'p_iate_se', 'p_iate_m_ate', 'p_bgate_sample_share',
            'p_gates_minus_previous', 'p_gates_smooth_bandwidth', 'p_gates_smooth',
            'p_gates_smooth_no_evalu_points', 'p_gates_no_evalu_points', 'p_qiate', 'p_qiate_se',
            'p_qiate_m_mqiate', 'p_qiate_m_opp', 'p_qiate_no_of_quantiles', 'p_qiate_smooth',
            'p_qiate_smooth_bandwidth', 'p_qiate_bias_adjust', 'p_bt_yes',
            'p_choice_based_sampling', 'p_choice_based_probs', 'p_cond_var', 'p_knn', 'p_knn_const',
            'p_knn_min_k', 'p_nw_bandw', 'p_nw_kern', 'p_ci_level', 'p_iv_aggregation_method',
            'p_se_boot_ate', 'p_se_boot_gate', 'p_se_boot_iate', 'p_se_boot_qiate',
            'post_bin_corr_threshold', 'post_bin_corr_yes', 'post_est_stats',
            'post_kmeans_yes', 'post_kmeans_no_of_groups', 'post_kmeans_max_tries',
            'post_kmeans_min_size_share', 'post_kmeans_replications', 'post_kmeans_single',
            'post_random_forest_vi', 'post_relative_to_first_group_only', 'post_plots', 'post_tree'.
            For 'p_ba', 'p_ba_use_prop_score', 'p_ba_use_prog_score', 'p_qiate', 'p_qiate_se'
            and 'p_bt_yes', new_keywords accepts only False. Passing True raises ValueError, even
            if the flag is already enabled on the instance.
            A dictionary, including {}, is accepted only before the first prediction on an
            instance already used for training. Otherwise, ValueError is raised.
            For repeated predictions with different settings, create copies
            of the trained instance with copy.deepcopy() before its first prediction.
            None skips this check.
            Updates to variable names require a list or tuple of strings, even for one variable.
            Names are casefolded and must match the feature lists stored in the trained instance.
            'var_x_name_balance_test_ord' and 'var_z_name_ord' require ordered features;
            'var_x_name_balance_test_unord' and 'var_z_name_unord' require unordered features.
            'var_x_name_balance_bgate' and 'var_x_name_ba' accept either feature type.
            Variable-name entries are checked only when their use is enabled after all updates.
            'var_x_name_ba' requires both 'p_ba' and 'p_ba_use_x'; an active selection must equal
            the trained set. Changing that set requires retraining.
            The balancing-test lists require 'p_bt_yes'. 'var_x_name_balance_bgate' requires
            'p_bgate' and enabled GATEs; CBGATEs do not use this balancing list.
            'p_ate_no_se_only' disables checks of grouping and balancing-variable updates.
            Inactive entries, including None, are ignored and the stored lists are retained.
            Dictionary order does not affect which variable-name entries are checked.
            For unordered features, supply original names; the internal '_prime' suffix is added.

        Returns
        -------
        results_global : Dictionary.
            Contains the results. This dictionary has the following structure:
            'ate': LATE, 'ate_se': Standard error of LATE,
            'ate_effect_list': List of names of estimated effects,
            'ate_1st': ATE 1st stage, 'ate_1st_se': Standard error of ATE (1st)
            'ate 1st_effect_list': List of names of estimated effects (1st),
            'ate_redf': ATE reduced form, 'ate_redf_se': Standard error of ATE of reduced form,
            'ate redf_effect_list': List of names of estimated effects (red.f.),
            'gate', 'gate_se', 'gate_diff', 'gate_diff_se': None,
            'cbgate', 'cbgate_diff', 'cbgate_diff_se': None,
            'bgate', 'bgate_se', 'bgate_diff', 'bgate_diff_se': None,
            'gate_names_values': Dictionary: Order of gates parameters and name and values of
                                             LGATE effects.
            'iate', 'iate_1st', 'iate_redf': NumPy arrays or None
                Individual effects for the IV model, first stage and reduced form, respectively.
                Shape is (n_prediction, n_outcomes, 1, 2); the first stage has one outcome.
                The final axis contains the effect at index 0 and its deviation from the
                corresponding local LATE or stage ATE at index 1. Unrequested deviations are NaN.
                None if IATE estimation is disabled. Available even when DataFrame return is off.
            'iate_se', 'iate_1st_se', 'iate_redf_se': NumPy arrays or None
                Standard errors with the same axes as the corresponding individual-effect array.
                None if IATE estimation or IATE standard errors are disabled.
                Unrequested deviation standard errors are NaN.
            'iate_data_df': DataFrame or None
                Processed prediction data with LIATE results, including
                potential outcomes and, when requested, standard errors and
                differences from the corresponding average effects.
                None if IATE estimation or IATE DataFrame return is disabled;
                see _int_with_output and _int_return_iate_sp.
            'iate_1st_pred_df': DataFrame or None
                Same structure and None conditions for first-stage IATEs.
            'iate_redf_pred_df': DataFrame or None
                Same structure and None conditions for reduced-form IATEs.
            'iate_names_dic': Tuple of two dictionaries or None
                LIATE column-name metadata. The dictionaries have the same
                structure as 'iate_names_dic' returned by predict().
                None if IATE estimation is disabled; otherwise available
                even when IATE DataFrame return is disabled.
            'iate_1st_names_dic': Tuple of two dictionaries or None
                Corresponding metadata and None condition for first-stage IATEs.
            'iate_redf_names_dic': Tuple of two dictionaries or None
                Corresponding metadata and None condition for reduced-form IATEs.
            'qiate', 'qiate_se', 'qiate_diff', 'qiate_diff_se', 'qiate_mopp',
            'qiate_mopp_se': Always None; QLIATE estimation is not implemented
                for IV prediction.
            'bala_1st': Effects of balancing tests (1st stage),
            'bala_1st_se': Standard error of effects of balancing tests (1st),
            'bala_1st_effect_list': Names of effects of balancing tests (1st),
            'bala_redf': Effects of balancing tests (reduced form),
            'bala_redf_se': Standard error of effects of balancing tests (red.),
            'bala_redf_effect_list': Names of effects of balancing tests (red.).
            'common_support_probabilities': DataFrame (or None)
                Estimated probabilities of each instrument value for the prediction
                sample before common-support trimming, plus the observation identifier
                and 'on_support' (1 if retained, 0 if excluded).
                None if 'cs_type' is 0 or '_int_with_output' is False.
            'path_output' : Path or None
                Directory in which output is saved.
                None if _int_with_output is False.

            It is empty if the IV estimation method 'global' has not been used.

        results_local : Dictionary.
            Empty if 'local' is not selected in 'p_iv_aggregation_method'.
            Otherwise, contains the keys listed for results_global and the additional
            key 'cbgate_se'.
            'ate' and 'ate_se' contain LATE and its standard error from local aggregation.
            Group effects also use local aggregation:
            'gate': LGATE,
            'bgate': LBGATE (prespecified covariates balanced),
            'cbgate': LCBGATE (all covariates balanced).
            Their '_se' entries contain the standard errors. Their '_diff' and
            '_diff_se' entries contain the differences selected by 'p_gates_minus_previous'
            and the corresponding standard errors.
            Group-effect entries are None when their estimators are disabled.
            The remaining entries have the same meaning as in results_global.

        """
        if new_keywords is not None:
            change_keywords(self, 'iv', 'predict', new_keywords)

        self.predict_iv_done = True
        self.instance_used_for_prediction = True

        # Reduce sample size to upper limit
        data_df, rnd_reduce, txt_red = check_reduce_dataframe(
            data_df,
            title='Prediction', max_obs=self.int_cfg.max_obs_prediction,
            seed=124535, ignore_index=True,
            )
        if rnd_reduce and self.gen_cfg.with_output:
            print_mcf(self.gen_cfg, txt_red, summary=True)

        results_global, results_local = predict_iv_main(self, data_df)

        if self.int_cfg.mp_use_old_ray:
            check_ray_shutdown(self.int_cfg.mp_ray_shutdown, self.gen_cfg.mp_parallel)

        return results_global, results_local

    def analyse(self, results, new_keywords=None):
        """
        Analyse estimated IATEs with various descriptive tools.

        Parameters
        ----------
        results : Dictionary
            Results from predict(), or one nonempty dictionary returned by predict_iv():
            results_global or results_local. For IV estimation, pass that dictionary rather than
            the two-element return tuple. Use results from this trained instance, with IATE
            estimation and IATE DataFrame return enabled; see 'post_est_stats'.
            For IV results, correlations and sorted-effect/density analyses cover LIATEs,
            first-stage IATEs and reduced-form IATEs. Clustering and descriptive random-forest
            and regression-tree analyses use LIATEs only.
        
        new_keywords: Dictionary (or None). Default is None.
            Parameters of mcf instance to be changed. The keys in the dictionary are the parameters
            to be changed when running this method (and all methods that are run subsequently, like
            analyse or sensitivity), and the values corresponding to the keys are the new value
            (None is allowed only for ignored variable-name entries; see below).
            However, not all parameters can differ from those
            used during training. The following parameters can be changed, and thus specified as
            keys in this dictionary (some of these will however not influence the results of this
            method, but of other other used with the same instance):
            'gen_output_type';
            'var_x_name_balance_test_ord',  'var_x_name_balance_test_unord',
            'var_x_name_balance_bgate', 'var_x_name_ba', 'var_z_name_ord', 'var_z_name_unord',
            'p_ba', 'p_ba_adj_method', 'p_ba_pos_weights_only', 'p_ba_use_x', 'p_ba_use_prop_score',
            'p_ba_use_prog_score', 'p_ba_ridge', 'p_ate_no_se_only', 'p_atet', 'p_gatet',
            'p_bgate', 'p_cbgate', 'p_iate', 'p_iate_se', 'p_iate_m_ate', 'p_bgate_sample_share',
            'p_gates_minus_previous', 'p_gates_smooth_bandwidth', 'p_gates_smooth',
            'p_gates_smooth_no_evalu_points', 'p_gates_no_evalu_points', 'p_qiate', 'p_qiate_se',
            'p_qiate_m_mqiate', 'p_qiate_m_opp', 'p_qiate_no_of_quantiles', 'p_qiate_smooth',
            'p_qiate_smooth_bandwidth', 'p_qiate_bias_adjust', 'p_bt_yes',
            'p_choice_based_sampling', 'p_choice_based_probs', 'p_cond_var', 'p_knn', 'p_knn_const',
            'p_knn_min_k', 'p_nw_bandw', 'p_nw_kern', 'p_ci_level', 'p_se_boot_ate',
            'p_se_boot_gate', 'p_se_boot_iate', 'p_se_boot_qiate';
            'post_bin_corr_threshold', 'post_bin_corr_yes', 'post_est_stats', 'post_kmeans_yes',
            'post_kmeans_no_of_groups', 'post_kmeans_max_tries', 'post_kmeans_min_size_share',
            'post_kmeans_replications', 'post_kmeans_single', 'post_random_forest_vi',
            'post_relative_to_first_group_only', 'post_plots', 'post_tree'.
            For 'p_ba', 'p_ba_use_prop_score', 'p_ba_use_prog_score', 'p_qiate', 'p_qiate_se'
            and 'p_bt_yes', new_keywords accepts only False. Passing True raises ValueError, even
            if the flag is already enabled on the instance.
            Updates to variable names require a list or tuple of strings, even for one variable.
            Names are casefolded and must match the feature lists stored in the trained instance.
            'var_x_name_balance_test_ord' and 'var_z_name_ord' require ordered features;
            'var_x_name_balance_test_unord' and 'var_z_name_unord' require unordered features.
            'var_x_name_balance_bgate' and 'var_x_name_ba' accept either feature type.
            Variable-name entries are checked only when their use is enabled after all updates.
            'var_x_name_ba' requires both 'p_ba' and 'p_ba_use_x'; an active selection must equal
            the trained set. Changing that set requires retraining.
            The balancing-test lists require 'p_bt_yes'. 'var_x_name_balance_bgate' requires
            'p_bgate' and enabled GATEs; CBGATEs do not use this balancing list.
            'p_ate_no_se_only' disables checks of grouping and balancing-variable updates.
            Inactive entries, including None, are ignored and the stored lists are retained.
            Dictionary order does not affect which variable-name entries are checked.
            For unordered features, supply original names; the internal '_prime' suffix is added.
            'cs_type' is also accepted, but only with value 0. 'p_iv_aggregation_method' is accepted
            only if predict_iv() has set self.predict_iv_done to True; otherwise, ValueError is
            raised. These settings persist on the instance. They do not rerun common-support checks
            or recompute the supplied effects when analyse() is called.
            A dictionary, including {}, requires an instance already used for prediction by
            predict() or predict_iv(); otherwise, ValueError is raised. None skips this check.
            
        Raises
        ------
        ValueError
            Some of the attribute are not compatible with running this method.

        Returns
        -------
        results_plus_cluster : Dictionary or None
            If 'post_kmeans_yes' is True, a copy of results with cluster labels
            added to 'iate_data_df'. 'IATE_Cluster' contains integer labels starting at 0.
            If 'post_kmeans_single' is True, additional 'IATE_Cluster_<effect>' columns
            contain labels from separate clustering runs for the selected effects.
            The returned 'iate_data_df' may contain fewer observations when
            '_int_max_obs_kmeans' limits the sample used for clustering.
            If 'post_kmeans_yes' is False, returns None; the other enabled analyses still run.

        """
        estimator = 'iv' if self.predict_iv_done else 'unconfound'

        if new_keywords is not None:
            change_keywords(self, estimator, 'analyse', new_keywords)

        results_plus_cluster = analyse_main(self, results)

        if self.int_cfg.mp_use_old_ray:
            check_ray_shutdown(self.int_cfg.mp_ray_shutdown, self.gen_cfg.mp_parallel)

        return results_plus_cluster

    def sensitivity(self, train_df, predict_df, *,
                    results,
                    sens_cbgate=None, sens_bgate=False, sens_gate=False,
                    sens_iate=None, sens_iate_se=False,
                    sens_scenarios=('basic',), sens_cv_k=5, sens_replications=2,
                    sens_reference_population=None,
                    ):
        """
        Compute simulation based sensitivity indicators.

        Parameters
        ----------
        train_df : DataFrame.
            Data with real outcomes, treatments, and covariates. Data will be transformed to
            compute sensitivity indicators.

        predict_df : DataFrame or None
            Required argument. Prediction sample for sensitivity estimation.
            Previous IATE results supplied through results take precedence over
            this argument. Otherwise, a DataFrame supplied here is used; any
            other value makes the method use a copy of train_df.
            The selected sample must satisfy the input requirements of
            :meth:`~ModifiedCausalForest.predict` for the enabled effects.

        results : Dictionary or None
            Required keyword argument. Pass None when no previous prediction
            results are to be used.
            Otherwise, supply the output of :meth:`~ModifiedCausalForest.predict`.
            If it contains an 'iate_data_df' DataFrame and 'iate_names_dic',
            that DataFrame replaces predict_df and enables IATE computation.
            Common-support trimming is then disabled for the placebo estimations,
            and original and placebo IATEs are compared when output is enabled.
            A result dictionary without IATEs does not affect the analysis.

        sens_gate : Boolean (or None), optional
            Compute GATEs for sensitivity analysis.
            Automatically enabled if sens_bgate or sens_cbgate is True, even
            when sens_gate is False or None.
            Otherwise, only True enables GATEs.
            The default is False.

        sens_bgate : Boolean (or None), optional
            Compute BGATEs for sensitivity analysis.
            Requires 'p_bgate' to be enabled on this instance; otherwise raises ValueError.
            Default (or None) is False.

        sens_cbgate : Boolean (or None), optional
            Compute CBGATEs for sensitivity analysis.
            Requires 'p_cbgate' to be enabled on this instance; otherwise raises ValueError.
            Default (or None) is False.

        sens_iate : Boolean (or None), optional
            Compute IATEs for sensitivity analysis.
            If results contains IATEs in 'iate_data_df' and their names in
            'iate_names_dic', IATE computation is enabled even if sens_iate is False.
            Otherwise, IATEs are computed only if sens_iate is True.
            The default is None.

        sens_iate_se : Boolean (or None), optional
            Compute Standard errors of IATEs for sensitivity analysis.
            Default (or None) is False.

        sens_scenarios : String, list or tuple of strings (or None), optional
            Only the 'basic' scenario is supported. Pass 'basic' as a string
            or in a nonempty list or tuple. Other scenario names raise ValueError.
            This scenario draws placebo treatments using estimated treatment
            probabilities for observations in the selected reference population.
            Their outcomes remain unchanged, giving a scenario with no treatment effect.
            Default (or None) is ('basic',).

        sens_cv_k : Integer or float (or None), optional
            Number of cross-fitting folds for estimating treatment probabilities
            used to simulate placebo treatments.
            None or values below 0.5 select 5 folds; otherwise, the value is
            rounded with Python's round().
            Use at least 2 folds and no more folds than the number of training
            observations remaining after data preparation.
            The default (or None) is 5.

        sens_replications : Integer or float (or None), optional
            Number of placebo replications per sensitivity scenario.
            Each replication simulates new treatments, retrains the model and
            predicts the requested effects.
            None or values below 0.5 select 2 replications; otherwise, the value
            is rounded with Python's round().
            Choose a value that produces at least 1 replication.
            The default (or None) is 2.

        sens_reference_population : Integer or float (or None), optional
            Treatment group in train_df used to construct the placebo training samples.
            Specify the treatment code after train_df has been recoded to consecutive
            integers starting at 0. For example, input values 1, 4 and 5 become 0, 1 and 2.
            Treatment probabilities are estimated using all retained training observations;
            placebo treatments are then simulated only for the selected group.
            The prediction sample is determined separately by predict_df and results.
            Default (or None) selects the largest treatment group after data preparation.

        Returns
        -------
        results_avg : Dictionary
            Dictionary keyed by sensitivity scenario. Currently, only 'basic'
            is supported; access its results as results_avg['basic'].
            Each value has the structure of a result dictionary returned by
            :meth:`~ModifiedCausalForest.predict`.
            ATEs and available GATE, BGATE and CBGATE estimates, including their
            differences, are averaged across replications. Their standard errors
            are also averaged arithmetically.
            If IATEs are computed, 'iate_data_df' contains averages across
            replications for observation identifiers present in every replication.
            Other entries are taken from the first replication.

        Raises
        ------
        NotImplementedError
            If feature selection is enabled ('fs_yes=True') or treatments
            are continuous ('gen_d_type' is 'continuous').

        """
        inconsistencies_sens(self)
        results_avg = sensitivity_main(self, train_df, predict_df=predict_df, results=results,
                                       sens_cbgate=sens_cbgate, sens_bgate=sens_bgate,
                                       sens_gate=sens_gate, sens_iate=sens_iate,
                                       sens_iate_se=sens_iate_se, sens_scenarios=sens_scenarios,
                                       sens_cv_k=sens_cv_k, sens_replications=sens_replications,
                                       sens_reference_population=sens_reference_population
                                       )
        if self.int_cfg.mp_use_old_ray:
            check_ray_shutdown(self.int_cfg.mp_ray_shutdown, self.gen_cfg.mp_parallel)

        return results_avg
