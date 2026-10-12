from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
from pandas import DataFrame

from mcf.mcf_feature_selection import FsCfg
from mcf.optpolicy_data import dataframe_checksum
from mcf import optpolicy_init as op_init
from mcf import optpolicy_methods as op_methods
from mcf import optpolicy_version as op_version
from mcf.mcf_print_stats import print_mcf


class OptimalPolicy:
    """
    Optimal policy learning

    Parameters
    ----------
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

    estrisk_value : Float or integer (or None), optional
        Finite multiplier k in 'policy_score - k * standard_error', used to adjust policy scores
        for estimation risk. NaN and infinite values raise ValueError during initialization.
        If computing the adjustment overflows floating-point arithmetic, adjustment raises
        ValueError even when the multiplier, original scores and standard errors are finite.
        Default (or None) is 1.

    fair_adjust_target : String (or None), optional
        Target of the fairness adjustment in :meth:`~OptimalPolicy.solvefair`.
        ``'scores'`` adjusts policy scores; ``'xvariables'`` adjusts decision variables;
        ``'scores_xvariables'`` adjusts both.
        For ``'gen_method'`` == ``'best_policy_score'``, ``'xvariables'`` raises ValueError during
        initialization. Use ``'scores'`` or ``'scores_xvariables'`` instead.
        Default (or None): ``'scores'`` for ``'best_policy_score'``; ``'xvariables'`` otherwise.

    fair_consistency_test : Boolean (or None), optional
        When 'fair_adjust_target' is 'scores' or 'scores_xvariables', compare two ways to adjust
        each pairwise score difference: subtract the individually adjusted scores, or apply the
        fairness adjustment directly to the original difference.
        Scores are excluded if the sum of their original and adjusted standard deviations is at
        most 1e-8. All pairs of the remaining scores are compared.
        Report descriptive diagnostics: mean absolute discrepancy divided by the standard
        deviation of the original score difference, share of matching signs and Pearson correlation.
        These additional adjustments can cost more than the original fairness correction.
        Default (or None) is False.

    fair_cont_min_values : Integer or float (or None), optional
        Minimum number of distinct non-missing values for an ordered decision variable
        to be treated as continuous when applying fairness corrections.
        Ordered decision variables with fewer values are treated as discrete.
        Unordered decision variables are always treated as discrete.
        The threshold also determines whether protected variables are treated as continuous
        when computing the reported fairness statistics.
        Values of at least 1 are rounded to the nearest integer.
        None or values below 1 use 20. The default is 20.

    fair_material_disc_method : String (or None), optional
        Method for preparing materially relevant features for quantile fairness corrections.
        ``'NoDiscretization'``: Keep the original values and treat the features as continuous.
        ``'EqualCell'``: Bin each feature with more than ``'fair_material_max_groups'``
        distinct values into approximately equally populated groups, then combine the
        resulting values into joint cells.
        ``'Kmeans'``: Cluster the features jointly using ``'fair_material_max_groups'`` clusters.
        Before applying the selected method, if the sum of the distinct-value counts
        across materially relevant features is at most ``'fair_material_max_groups'``,
        exact cells defined by observed feature combinations are used instead.
        When adjusting decision variables for policy trees, ``'NoDiscretization'``
        is replaced by ``'Kmeans'`` before this check.
        The default (or None) is ``'Kmeans'``.

    fair_protected_disc_method : String (or None), optional
        Method for preparing protected features for quantile fairness corrections.
        ``'NoDiscretization'``: Keep the original values and treat the features as continuous.
        ``'EqualCell'``: Bin each feature with more than ``'fair_protected_max_groups'``
        distinct values into approximately equally populated groups, then combine the
        resulting values into joint cells.
        ``'Kmeans'``: Cluster the features jointly using ``'fair_protected_max_groups'`` clusters.
        Before applying the selected method, if the sum of the distinct-value counts
        across protected features is at most ``'fair_protected_max_groups'``,
        exact cells defined by observed feature combinations are used instead.
        When adjusting decision variables for policy trees, ``'NoDiscretization'``
        is replaced by ``'Kmeans'`` before this check.
        The default (or None) is ``'Kmeans'``.

    fair_material_max_groups : Integer or float (or None), optional
        Grouping parameter for materially relevant features in ``'Quantiled'`` fairness
        corrections. Its meaning depends on ``'fair_material_disc_method'``.
        With ``'EqualCell'``, it is the maximum number of groups per feature.
        Observed combinations of these groups define joint cells, so the number of
        joint cells can exceed this value.
        With 'Kmeans', it is the requested number of clusters across all features, capped
        at the number of distinct observed feature profiles.
        If the sum of distinct-value counts across the features is at most this value,
        exact cells are used instead of the selected discretization method.
        Finite numeric values are rounded to the nearest integer and bounded below by 1.
        NaN and infinite values raise ValueError. The default (or None) is 5.


    fair_protected_max_groups : Integer or float (or None), optional
        Grouping parameter for protected features in ``'Quantiled'`` fairness corrections.
        Its meaning depends on ``'fair_protected_disc_method'``.
        With ``'EqualCell'``, it is the maximum number of groups per feature.
        Observed combinations of these groups define joint cells, so the number of
        joint cells can exceed this value.
        With 'Kmeans', it is the requested number of clusters across all features, capped
        at the number of distinct observed feature profiles.
        If the sum of distinct-value counts across the features is at most this value,
        exact cells are used instead of the selected discretization method.
        Finite numeric values are rounded to the nearest integer and bounded below by 1.
        NaN and infinite values raise ValueError. The default (or None) is 5.


    fair_regression_method : String (or None), optional
        Regression method for 'Mean' and 'MeanVar' fairness corrections; see 'fair_type'.
        Available methods are ``'RandomForest'``, ``'RandomForestNminl5'``,
        ``'RandomForestNminls5'``, ``'SupportVectorMachine'``, ``'SupportVectorMachineC2'``,
        ``'SupportVectorMachineC4'``, ``'AdaBoost'``, ``'AdaBoost100'``, ``'AdaBoost200'``,
        ``'GradBoost'``, ``'GradBoostDepth6'``, ``'GradBoostDepth12'``, ``'NeuralNet'``,
        ``'NeuralNetLarge'``, ``'NeuralNetLarger'``, ``'Mean'``.
        With ``'automatic'``, select a method separately for each regression used to adjust a policy
        score or continuous decision variable. Selection can differ between mean and second-moment
        regressions, and between regressions using all conditioning variables and those using only
        materially relevant variables. Cross-validation uses min(5, n) folds, where n is the number
        of observations used for the adjustment. An explicitly selected method is used for all
        these regressions.
        ``'Mean'`` is included for cases in which regression methods have no explanatory power.
        Default (or None) is ``'RandomForest'``.

    fair_type : String (or None), optional
        Method for fairness corrections of policy scores and decision variables.
        ``'Mean'``: Adjust mean dependence on protected features by residualisation.
        ``'MeanVar'``: Additionally adjust conditional variances by rescaling.
        ``'Quantiled'``: Apply the quantile adjustment based on Strack and Yang (2024).
        ``'Mean'`` and ``'MeanVar'`` are available for adjusting policy scores.
        They can also adjust continuous decision variables when ``'gen_method'`` is
        not ``'policy_tree'``.
        Discrete decision variables always use ``'Quantiled'``.
        When ``'gen_method'`` is ``'policy_tree'``, all decision-variable adjustments
        use ``'Quantiled'``, regardless of this setting.
        Decision variables are classified using their ordered/unordered status and
        ``'fair_cont_min_values'``.
        See Bearth, Lechner, Mareckova, Muny (2024) for details on these methods.
        The default (or None) is ``'Quantiled'``.

    fs_yes : Boolean (or None), optional
        If True, solve() selects decision variables before fitting 'policy_tree', 'policy tree old',
        or 'bps_classifier'. The option is ignored by solve() for 'best_policy_score'.
        With solvefair(), True raises NotImplementedError for every allocation method.
        Subtract the first score in var_polscore_name from each remaining
        score and fit a separate random forest for each resulting difference.
        Use a classifier with accuracy scoring when a difference has fewer
        than 10 distinct values in the fitting sample; otherwise use a
        regressor with R2 scoring.
        Measure importance by the loss in held-out prediction score after
        permuting a feature; dummy columns of unordered features move together.
        A feature qualifies for removal if its importance is at or below
        fs_rel_vi_threshold for every score difference.
        Remove at most one feature per iteration and refit before considering
        another removal. Stop when no feature qualifies or at most two
        original features remain.
        Default (or None) is False.

    fs_rel_vi_threshold : Integer or float (or None), optional
        Threshold for feature importance in predicting policy-score
        differences relative to the first policy score. A feature qualifies
        for removal if its importance is at or below this threshold for
        every such difference.
        Values in (0, 1] are used directly; values in (1, 100] are divided
        by 100. Thus 0.05 and 5 both give 0.05, while 1 remains 1.
        None, nonpositive values and values above 100 use 0.
        Default (or None) is 0.

    fs_other_sample : Boolean (or None), optional
        True : Random sample from training data used. These observations will not be used for
        training the optimal assignment rule.
        False : Use the same sample as used for training optimal assignment rule.
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

    gen_method : String (or None), optional
        Method used to construct treatment allocations.
        ``'best_policy_score'``: Allocate directly from policy scores, producing
        additional allocation rules when treatment-share restrictions are active.
        Policy scores are required when allocating new observations as well.
        ``'bps_classifier'``: Train one classifier per allocation produced by
        ``'best_policy_score'``. New observations are allocated using decision variables.
        ``'policy_tree'``: Fit policy trees with depths and candidate splits controlled
        by the ``pt_`` parameters. Candidate split searches may be approximated.
        When two tree stages are enabled, they are optimized sequentially.
        ``'policy tree old'``: Use the older policy-tree implementation.
        For policy trees, treatment-share restrictions are incorporated through
        calibrated treatment costs; the resulting tree may exceed the share limits.
        The default (or None) is ``'best_policy_score'``.

    gen_mp_parallel : Integer or float (or None), optional
        Requested number of workers for parallel computations such as policy-tree search.
        Numeric values at most 1.5 select one worker; larger values are rounded
        to the nearest integer.
        None or a nonnumeric value selects 80% of logical CPU cores, rounded
        to the nearest integer. This is also the default behavior.
        Policy-tree parallelization uses ``'_int_mp_backend'``. If
        ``'_int_mp_use_old_ray'`` is True, the older Ray implementation is used.
        Setting this parameter to 1 runs the policy-tree search sequentially.

    gen_outfiletext : String (or None), optional
        Base name for the detailed '<name>.txt' and condensed '<name>_Summary.txt' output files.
        With '_int_with_output' enabled, initialization deletes existing files at these paths,
        including when 'gen_output_type' is 0. See '_int_output_no_new_dir' for directory reuse.
        Default (or None) is 'txtFileWithOutput'.

    gen_outpath : String or Pathlib object (or None), optional
        Base directory for text output and figures.
        For an explicitly supplied path, append a method subdirectory:
        'BPS' for 'best_policy_score', 'PT' for 'policy_tree',
        'PT_OLD' for 'policy tree old', or 'BPS_CLASSIF' for 'bps_classifier'.
        With None, use 'output' below the current working directory,
        without adding a method subdirectory.
        Missing directories are created. Directory reuse and numeric
        suffixes are controlled by _int_output_no_new_dir.
        Only used when _int_with_output is enabled. Otherwise the stored output path is None.
        Default is None.

    gen_output_type : Integer or float (or None), optional
        Destination of standard text output: 0 for terminal, 1 for files, 2 for both.
        Numeric values are converted with int(round(value)); the result must be 0, 1 or 2.
        Other rounded values raise TypeError. Halfway values round to the nearest even integer.
        '_int_with_output' set to False disables standard text output.
        Default (or None) is 2.

    gen_variable_importance : Boolean (or None), optional
        Compute variable importance for predicting each allocation with a random forest classifier
        during :meth:`~OptimalPolicy.evaluate`. Requires ``'_int_with_output'`` to be True.
        Supply predictors through ``'var_vi_x_name'`` and/or ``'var_vi_to_dummy_name'``.
        If neither list supplies variables, or any requested variable is missing from the evaluation
        data, this analysis is skipped. The random reference allocation and allocations assigning
        every observation to the same treatment are also skipped.
        Default (or None) is True.

    other_costs_of_treat : List or tuple of floats (or None), optional
        Treatment-specific costs, with one value per treatment in the same order as
        ``'var_polscore_name'``. Costs are subtracted from the corresponding policy scores
        and must therefore use the same units as those scores.
        The default is None, which sets the base cost of every treatment to zero.
        For policy trees with restrictions specified by ``'other_max_shares'``, additional
        costs are calibrated from these base costs using individual best policy score allocations
        in the training data. The adjusted costs are then used to fit the policy tree.
        This calibration does not guarantee that the resulting tree satisfies the share limits.

    other_costs_of_treat_mult : List or tuple of floats (or None), optional
        Multipliers for the additional costs calibrated for policy trees with restrictions
        specified by ``'other_max_shares'``.
        Supply one finite, strictly positive value per treatment, in the same order as
        ``'var_polscore_name'``. A scalar is not accepted.
        For each treatment, the cost used is the base cost from ``'other_costs_of_treat'``
        plus its multiplier times the additional calibrated cost.
        Values above 1 increase a positive calibrated cost increment and discourage
        allocation to that treatment; values between 0 and 1 reduce the increment.
        A zero increment is unaffected by its multiplier.
        The default is None, which uses a multiplier of 1 for every treatment.

    other_max_shares : List or tuple of floats (or None), optional
        Maximum treatment shares, with one value per treatment in the order of
        ``'var_polscore_name'``. Values must be finite and between 0 and 1
        inclusive, and their sum must be at least 1.
        Integer capacities are computed as ``floor(n * share)``, where ``n`` is
        the number of observations used by :meth:`~OptimalPolicy.solve`.
        If the total capacity falls short by one observation, one extra slot is
        assigned to a treatment with a positive share. Larger shortfalls raise
        ``ValueError``.
        The restricted ``'best_policy_score'`` rules use these integer capacities;
        the unrestricted ``'bb'`` allocation is also returned.
        For ``'bps_classifier'``, the restrictions apply to the constrained training
        allocations; classifier predictions need not satisfy them.
        Policy trees use calibrated costs and may exceed the requested shares.
        The default is None, which uses 1 for every treatment.

    pt_depth_tree_1 : Integer or float (or None), optional
        Maximum depth of the first policy tree, measured in splits from root to leaf.
        Depths of 1, 2, and 3 allow at most 2, 4, and 8 leaves, respectively.
        The fitted tree can be shallower or have fewer leaves.
        The default is 3. None or values below 1 also use 3.
        Other numeric inputs use an effective depth of ``int(round(value + 1)) - 1``.

    pt_depth_tree_2 : Integer or float (or None), optional
        Maximum depth of each additional tree fitted within a leaf of the first tree.
        Depth is measured in splits from the root of each additional tree.
        Set to 0 to skip this second stage.
        With the default depths of 3 and 1, the combined tree has at most 4 split levels
        and 16 leaves. The stages are optimized sequentially, and the fitted tree
        may be shallower or have fewer leaves.
        The default is 1. None or negative values also use 1.
        Other numeric inputs use an effective depth of ``int(round(value + 1)) - 1``.

    pt_enforce_restriction : Boolean (or None), optional
        Apply additional feasibility checks and reward penalties for treatment-share restrictions
        during tree search. Used by ``'policy_tree'`` and ``'policy tree old'`` when
        ``'other_max_shares'`` restricts at least one treatment. These checks do not guarantee
        that the final tree satisfies every share limit.
        Automatically set to False when a second tree stage is enabled. Set ``'pt_depth_tree_2'``
        to 0 when using this option.
        Default (or None) is False.

    pt_eva_cat_mult : Integer or float (or None), optional
        Multiplier controlling the maximum number of candidate splits for unordered
        (categorical) features. Only used when ``'gen_method'`` is ``'policy_tree'``.
        The candidate limit is the product of this multiplier and
        ``'pt_no_of_evalupoints'``, truncated to an integer and bounded below by 1.
        If the number of possible splits does not exceed this limit, all are considered.
        The default is 1. Explicit None, nonnumeric input, or a value below 0.1 uses 2.

    pt_no_of_evalupoints : Integer or float (or None), optional
        Controls candidate split values for ordered features in ``'policy_tree'`` and
        ``'policy tree old'``. Ordered features with at most this many distinct values retain all
        values as candidates. With more distinct values, ``'policy_tree'`` selects a grid from
        their sorted distinct values before the tree search; ``'policy tree old'`` selects
        candidates within each node.
        Also controls the candidate limit for unordered features; in ``'policy_tree'``, it is
        multiplied by ``'pt_eva_cat_mult'``.
        Smaller values reduce the search effort but can miss better splits.
        None or numeric values below 5 select 100; other numeric values are rounded to an integer.
        Default (or None) is 100.

    pt_min_leaf_size : Integer or float (or None), optional
        Minimum leaf size used when searching for policy-tree splits.
        Larger values can reduce computation time and limit overfitting.
        With None or a negative value, the minimum is computed as
        ``0.1 * n_train / 2 ** (d1 + d2)``, where ``n_train`` is the number
        of observations used for training and ``d1`` and ``d2`` are the effective
        split depths specified by ``'pt_depth_tree_1'`` and ``'pt_depth_tree_2'``.
        The denominator is the maximum leaf count implied by these depths.
        If treatment shares are restricted, this automatically computed value is
        multiplied by the smallest positive maximum treatment share.
        Supplied and automatically computed values are rounded to the nearest
        integer and bounded below by 1.
        When a second tree stage is enabled, the first-stage minimum is multiplied
        by ``2 ** d2``; the second stage uses the unscaled minimum.
        Only used for policy trees.
        The default is None.

    pt_select_values_cat : Boolean (or None), optional
        Selects how candidate splits for unordered features are approximated.
        Only used when ``'gen_method'`` is ``'policy_tree'``.
        If all distinct binary splits fit within the limit controlled by
        ``'pt_no_of_evalupoints'`` and ``'pt_eva_cat_mult'``, all are considered.
        When approximation is needed, True randomly selects a subset of categories
        and forms candidate groups from that subset, subject to the same limit.
        False orders categories by their mean policy-score differences and forms
        candidate groups from these orderings. Excess candidates are removed;
        additional random splits are added when too few candidates are obtained.
        The default (or None) is False.

    rnd_shares : List or tuple of floats (or None), optional
        Treatment probabilities for the random comparison allocation generated by
        :meth:`~OptimalPolicy.evaluate`.
        Supply one finite, nonnegative probability per treatment, summing to 1.
        With None, use the observed treatment shares in the evaluation data when
        the variable specified by ``'var_d_name'`` is available. Otherwise, use
        equal probabilities for all treatments.
        These are sampling probabilities; realized treatment shares may differ.
        The default is None.

    var_bb_restrict_name : String or one-element list/tuple of strings (or None), optional
        Name of a variable used to prioritize observations in an additional
        allocation under the treatment-share limits in ``'other_max_shares'``.
        Observations are processed in descending order of this variable. Each is
        assigned the treatment with the highest policy score after subtracting
        treatment costs among those with remaining capacity.
        Used by ``'best_policy_score'`` and to construct training labels for an
        additional classifier under ``'bps_classifier'``.
        This additional allocation is created only if the variable is available
        in the data and treatment shares are restricted.
        The default is None.

    var_d_name : String, list or tuple of strings (or None), optional
        Name of the observed treatment variable, given as a string or a one-element list or tuple.
        When present in evaluation data, it enables comparisons with the observed allocation and
        separate results for observations whose treatment changes.
        In training and evaluation data, values must be finite integer codes from 0 to K - 1,
        where K is the number of policy scores. Codes follow the order of ``'var_polscore_name'``.
        Default is None.

    var_id_name : String, list or tuple of strings (or None), optional
        Name of the identifier in the data, supplied as a string or a one-element list or tuple.
        Default is None.

    var_polscore_desc_name : String, list or tuple (or None), optional
        Names of additional treatment-specific scores used by :meth:`~OptimalPolicy.evaluate`.
        Supply a flat sequence of names or nested lists/tuples, which are flattened.
        Names are converted to lowercase and deduplicated before forming consecutive blocks of
        K names, where K is the number of treatments. Each block defines one additional score set
        and must follow the treatment order in 'var_polscore_name'.
        A block is evaluated only if it contains K names and all its columns exist in the
        evaluation data. Incomplete blocks and blocks with missing columns are skipped.
        Default is None.

    var_polscore_name : String, list or tuple of strings (or None), optional
        Names of treatment-specific policy scores, usually estimated potential outcomes.
        Their order defines treatment codes 0 to K - 1, where K is the number of scores.
        Names must be supplied when creating the instance, before calling any method.
        Default is None, but omitting the names or passing None currently raises TypeError during
        initialization.

    var_polscore_se_name : String, list or tuple of strings (or None), optional
        Names of columns containing the standard errors of the policy scores.
        Supply one name per entry in ``'var_polscore_name'``, in the same order.
        Required by :meth:`~OptimalPolicy.estrisk_adjust`, which computes
        'policy_score - estrisk_value * standard_error'.
        Standard errors must be numeric, finite, and nonnegative.
        After risk adjustment, :meth:`~OptimalPolicy.allocate` with
        ``'gen_method'`` set to ``'best_policy_score'`` also uses these columns
        if the prediction data do not already contain the adjusted score columns.
        The default is None.

    var_material_name_ord : String, list or tuple of strings (or None), optional
        Materially relevant ordered variables used to condition fairness adjustments. Effects of the
        protected variables captured by these variables are allowed. 'fair_adjust_target' selects
        whether policy scores, decision variables or both are adjusted. These variables may also be
        decision variables, but must not be included among the protected variables.
        Default is None.

    var_material_name_unord : String, list or tuple of strings (or None), optional
        Materially relevant unordered variables used to condition fairness adjustments. Effects of
        the protected variables captured by these variables are allowed. 'fair_adjust_target'
        selects whether policy scores, decision variables or both are adjusted. These variables may
        also be decision variables, but must not be included among the protected variables.
        Default is None.

    var_protected_name_ord : String, list or tuple of strings (or None), optional
        Names of protected ordered variables used in fairness adjustments of policy scores, decision
        variables or both, as selected by 'fair_adjust_target'. Adjustments are conditional on the
        materially relevant variables. If included in ``'var_x_name_ord'``, these variables are
        removed from that list.
        Default is None.

    var_protected_name_unord : String, list or tuple of strings (or None), optional
        Names of protected unordered variables used in fairness adjustments of policy scores,
        decision variables or both, as selected by 'fair_adjust_target'. Adjustments are conditional
        on the materially relevant variables. If included in ``'var_x_name_unord'``, these variables
        are removed from that list.
        Default is None.

    var_vi_x_name : String, list or tuple of strings or None, optional
        Names of variables for which variable importance is computed.
        Default is None.

    var_vi_to_dummy_name : String, list or tuple of strings or None, optional
        Names of variables for which variable importance is computed. These variables will be broken
        up into dummies.
        Default is None.

    var_x_name_ord : String, list or tuple of strings (or None), optional
        Names of ordered decision variables, including dummy variables, used to build policy trees
        or allocation classifiers. They are also used to describe allocations.
        For 'policy_tree', 'policy tree old' and 'bps_classifier', at least one decision variable
        must be supplied through this parameter or 'var_x_name_unord'.
        Default is None, which supplies no ordered decision variables.

    var_x_name_unord : String, list or tuple of strings (or None), optional
        Names of unordered decision variables used to build policy trees or allocation classifiers.
        They are also used to describe allocations.
        For 'policy_tree', 'policy tree old' and 'bps_classifier', use integer-coded categories with
        at least three distinct values in the training data retained after initial cleaning.
        Specify binary variables through 'var_x_name_ord' instead.
        The requirement for at least one decision variable is described under 'var_x_name_ord'.
        Default is None, which supplies no unordered decision variables.

    _int_dpi : Integer or float (or None), optional
        Requested resolution in dots per inch for saved figures. Numeric values are rounded to
        integers; None and values below 10 use 500.
        Policy-tree JPEGs may use a lower resolution to limit image dimensions and pixel count.
        Policy-tree PDFs remain vector graphics.
        Default (or None) is 500.
        Internal variable, change default only if you know what you do.
        
    _int_fontsize : Integer, float or string (or None), optional
        Font size for plot legends. Numeric values strictly between 0.5 and 7.5 are rounded
        to an index from 1 to 7. None and numeric values outside this range use index 2.
        Indices 1 to 7 correspond to ``'xx-small'``, ``'x-small'``, ``'small'``, ``'medium'``,
        ``'large'``, ``'x-large'`` and ``'xx-large'``. These strings can also be supplied directly.
        Other strings raise ValueError.
        Default (or None) is 2 (``'x-small'``).
        Internal variable, change default only if you know what you do.

    _int_output_no_new_dir : Boolean (or None), optional
        Reuse an existing output directory when True, even if it is nonempty.
        With False or None, reuse an empty directory; for a nonempty one,
        search for an unused or empty directory with a numeric suffix.
        Missing directories are created in either case.
        Only relevant when output is enabled.
        During initialization, existing text output and summary files with
        the configured names are deleted from the selected directory.
        Default (or None) is False.
        Internal variable, change default only if you know what you do.

    _int_show_plots : Boolean (or None), optional
        Show policy-tree and Qini figures interactively.
        If False, suppress interactive display while retaining figure files and policy-tree figures
        in PDF reporting. Only used when _int_with_output is True.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_with_numba : Boolean (or None), optional
        Use Numba to speed up computations.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_with_output : Boolean (or None), optional
        Enable standard text output and output-dependent diagnostics,
        figures and report information. Text output follows gen_output_type.
        False disables verbose output and leaves output paths unset during initialization.
        It also disables the variable-importance analysis requested by gen_variable_importance.
        evaluate_multiple() requires this option to be enabled.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_xtr_parallel : Boolean (or None), optional
        Divide each ordered feature's candidate values for the first split into up to four tasks.
        False uses one task per feature. Unordered features use one task in either case.
        Used for ``'gen_method'`` == ``'policy_tree'`` when ``'gen_mp_parallel'`` exceeds 1.
        Works with the legacy Ray implementation and with the configured execution backend.
        Default (or None) is True.
        Internal variable, change default only if you know what you do.

    _int_mp_use_old_ray : Boolean (or None), optional
        Use old implementation of ray.
        Default is False.
        Internal variable, change default only if you know what you do.

    _int_mp_backend : String (or None), optional
        Backend used for policy-tree parallelization. Available values are
        ``'ray'``, ``'joblib'`` and ``'sequential'``.
        Only used when ``'_int_mp_use_old_ray'`` is False.
        The default is None. This selects ``'joblib'`` on Windows or when Ray
        cannot be imported. Otherwise, None selects ``'ray'``.
        Internal variable, change default only if you know what you do.

    _int_mp_batches : Integer, float or string (or None), optional
        Requested maximum number of task batches for parallel policy-tree search.
        Numeric values must be at least 1; floats are truncated to integers.
        ``'automatic'`` or None selects batch sizes from the number of tasks,
        workers and the backend. Other strings are not accepted.
        Minimum batch sizes depend on the worker count and backend, so the
        actual number of batches can be smaller than requested.
        Only used when ``'_int_mp_use_old_ray'`` is False.
        The default is ``'automatic'``.
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

    Attributes
    ----------
    __version__ : String
        Version of mcf module used to create the instance.

    <NOT-ON-API>

    dc_cfg : DataCleanCfg dataclass
        Parameters used in data cleaning.

    estriskcfg : EstRiskCfg dataclass
        Parameters used to account for estimation uncertainty in policy scores.

    fair_cfg : FairCfg dataclass
        Parameters used in fairness adjustment of policy scores and decision variables.

    gen_cfg : GenCfg dataclass
        General parameters used in various parts of the programme.

    int_cfg : IntCfg dataclass
        Parameters used in many parts of the class.

    number_scores : Integer
        Number of policy scores.

    other_cfg : OtherCfg dataclass
        Contains other relevant information needed for allocation (like cost, constraints).

    pt_cfg : PtCfg dataclass
        Parameters used to build policy tree.

    rnd_cfg : RndCfg dataclass
        Shares for random allocation.

    time_strings : Dictionary
        Maps computation-step labels to stored timing summaries.

    var_cfg : VarCfg dataclass
        Variable names.

    var_x_type : Dictionary
        Types of covariates (internal).

    var_x_values : Dictionary
        Values of covariates (internal).

    </NOT-ON-API>
    
    """

    def __init__(self, *,
                 dc_check_perfectcorr=True, dc_min_dummy_obs=10,
                 estrisk_value=1,
                 fair_adjust_target=None, fair_consistency_test=False,
                 fair_cont_min_values=20, fair_material_disc_method='Kmeans',
                 fair_material_max_groups=5, fair_regression_method='RandomForest',
                 fair_protected_disc_method='Kmeans', fair_protected_max_groups=5,
                 fair_type='Quantiled',
                 fs_rel_vi_threshold=0, fs_other_sample=True, fs_other_sample_share=0.33,
                 fs_yes=False,
                 gen_method='best_policy_score', gen_mp_parallel='None',
                 gen_outfiletext='txtFileWithOutput', gen_outpath=None, gen_output_type=2,
                 gen_variable_importance=True,
                 other_costs_of_treat=None, other_costs_of_treat_mult=None, other_max_shares=None,
                 pt_depth_tree_1=3, pt_depth_tree_2=1, pt_enforce_restriction=False,
                 pt_eva_cat_mult=1, pt_no_of_evalupoints=100, pt_min_leaf_size=None,
                 pt_select_values_cat=False,
                 rnd_shares=None,
                 var_bb_restrict_name=None, var_d_name=None,
                 var_id_name=None, var_material_name_ord=None,
                 var_material_name_unord=None, var_polscore_desc_name=None, var_polscore_name=None,
                 var_polscore_se_name=None, var_protected_name_ord=None,
                 var_protected_name_unord=None, var_vi_x_name=None, var_vi_to_dummy_name=None,
                 var_x_name_ord=None, var_x_name_unord=None,
                _int_dpi=500, _int_fontsize=2, _int_output_no_new_dir=False,
                _int_with_numba=True, _int_with_output=True, _int_xtr_parallel=True,
                _int_mp_use_old_ray=False,
                _int_mp_backend=None, _int_mp_batches='automatic',
                _int_mp_memmap_min_bytes=4*1024*1024,
                _int_mp_memmap_dir=Path.cwd() / 'joblibtemp',
                _int_show_plots: bool | None = True,
                ):
        self.__version__ = '0.11.0'

        self.int_cfg = op_init.IntCfg.from_args(cuda=False,
                                                output_no_new_dir=_int_output_no_new_dir,
                                                with_numba=_int_with_numba,
                                                with_output=_int_with_output,
                                                show_plots=_int_show_plots,
                                                xtr_parallel=_int_xtr_parallel,
                                                dpi=_int_dpi, fontsize=_int_fontsize,
                                                mp_use_old_ray=_int_mp_use_old_ray,
                                                mp_backend=_int_mp_backend,
                                                mp_batches=_int_mp_batches,
                                                mp_memmap_min_bytes=_int_mp_memmap_min_bytes,
                                                mp_memmap_dir=_int_mp_memmap_dir,
                                                )
        self.gen_cfg = op_init.GenCfg.from_args(method=gen_method, mp_parallel=gen_mp_parallel,
                                                outfiletext=gen_outfiletext,
                                                outpath=gen_outpath, output_type=gen_output_type,
                                                variable_importance=gen_variable_importance,
                                                with_output=self.int_cfg.with_output,
                                                new_outpath=not self.int_cfg.output_no_new_dir,
                                                )
        self.dc_cfg = op_init.DataCleanCfg.from_args(check_perfectcorr=dc_check_perfectcorr,
                                                     min_dummy_obs=dc_min_dummy_obs,
                                                     )
        self.pt_cfg = op_init.PtCfg.from_args(depth_tree_1=pt_depth_tree_1,
                                              depth_tree_2=pt_depth_tree_2,
                                              eva_cat_mult=pt_eva_cat_mult,
                                              enforce_restriction=pt_enforce_restriction,
                                              no_of_evalupoints=pt_no_of_evalupoints,
                                              select_values_cat=pt_select_values_cat,
                                              min_leaf_size=pt_min_leaf_size,
                                              )
        self.other_cfg = op_init.OtherCfg.from_args(
            other_costs_of_treat=other_costs_of_treat,
            other_costs_of_treat_mult=other_costs_of_treat_mult,
            other_max_shares=other_max_shares,
            )
        self.rnd_cfg = op_init.RndCfg.from_args(rnd_shares=rnd_shares)

        self.var_cfg = op_init.VarCfg.from_args(bb_restrict_name=var_bb_restrict_name,
                                                d_name=var_d_name,
                                                id_name=var_id_name,
                                                polscore_desc_name=var_polscore_desc_name,
                                                material_ord_name=var_material_name_ord,
                                                material_unord_name=var_material_name_unord,
                                                polscore_name=var_polscore_name,
                                                polscore_se_name=var_polscore_se_name,
                                                protected_ord_name=var_protected_name_ord,
                                                protected_unord_name=var_protected_name_unord,
                                                x_ord_name=var_x_name_ord,
                                                x_unord_name=var_x_name_unord,
                                                vi_x_name=var_vi_x_name,
                                                vi_to_dummy_name=var_vi_to_dummy_name,
                                                )
        self.fair_cfg = op_init.FairCfg.from_args(self.gen_cfg,
                                                  adjust_target=fair_adjust_target,
                                                  consistency_test=fair_consistency_test,
                                                  cont_min_values=fair_cont_min_values,
                                                  material_disc_method=fair_material_disc_method,
                                                  protected_disc_method=fair_protected_disc_method,
                                                  material_max_groups=fair_material_max_groups,
                                                  regression_method=fair_regression_method,
                                                  protected_max_groups=fair_protected_max_groups,
                                                  adj_type=fair_type,
                                                  )
        self.fs_cfg = FsCfg.from_args(rel_vi_threshold=fs_rel_vi_threshold,
                                      other_sample=fs_other_sample,
                                      other_sample_share=fs_other_sample_share,
                                      yes=fs_yes
                                      )
        self.estriskcfg = op_init.EstRiskCfg.from_args(value=estrisk_value)

        self.time_strings, self.var_x_type, self.var_x_values = {}, {}, {}
        self.bps_class_dict = {}
        self.report = {'fairscores': False,
                       'solvefair': False,
                       'training': False,
                       'evaluation': False,
                       'allocation': False,
                       'estriskscores': False,
                       'opt_versions': False,
                       'training_data_chcksm': 0,   # To identify training data
                       'training_alloc_chcksm': 0,  # To identify train. alloc.
                       'alloc_list': [],            # List because of possible multiple
                       'evalu_list': [],            # allocation, evaluation methods
                       }                   # might be used multiple times.
        self.number_scores = len(self.var_cfg.polscore_name)

    def allocate(self, data_df, data_title='', fair_adjust_decision_vars=False):
        """
        Allocate observations to treatment state.

        For 'policy_tree', 'policy tree old' and 'bps_classifier', first run solve() or solvefair()
        on this instance to fit the allocation rule. With 'best_policy_score', allocate() computes
        allocations directly from the supplied policy scores and does not require prior training.

        Parameters
        ----------
        data_df : DataFrame
            Input data with at least features or policy scores
            (depending on algorithm).

        data_title : String, optional
            This string is used as title in outputs. The default is ''.

        fair_adjust_decision_vars : Boolean, optional
            Controls recomputation of decision-variable fairness adjustments.
            Use True only after :meth:`~OptimalPolicy.solvefair` with
            ``'fair_adjust_target'`` set to ``'xvariables'`` or ``'scores_xvariables'``.
            True recomputes adjusted variables and replaces existing columns with those names.
            With False, available adjusted variables are reused; missing adjusted decision variables
            required by the trained rule are computed automatically. Best-policy-score allocation
            does not require decision variables, so False does not trigger their adjustment.
            If all current policy-score columns are supplied, earlier risk and fairness inputs are
            unnecessary for best-policy-score allocation unless True requests variable adjustment.
            On an instance that has not used :meth:`~OptimalPolicy.solvefair`,
            this option has no effect.
            The default is False.

        Returns
        -------
        results : Dictionary.
            Contains the results. This dictionary has the following structure:
            'allocation_df' : DataFrame
                Treatment allocations, with one row per observation used for allocation
                and one column per allocation rule. Entries are treatment codes.
                Column names depend on the allocation method and restrictions.
                The input features and policy scores are not included.
            'outpath' : Path or None
                Directory in which output is saved.
                None if _int_with_output is False.

        """
        allocation_df, self.gen_cfg.outpath = op_methods.allocate_method(
            self, data_df, data_title=data_title,
            fair_adjust_decision_vars=fair_adjust_decision_vars,
            )
        results_dic = {'allocation_df': allocation_df, 'outpath': self.gen_cfg.outpath}

        return results_dic

    def evaluate(self, allocation_df, data_df, data_title='', seed=12434):
        """
        Evaluate allocation with potential outcome data.

        Run :meth:`~OptimalPolicy.solve` or :meth:`~OptimalPolicy.solvefair` on this instance first.
        With 'gen_method' == 'best_policy_score', :meth:`~OptimalPolicy.allocate` also initializes
        the instance for evaluation.

        Parameters
        ----------
        allocation_df : DataFrame
            Optimal allocation as outputed by the
            :meth:`~OptimalPolicy.solve`, :meth:`~OptimalPolicy.solvefair`,
            and :meth:`~OptimalPolicy.allocate` methods.
            Names 'random', 'observed' and 'best ATE' are used for automatic reference allocations.
            A ValueError is raised if a supplied column name conflicts with a reference generated
            for this call.

        data_df : DataFrame
            Evaluation variables for the observations in ``allocation_df``. Both DataFrames must
            have the same number of rows in the same observation order. Rows are paired by
            position; index labels and identifier columns are not used to align them.
            Welfare uses the complete set of current 'var_polscore_name' columns when available.
            After fairness or estimation-risk adjustment, missing current scores trigger a fallback
            to the last complete available treatment block in 'var_polscore_desc_name'.
            If neither is available, primary welfare measures are omitted; allocation shares and
            available descriptive statistics are still reported.
            After solvefair(), also include every variable in 'var_protected_name_ord' and
            'var_protected_name_unord'. These columns are required for fairness diagnostics, even
            when '_int_with_output' is False. Pearson correlations are reported only for numeric
            protected columns; nonnumeric categories receive NaN for this measure. Categorical
            dependence measures are still computed when the distinct-value threshold permits them.

        data_title : String, optional
            This string is used as title in outputs. The default is ''.

        seed : Integer, optional
            Seed for random number generators. The default is 12434.

        Returns
        -------
        results_all_dic : Dictionary
            'results_dic': Dictionary with one entry per population and allocation rule.
                Keys are 'All <allocation>' and, when observed treatments are available,
                'Switchers <allocation>'. Switchers receive a different treatment from their
                observed treatment; this subset is specific to each allocation rule.
                Reference allocations are added automatically: 'random', 'observed' when available,
                and 'best ATE' when welfare scores are available. 'best ATE' assigns everyone the
                treatment with the highest mean policy score minus its treatment cost.
                Each entry contains 'treatment share' and 'number of observations'. When welfare
                scores are available, gross mean welfare is stored under the same key as the outer
                entry, and net mean welfare under 'welfare net of costs'. Additional score summaries
                and fairness diagnostics are included when applicable.
            'outpath' : Path or None
                Directory in which output is saved.
                None if _int_with_output is False.

        """
        results_dic, self.gen_cfg.outpath = op_methods.evaluate_method(
            self, allocation_df, data_df, data_title=data_title, seed=seed
            )
        results_all_dic = {'results_dic': results_dic, 'outpath': self.gen_cfg.outpath}

        return results_all_dic

    def evaluate_multiple(self, allocations_dic, data_df):
        """
        Evaluate several allocations simultaneously and write comparison output.

        Requires '_int_with_output' to be True. Otherwise, raises ValueError before evaluation.

        Run :meth:`~OptimalPolicy.solve` or :meth:`~OptimalPolicy.solvefair` on this instance first.
        With 'gen_method' == 'best_policy_score', :meth:`~OptimalPolicy.allocate` also initializes
        the instance for evaluation.

        Parameters
        ----------
        allocations_dic : Dictionary
            Maps allocation names (strings) to one-column DataFrames or Series of treatment codes.
            Each value describes one allocation and must contain one entry per row of ``data_df``.
            Codes must be finite integers from 0 to K - 1, in the order of ``'var_polscore_name'``,
            where K is the number of treatments.

        data_df : DataFrame
            Policy-score data for evaluating the allocations. Every allocation must have the same
            row order as this DataFrame; rows are matched by position, without index alignment.
            Requires a complete treatment block of policy scores. Selection follows evaluate():
            use current scores, or the descriptive-score fallback after fairness or estimation-risk
            adjustment. If neither is available, this method raises ValueError.

        Returns
        -------
        results_dic : Dictionary.
            Contains the results. This dictionary has the following structure:
            'outpath' : Path
                Location of directory in which output is saved.

        """
        if not self.gen_cfg.with_output:
            raise ValueError('To use this method, allow output to be written.')

        self.gen_cfg.outpath = op_methods.evaluate_multiple_self(self, allocations_dic, data_df)
        results_dic = {'outpath': self.gen_cfg.outpath}

        return results_dic

    def estrisk_adjust(self, data_df, data_title=''):
        """
        Adjust policy scores for estimation risk and select them for subsequent policy learning.

        The instance uses the returned 'estrisk_scores_names' as its active policy-score names.
        Pass the returned 'data_estrisk_df' to solve() or solvefair() to use these adjusted scores.
        For 'gen_method' == 'best_policy_score', allocate() also creates adjusted scores when they
        are absent, using the original scores and their standard errors.
        Repeated calls use the original score names saved by the first successful call and replace
        existing '_estrisk' columns. They do not subtract the risk adjustment a second time.
        A failed call leaves the instance's risk-adjustment state unchanged.

        Parameters
        ----------
        data_df : DataFrame
            Data containing the original policy scores and their standard errors in matching
            order. Values must be numeric and finite, with nonnegative standard errors.
            Names formed by adding '_estrisk' to score names must not coincide with any original
            configured input variable name; conflicting names raise ValueError. Rename the input
            column and its keyword entry. Repeated calls may replace previously generated scores.
            Column names are converted to lowercase in the returned data.

        data_title : String, optional
            This string is used as title in outputs. The default is ''.

        Returns
        -------
        results_dic : Dictionary.
            Contains the results. This dictionary has the following structure:
            'data_estrisk_df' : DataFrame
                Input data with additional policy scores adjusted for estimation risk:
                policy_score - estrisk_value * standard_error.
                The added column names have the suffix '_estrisk'.
                The row index is reset to consecutive integers starting at zero.
            'estrisk_scores_names' : List of strings.
                Names of adjusted scores.
            'outpath' : Path or None
                Directory in which output is saved.
                None if _int_with_output is False.

        """
        (data_estrisk_df, estrisk_scores_names, self.gen_cfg.outpath
         ) = op_methods.estrisk_adjust_method(self, data_df, data_title=data_title)
        results_dic = {'data_estrisk_df': data_estrisk_df,
                       'estrisk_scores_names': estrisk_scores_names,
                       'outpath': self.gen_cfg.outpath
                       }
        return results_dic

    def solvefair(self, data_df, data_title=''):
        """
        Solve for optimal allocation rule with fairness adjustments.

        Follows the suggestions of Bearth, Lechner, Muny, Mareckova (2025, arXiV).
        It accepts the same arguments as :meth:`~OptimalPolicy.solve`.
        The return value is a single dictionary containing 'allocation_df',
        'result_dic', and 'outpath'.
        Repeated calls use the variable names and adjustment settings saved on the first call.
        Supply the corresponding input columns again; existing fairness-adjusted columns are
        replaced rather than adjusted a second time. The latest call replaces the cached training
        data used for decision-variable adjustment during allocate(). If estrisk_adjust() was used
        before the first call, these input policy scores are the risk-adjusted scores.
        Generated column names must not coincide with another configured input variable name.
        Such conflicts raise ValueError; rename the conflicting input column and its keyword entry.
        When both scores and decision variables are adjusted, use separately named input columns for
        these roles; sharing a name would also share the adjusted column and raises ValueError.


        Parameters
        ----------
        data_df : DataFrame
            Input data to train particular allocation algorithm.
        data_title : String, optional
            This string is used as title in outputs. The default is ''.

        Returns
        -------
        results_all_dict : Dictionary.
            Contains the results. This dictionary has the following structure:
            'allocation_df' : DataFrame
                Treatment allocations, with one row per observation used for allocation
                and one column per allocation rule. Entries are treatment codes.
                Column names depend on the allocation method and restrictions.
                The input features and policy scores are not included.
            'result_dic' : Dictionary
                Contains additional information about trained allocation rule.
                Only complete when keyword _int_with_output is True.
            'outpath' : Path or None
                Directory in which output is saved.
                None if _int_with_output is False.

        """
        (allocation_df, result_dic, self.gen_cfg.outpath
         ) = op_methods.solvefair_method(self, data_df, data_title=data_title)
        results_all_dic = {'allocation_df': allocation_df,
                           'result_dic': result_dic,
                           'outpath': self.gen_cfg.outpath
                           }
        return results_all_dic

    def solve(self, data_df, data_title=''):
        """
        Solve for optimal allocation rule.
        Training preparation keeps the required columns and removes rows with missing
        values in those columns. This cleaning is always performed.
        Covariate screening runs automatically for 'policy_tree', 'policy tree old'
        and 'bps_classifier'. It is skipped for 'best_policy_score'.        

        Parameters
        ----------
        data_df : DataFrame
            Input data to train particular allocation algorithm.
        data_title : String, optional
            This string is used as title in outputs. The default is ''.

        Returns
        -------
        results_all_dict : Dictionary.
            Contains the results. This dictionary has the following structure:
            'allocation_df' : DataFrame
                Treatment allocations, with one row per observation used for allocation
                and one column per allocation rule. Entries are treatment codes.
                Column names depend on the allocation method and restrictions.
                The input features and policy scores are not included.
            'result_dic' : Dictionary
                Contains additional information about trained allocation rule.
                Only complete when keyword _int_with_output is True.
            'outpath' : Path or None
                Directory in which output is saved.
                None if _int_with_output is False.
        data_df : DataFrame
            Second item of the returned tuple. Contains the input data for the observations used to
            train the allocation rule. Rows removed during preparation or reserved for feature
            selection are excluded. Rows align with 'allocation_df', and the index is reset to
            consecutive integers starting at zero.
            Input columns are retained, with names converted to lowercase. An identifier column may
            be added during preparation.

        """
        allocation_df, result_dic, self.gen_cfg.outpath, data_df = op_methods.solve_method(
            self, data_df, data_title=data_title
            )
        results_all_dic = {'allocation_df': allocation_df,
                           'result_dic': result_dic,
                           'outpath': self.gen_cfg.outpath
                           }
        return results_all_dic, data_df

    def print_time_strings_all_steps(self, title='', line_length=100):
        """Print the computation-time summaries currently stored on this instance.

        Parameters
        ----------
        title : String, optional
            Text appended to the summary heading. Default is ''.
        line_length : Integer, optional
            Number of '=' characters in the heading's separator line. It does not wrap the text.
            Default is 100.

        Returns
        -------
        None
        """
        txt = '\n' + '=' * line_length + '\nSummary of computation times of all steps '
        txt += title
        print_mcf(self.gen_cfg, txt, summary=True)
        val_all = ''
        for _, val in self.time_strings.items():
            val_all += val
        print_mcf(self.gen_cfg, val_all, summary=True)

    def winners_losers(self: 'OptimalPolicy',
                       data_df: DataFrame,
                       welfare_df: DataFrame, *,
                       welfare_reference_df: DataFrame | None = None,
                       outpath: str | Path | None = None,
                       title: str = '',
                       ) -> dict[str, DataFrame | Path | None]:
        """
        Compare the winners and loser.

        Cluster observations by their welfare change relative to the reference allocation.
        Only welfare changes enter the clustering; policy scores and decision, protected and
        materially relevant variables are used to describe the resulting groups.
        Actual and reference welfare, and their differences, must be numeric and finite.
        Invalid values raise ValueError before a cluster-label column is added to data_df.
        For varying welfare changes, try 2 through min(8, n - 1, u) clusters, where n is the number
        of observations and u the number of distinct changes. Merge clusters containing fewer than
        1% of observations with their nearest remaining cluster, then select the candidate with the
        highest valid average silhouette score. The retained number of clusters can be smaller.
        Varying changes require at least three observations. A ValueError is raised if no candidate
        clustering has a valid silhouette score.

        Parameters
        ----------
        data_df : Dataframe
            Variables used for descriptions.

        welfare_df : DataFrame
            Welfare under the allocation being evaluated. Must contain exactly one
            column and the same number of observations as ``data_df``, in the same
            row order. Rows are matched by position.

        welfare_reference_df : DataFrame (or None), optional
            Welfare under the reference allocation. If supplied, must contain exactly
            one column and match the number and row order of observations in
            ``data_df`` and ``welfare_df``.
            The default is None, which uses zero reference welfare for every observation.

        outpath : String or Path (or None), optional
            Directory for this call's text output. An explicit directory
            is created if needed and used with the configured text-file
            basenames.
            None reuses the instance's configured text-file paths.
            The default is None.

        title : String, optional
            Title used in the statistics. The default is ''.

        Returns
        -------
        results_dict : Dictionary.
            Contains the results. This dictionary has the following structure:
            'data_plus_cluster_number_df' : DataFrame
                ``data_df`` with a cluster-label column named
                ``'Welfare_change_cluster_<welfare>_minus_<reference>'``.
                ``<welfare>`` and ``<reference>`` are the column names from
                ``welfare_df`` and ``welfare_reference_df``; the reference name
                is ``'zero'`` when ``welfare_reference_df`` is None.
                Labels start at 0 and increase with the cluster's mean welfare change.
                If welfare changes are constant, every observation receives label 0.
                The supplied ``data_df`` is modified in place.
            'outpath' : Path or None
                Supplied directory converted to Path, or the existing
                self.gen_cfg.outpath when outpath is None.
                Assigned to self.gen_cfg.outpath on successful return.
                Can be None when welfare changes do not vary and no
                output directory is configured.

        Notes
        -----
        Clustering descriptions are written to both text output files,
        even when gen_output_type is 0. Initialize with _int_with_output=True
        to configure the required text-file paths. An explicit outpath
        also needs their configured filenames.
        If welfare changes do not vary, the method returns before writing
        text. With outpath=None, this case can also run with output disabled.

        """
        data_plus_cluster_number_df, self.gen_cfg.outpath = op_methods.winners_losers_method(
             self, data_df, welfare_df,
             welfare_reference_df=welfare_reference_df, outpath=outpath, title=title,
             )
        results_dic = {'data_plus_cluster_number_df': data_plus_cluster_number_df,
                       'outpath': self.gen_cfg.outpath
                       }
        return results_dic


class OptimalPolicyVersions:
    """
    Optimal policy learning when there are versions in some of the main treatments.

    Run :meth:`~OptimalPolicyVersions.solve` on this instance before using
    :meth:`~OptimalPolicyVersions.allocate` or :meth:`~OptimalPolicyVersions.evaluate`.
    This also applies to 'best_policy_score': solve() creates the underlying OptimalPolicy model.

    Parameters
    ----------
    policyscores_dict : Dictionary
        Nonempty dictionary mapping policy score names for main treatments to the
        names of their version scores.
        Each key must be a string. Its value must be None, a string, or a nonempty
        list or tuple of strings.
        None uses the key itself as the only score for that main treatment.
        A string specifies one version score; a list or tuple specifies one or more.
        Main treatments follow dictionary order, and versions follow the order of their scores.
        Version-score names must be distinct across all main treatments, ignoring case, and must
        not be '0'. With 'policy_tree', the same rule applies separately to the main-score keys.
        A main-score key may also identify one version score. For distinct treatments with
        identical scores, use separate columns with distinct names.
        Invalid score-name combinations raise ValueError in solve().
        This argument must be provided; the dictionary itself cannot be None.

    depth_version_tree : Integer, float, list or tuple of numbers (or None), optional
        Maximum depth of the version trees for main treatments with multiple versions.
        Used when 'gen_method' in 'params_optpol' is 'policy_tree'.
        A scalar applies to all main treatments. A list or tuple must contain one
        value per main treatment, in the key order of 'policyscores_dict', including
        main treatments with only one version.
        Every supplied value must lie between 2 and 10 inclusive before rounding.
        Float values are rounded to the nearest integer.
        For each version tree, this value overrides 'pt_depth_tree_1', and
        'pt_depth_tree_2' is set to 0, so no second tree stage is fitted.
        The default (or None) is 2 for every main treatment.

    params_optpol : Dictionary (or None), optional
        Keyword arguments for the underlying OptimalPolicy instances. Unspecified settings use
        their defaults. ``'var_polscore_name'`` is set internally from ``'policyscores_dict'``.
        If supplied, ``'other_costs_of_treat'`` needs one value per main treatment, in dictionary
        order; each version inherits its main treatment's cost.
        For ``'gen_method'`` == ``'policy_tree'``, supplied ``'other_max_shares'`` and
        ``'other_costs_of_treat_mult'`` values also refer to main treatments. Version trees have
        unrestricted shares and cost multipliers of 1; their depths follow ``'depth_version_tree'``.
        With other methods, these two parameters use one value per treatment version, in the
        flattened score order of ``'policyscores_dict'``.
        For 'policy_tree', 'var_d_name' in this dictionary refers to observed main treatments and
        is ignored by version trees. Set this class's 'var_d_name' to provide the observed main
        treatment/version pair for final evaluation.
        ``'fs_yes'`` set to True raises NotImplementedError with ``'policy_tree'``.
        For 'policy_tree', final treatment-version evaluation ignores supplied 'rnd_shares'.
        Its random reference allocation uses observed treatment-version shares when available;
        otherwise, it uses equal probabilities across all treatment versions.
        Default is None, which uses ``'best_policy_score'`` with the applicable default settings.

    var_d_name : List or tuple of two strings (or None), optional
        Names of the columns containing observed main treatments and their versions,
        in that order. Used by :meth:`~OptimalPolicyVersions.evaluate` when both
        columns are available in the evaluation data.
        Main treatments must be coded 0, 1, ... in the key order of
        ``'policyscores_dict'``. Versions must be coded 0, 1, ... separately
        within each main treatment, following the order of its policy-score names.
        For a main treatment with a single version, the version code must be 0.
        Values must be finite integers. Each pair is converted internally to a
        single treatment-version code for evaluating the observed allocation.
        With 'policy_tree' and any main treatment having multiple versions, both columns are
        required for this comparison. A main-treatment column supplied through 'params_optpol'
        is not sufficient.
        The default is None.


    Attributes
    ----------
    __version__ : String
        Version of mcf module used to create the instance.

    <NOT-ON-API>

    version_cfg : VersionCfg dataclass
        Parameters used in version estimation.

    optp: Instance of OptimalPolicy class or list of Instances of OptimalPolicy class
        (many instances for policy trees, otherwise only one instance)
        
    </NOT-ON-API>
    
    """

    def __init__(self, *,
                 depth_version_tree=2,
                 params_optpol=None, policyscores_dict=None,
                 var_d_name=None,
                 ):
        self.__version__ = '0.11.0'
        self.version_cfg = op_init.VersionCfg.from_args(policyscores_dict=policyscores_dict,
                                                        depth_version_tree=depth_version_tree,
                                                        params_optpol=params_optpol,
                                                        d_name=var_d_name,
                                                        )
        self.policy_scores_all = self.optp = self.gen_cfg_print = None
        self.time_strings = {}
        self.report = {'opt_versions': True}

    def allocate(self: 'OptimalPolicyVersions',
                 data_df: DataFrame,
                 data_title: str = '',
                 ) -> dict[str, DataFrame | Path | None]:
        """
        Allocate observations to treatment state.

        Parameters
        ----------
        data_df : DataFrame
            Input data with at least features or policy scores
            (depending on algorithm).

        data_title : String, optional
            This string is used as title in outputs. The default is ''.

        Returns
        -------
        results : Dictionary.
            Contains the results. This dictionary has the following structure:
            'allocation_df' : DataFrame
                Allocations to treatment versions, with one row per observation used for
                allocation and one column per allocation rule.
                Entries are integer codes starting at zero. Main treatments follow the
                order of keys in 'policyscores_dict'; their versions follow the order of
                the corresponding score names. Codes run consecutively across all main
                treatments. A main treatment without separate versions occupies one code.
                The input features and policy scores are not included.
            'outpath' : Path or None
                Directory in which output is saved.
                None if params_optpol sets _int_with_output to False.

        """
        return op_version.allocate_version(self,
                                           data_df,
                                           data_title=data_title,
                                           )

    def evaluate(self, allocation_df, data_df, data_title='', seed=12434):
        """
        Evaluate allocation with potential outcome data.

        Parameters
        ----------
        allocation_df : DataFrame
            Treatment-version allocations, for example the 'allocation_df' returned by
            :meth:`~OptimalPolicyVersions.solve` or :meth:`~OptimalPolicyVersions.allocate`.
            Each column represents one allocation rule. Codes run from 0 to K - 1, where K counts
            all treatment versions. Main treatments follow the order in 'policyscores_dict';
            within each main treatment, follow the order of its version-score names.
            A main treatment without separate versions occupies one code.
            Names 'random', 'observed' and 'best ATE' are used for automatic reference allocations.
            A ValueError is raised if a supplied column name conflicts with a reference generated
            for this call.

        data_df : DataFrame
            Evaluation variables for the observations in ``allocation_df``. Both DataFrames must
            have the same number of rows in the same observation order. Rows are paired by
            position; index labels and identifier columns are not used to align them.

        data_title : String, optional
            This string is used as title in outputs. The default is ''.

        seed : Integer, optional
            Seed for random number generators. The default is 12434.

        Returns
        -------
        results_all_dic : Dictionary
            'results_dic': Dictionary with one entry per population and allocation rule.
                Keys are 'All <allocation>' and, when observed treatments are available,
                'Switchers <allocation>'. Switchers receive a different treatment from their
                observed treatment; this subset is specific to each allocation rule.
                Reference allocations are added automatically: 'random', 'observed' when available,
                and 'best ATE' when welfare scores are available. 'best ATE' assigns everyone the
                treatment with the highest mean policy score minus its treatment cost.
                Each entry contains 'treatment share' and 'number of observations'. When welfare
                scores are available, gross mean welfare is stored under the same key as the outer
                entry, and net mean welfare under 'welfare net of costs'. Additional score summaries
                and fairness diagnostics are included when applicable.
            'outpath' : Path or None
                Directory in which output is saved.
                None if params_optpol sets _int_with_output to False.

        """
        return op_version.evaluate_version(self,
                                           allocation_df,
                                           data_df,
                                           data_title=data_title,
                                           seed=seed,
                                           )

    def solve(self, data_df, data_title=''):
        """
        Solve for optimal allocation rule.

        Parameters
        ----------
        data_df : DataFrame
            Input data to train particular allocation algorithm.
        data_title : String, optional
            This string is used as title in outputs. The default is ''.

        Returns
        -------
        results_all_dict : Dictionary.
            Contains the results. This dictionary has the following structure:
            'allocation_df' : DataFrame
                Allocations to treatment versions, with one row per observation used for
                allocation and one column per allocation rule.
                Entries are integer codes starting at zero. Main treatments follow the
                order of keys in 'policyscores_dict'; their versions follow the order of
                the corresponding score names. Codes run consecutively across all main
                treatments. A main treatment without separate versions occupies one code.
                The input features and policy scores are not included.
            'result_dic' : Dictionary or list
                Additional information about the fitted allocation rules.
                For 'policy_tree', a list with the 'result_dic' of the main tree first,
                followed by one entry per main treatment, in 'policyscores_dict' order.
                Each subsequent entry is the 'result_dic' of its version tree, or None
                if no version tree was fitted.
                For other methods, the underlying OptimalPolicy instance's
                'result_dic' dictionary.
                Contents depend on the allocation method and output settings.
            'outpath' : Path or None
                Directory in which output is saved.
                None if params_optpol sets _int_with_output to False.

        data_df : DataFrame
            Second item of the returned tuple. Contains the retained training observations, in the
            same row order as 'allocation_df', with an index starting at zero.
            Rows removed during preparation are excluded. For 'policy_tree', this includes removals
            in version trees. For other methods, any separate feature-selection sample is also
            excluded.
            Input columns are retained, with names converted to lowercase. An identifier column may
            be added during preparation.

        """
        start_time = time()

        (policy_scores_main, self.policy_scores_all, policy_scores_main_version, txt_descr
         ) = op_version.analyse_treatment_version(self.version_cfg.policyscores_dict)

        if self.version_cfg.params_optpol.get('gen_method') == 'policy_tree':
            if self.version_cfg.params_optpol.get('fs_yes') is True:
                raise NotImplementedError('Automatic  Feature selection not (yet?) implemented for '
                                          'sequential version policy tree.'
                                          )

            # Sequential procedure:
            # Step 1: The policy tree will be build for the main treatments only
            params_optpol = op_version.solve_main_para_pt(self.version_cfg.params_optpol,
                                                          policy_scores_main,
                                                          )
            optp_main = OptimalPolicy(**params_optpol)
            gen_cfg_print = deepcopy(optp_main.gen_cfg)  # Output locations are needed below
            if gen_cfg_print.with_output:
                txt_descr += op_version.tree_main_header(line_length=100)
                print_mcf(gen_cfg_print, txt_descr, summary=True)
            results_all_dic_main, data_df = optp_main.solve(data_df, data_title=data_title)
            # Step 2: If there are treatment versions, for each main treatment we build a new tree
            #         using training observations allocated to that treatment only.
            #  - Split according to main treatments (and adjust parameters accordings)
            params_v, data_v, title_v, tree_yes_v = op_version.split_main_treatment_pt_only(
                self,
                optp_main=optp_main,
                alloc_main_df=results_all_dic_main['allocation_df']['Policy Tree'],
                data_df=data_df, data_title=data_title,
                policy_scores_main_version=policy_scores_main_version,
                )
            self.optp = [optp_main]
            results_all_dic_list = [results_all_dic_main,]

            # Iterate over main treatments
            dropped_version_indices = []
            for idx, _ in enumerate(params_v):
                _ = op_version.print_title_for_version_tree(idx,
                                                            tree_yes_v[idx],
                                                            policy_scores_main_version[idx],
                                                            gen_cfg_print
                                                            )
                if tree_yes_v[idx]:
                    optp_vers = OptimalPolicy(**params_v[idx])

                    (optp_vers, old_index, data_version, row_name,
                     ) = op_version.prepare_version_for_solve(optp_vers, gen_cfg_print, data_v[idx])

                    results_all_dic_vers, data_version = optp_vers.solve(data_version,
                                                                         data_title=title_v[idx],
                                                                         )
                    kept_positions = data_version[row_name].to_numpy(dtype=np.int64, copy=False)
                    kept_old_index = old_index.take(kept_positions)

                    dropped_old_index = old_index.difference(kept_old_index)
                    if len(dropped_old_index):
                        dropped_version_indices.extend(dropped_old_index.tolist())

                    results_all_dic_vers['allocation_df'].index = kept_old_index

                else:
                    optp_vers = results_all_dic_vers = None
                results_all_dic_list.append(results_all_dic_vers)
                self.optp.append(deepcopy(optp_vers))

            results_all_dic = op_version.combine_results_all_dic(
                results_all_dic_list, policy_scores_main_version, gen_cfg=optp_main.gen_cfg,
                )
            if dropped_version_indices:
                dropped_version_indices = sorted(set(dropped_version_indices))

                results_all_dic['allocation_df'] = (results_all_dic['allocation_df']
                                                    .drop(index=dropped_version_indices)
                                                    .reset_index(drop=True)
                                                    )
                data_df = data_df.drop(index=dropped_version_indices).reset_index(drop=True)
            self.gen_cfg_print = gen_cfg_print

        else:
            # All treatment versions will be treated symmetrically
            params_optpol = deepcopy(self.version_cfg.params_optpol)
            params_optpol['var_polscore_name'] = self.policy_scores_all

            costs_main = params_optpol.get('other_costs_of_treat')
            if costs_main is not None:
                version_counts = [len(scores) for scores in policy_scores_main_version]
                if len(costs_main) != len(version_counts):
                    raise ValueError('For OptimalPolicyVersions, other_costs_of_treat must '
                                     'contain one value per main treatment.'
                                     )
                params_optpol['other_costs_of_treat'] = np.repeat(
                    np.asarray(costs_main, dtype=float),
                    version_counts,
                    ).tolist()

            self.optp = OptimalPolicy(**params_optpol)

            results_all_dic, data_df = self.optp.solve(data_df, data_title=data_title)
            if self.optp.gen_cfg.with_output:
                print_mcf(self.optp.gen_cfg, txt_descr, summary=True)

            self.gen_cfg_print = deepcopy(self.optp.gen_cfg)

        self.report['training_alloc_chcksm'] = dataframe_checksum(results_all_dic['allocation_df'])
        # Timing
        time_title = f'Solve using {self.version_cfg.params_optpol["gen_method"]}'
        key, time_str = op_version.timestr_version(self.gen_cfg_print,
                                                   start_time,
                                                   title=time_title,
                                                   data_title=data_title,
                                                   )
        self.time_strings[key] = time_str

        return results_all_dic, data_df

    def print_time_strings_all_steps(self, title='', line_length=100):
        """Print the computation-time summaries currently stored on this instance.

        Parameters
        ----------
        title : String, optional
            Text appended to the summary heading. Default is ''.
        line_length : Integer, optional
            Number of '=' characters in the heading's separator line. It does not wrap the text.
            Default is 100.

        Returns
        -------
        None
        """
        txt = '\n' + '=' * line_length + '\nSummary of computation times of all steps '
        txt += title
        print_mcf(self.gen_cfg_print, txt, summary=True)
        val_all = ''
        for _, val in self.time_strings.items():
            val_all += val
        print_mcf(self.gen_cfg_print, val_all, summary=True)
