from pathlib import Path

from pandas import DataFrame

from mcf import optpolicy_evaluation_functions as op_eval
from mcf import optpolicy_init_functions as op_init
from mcf import optpolicy_methods as op_methods
from mcf import mcf_print_stats_functions as mcf_ps


class OptimalPolicy:
    """
    Optimal policy learning

    Parameters
    ----------
    dc_screen_covariates : Boolean (or None), optional
        Check features.
        Default (or None) is True.

    dc_check_perfectcorr : Boolean (or None), optional
        Features that are perfectly correlated are deleted (1 of them).
        Only relevant if dc_screen_covariates is True.
        Default (or None) is True.

    dc_min_dummy_obs : Integer (or None), optional
        Delete dummy variables that have less than dc_min_dummy_obs in one
        of their categories. Only relevant if dc_screen_covariates is True.
        Default (or None) is 10.

    dc_clean_data : Boolean (or None), optional
        Remove all missing & unnecessary variables.
        Default (or None) is True.

    estrisk_value : Float or integer (or None), optional
        The is k in the formula  'policy_score - k * standard_error'
        used to adjust the scores for estimation risk.
        Default (or None) is 1.

    fair_adjust_target :  String (or None), optional\
        Target for the fairness adjustment.\
        ``'scores'`` : Adjust policy scores.\
        ``'xvariables'`` : Adjust decision variables.\
        ``'scores_xvariables'`` : Adjust both decision variables and score.\
        Default (or None) is 'xvariables'.

    fair_consistency_test : Boolean (or None), optional\
        Test for internally consistency of fairness correction.
        When ``'fair_adjust_target'`` is ``'scores'`` or
        ``'scores_xvariables'``, then the fairness corrections are applied
        independently to every policy score
        (which usually is a potential outcome or an IATE(x) for each
        treatment relative to some base treatment (i.e. comparing 1-0, 2-0,
        3-0, etc.). Thus the IATE for the 2-1 comparison can be computed as
        IATE(2-0)-IATE(1-0). This tests compares two ways to compute a
        fair score for the 2-1 (and all#  other comparisons) which should
        give simular results:\
        a) Difference of two fair (!) scores\
        b) Difference of corresponding scores, subsequently made fair.\
        Note: Depending on the number of treatments, this test may be
        computationally more expensive than the orginal fairness corrections.\
        Default (or None) is False.

    fair_cont_min_values : Integer or float (or None),  optional
         The methods used for fairness corrections depends on whether the
         variable is consider as continuous or discrete. All unordered
         variables are considered being discrete, and all ordered
         variables with more than ``fair_cont_min_values`` are considered as
         being discrete as well. The default (or None) is 20.

    fair_material_disc_method : String (or None), optional\
        Method on how to perform the discretization for materially relevant
        features.\
        ``'NoDiscretization'`` : Variables are not changed. If one of the
        features has more different values than fair_material_max_groups, all
        materially relevant features will formally be treated as continuous.
        The latter may become unreliable if their dimension is not year small.\
        ``'EqualCell'`` : Attempts to create equal cells for each variable.
        Maybe be useful for a very small number of variables with few different
        values.\
        ``'Kmeans'`` : Use Kmeans clustering algorithm to form homogeneous
        cells.\
        Default (or None) is 'Kmeans'.

    fair_protected_disc_method : String (or None), optional\
        Method on how to perform the discretization for protected features.\
        ``'NoDiscretization'`` : Variables are not changed. If one of the
        features has more different values than fair_protected_max_groups,
        all protected features will formally be treated as continuous. The
        latter may become unreliable if their dimension is not very small.\
        ``'EqualCell'`` : Attempts to create equal cells for each variable.
        Maybe be useful for a very small number of variables with few different
        values.\
        ``'Kmeans'`` : Use Kmeans clustering algorithm to form homogeneous
        cells.\
        Default (or None) is ``'Kmeans'``.

    fair_material_max_groups : Integer (or None), optional\
        Level of discretization of materially relavant variables (only if
        needed). Number of groups of materially relavant features for cases when
        materially relavant variables are needed in protected form. This is
        currently only necessary for 'Quantilized'.\
        Its meaning depends on fair_material_disc_method:\
        If ``'EqualCell'``: If more than 1 variable is included among the
        protected variables, this restriction is applied to each variable.\
        If ``'Kmeans'``: This is the number of clusters used by Kmeans.\
        Default (or None) is 5.

    fair_protected_max_groups : Integer (or None), optional\
        Level of discretization of protected variables (only if needed).
        Number of groups of protected features for cases when protected
        variables are needed in discretized form. This is currently only
        necessary for ``'Quantilized'``.\
        Its meaning depends on fair_protected_disc_method:
        If ``'EqualCell'`` : If more than 1 variable is included among the
        protected variables, this restriction is applied to each variable.\
        If ``'Kmeans'`` : This is the number of clusters used by Kmeans.
        Default (or None) is 5.

    fair_regression_method : String (or None), optional\
        Method choice when predictions from machine learning are needed for
        fairnesss corrections (fair_type in (``'Mean'``, ``'MeanVar'``).\
        Available methods are ``'RandomForest'``, ``'RandomForestNminl5'``,
        ``'RandomForestNminls5'``, ``'SupportVectorMachine'``,
        ``'SupportVectorMachineC2'``, ``'SupportVectorMachineC4'``,
        ``'AdaBoost'``, ``'AdaBoost100'``, ``'AdaBoost200'``, ``'GradBoost'``,
        ``'GradBoostDepth6'``, ``'GradBoostDepth12'``, ``'LASSO'``,
        ``'NeuralNet'``, ``'NeuralNetLarge'``, ``'NeuralNetLarger'``,
        ``'Mean'``. If ``'automatic'``, an optimal method will be chosen based
        on 5-fold cross-validation in the training data. If a method is
        specified it will be used for all scores and all adjustments. If
        'automatic', every policy score might be adjusted with a different
        method. 'Mean' is included for cases in which regression methods have
        no explanatory power.\
        Default (or None) is ``'RandomForest'``.

    fair_type : String (or None), optional\
        Method to choose the type of correction for the policy scores.\
        ``'Mean'`` :  Mean dependence of the policy score on protected var's is
        removed by residualisation.\
        ``'MeanVar'`` :  Mean dependence and heteroscedasticity is removed by
        residualisation and rescaling.\
        ``'Quantiled'`` : Removing dependence via (an empricial version of) the
        approach by Strack and Yang (2024) using quantiles.\
        ``'Mean'`` and ``'MeanVar'`` are only availabe for adjusting the score
        (not the decision variables).\
        See the paper by Bearth, Lechner, Mareckova, Muny (2024) for details on
        these methods.\
        Default (or None) is 'Quantiled'.

    gen_method : String (or None), optional.\
        Method to compute assignment algorithm (available methods:
        ``'best_policy_score'``, ``'bps_classifier'``, ``'policy tree'``).
        ``'best_policy_score'`` conducts Black-Box allocations, which are
        obtained by using the scores directly (potentially subject to
        restrictions). When the Black-Box allocations are used for
        allocation of data not used for training, the respective scores
        must be available.
        ``'bps_classifier'`` uses the allocations obtained by
        ``'best_policy_score'`` and trains classifiers. The output will be a
        decision rule that depends on features only and does not require
        knowledge of the policy scores. The actual classifier used is selected
        among four different classifiers offered by sci-kit learn, namely a
        simple neural network, two classification random forests with minimum
        leaf size of 2 and 5, and ADDABoost. The selection is made according
        to the out-of-sample performance on scikit-learns Accuracy Score.
        The implemented ``'policy tree'`` 's are optimal trees, i.e. all
        possible trees are checked if they lead to a better performance.
        If restrictions are specified, then this is incorporated into
        treatment specific cost parameters. Many ideas of the
        implementation follow Zhou, Athey, Wager (2022). If the provided
        policy scores fulfil their conditions (i.e., they use a doubly
        robust double machine learning like score), then they also provide
        attractive theoretical properties.\
        Default (or None) is ``'best_policy_score'``.

    gen_mp_parallel : Integer (or None), optional
        Number of parallel processes (using ray on CPU). The smaller this
        value is, the slower the programme, the smaller its demands on RAM.
        None : 80% of logical cores.
        Default is None.

    gen_outfiletext : String (or None), optional
        File for text output. (.txt) file extension will be automatically
        added.
        Default (or None) is 'txtFileWithOutput'.

    gen_outpath : String or Pathlib object (or None), optional
        Directory to where to put text output and figures. If it does not
        exist, it will be created.
        None : Directory just below the directory where the programme is run.
        Default is None.

    gen_output_type : Integer (or None), optional
        Destination of the output.
        0 : Terminal.
        1 : File.
        2 : File and terminal.
        Default (or None) is 2.

    gen_variable_importance : Boolean
        Compute variable importance statistics based on random forest
        classifiers.
        Default (or None) is True.

    other_costs_of_treat : List of floats (or None), optional
        Treatment specific costs. These costs are directly subtracted from
        the policy scores. Therefore, they should be measured in the same
        units as the scores.
        Default value (or None) with constraints: It defaults to 0.
        Default value (or None) without constraints: Costs will be automatically
        determined such as to enforce constraints in the training data by
        finding cost values that lead to an allocation ('best_policy_score')
        that fulfils restrictions other_max_shares.
        Default (or None) is None.

    other_costs_of_treat_mult : Float or tuple of floats (with as many
                                elements as treatments) (or None), optional
        Multiplier of automatically determined cost values. Use only when
        automatic costs violate the constraints given by other_max_shares.
        This allows to increase (>1) or decrease (<1) the share of treated
        in particular treatment. None: (1, ..., 1).
        Default (or None) is None.

    other_max_shares : Tuple of float elements as treatments) (or None),
                        optional
        Maximum share allowed for each treatment.
        Default (or None) is None.

    pt_depth_tree_1 : Integer (or None), optional
        Depth of 1st optimal tree.
        Default is 3.
        Note that tree depth is defined such that a depth of 1 implies 2
        leaves, a depth of 3 implies 4 leaves, a depth of 3 implies 8 leaves,
        etc.

    pt_depth_tree_2 : Integer (or None), optional
        Depth of 2nd optimal tree. This set is built within the strata
        obtained from the leaves of the first tree. If set to 0, a second
        tree is not built. Default is 1 (together with the default for
        pt_depth_tree_1 this leads to a (not optimal) total tree of level
        of 4. Note that tree depth is defined such that a depth of 1 implies 2
        leaves, a depth of 2 implies 4 leaves, a depth of 3 implies 8 leaves,
        etc.

    pt_enforce_restriction : Boolean (or None), optional
        Enforces the imposed restriction (to some extent) during the
        computation of the policy tree. This increases the quality of trees
        concerning obeying the restrictions, but can be very time consuming.
        It will be automatically set to False if more than 1 policy tree is
        estimated.
        Default (or None) is False.

    pt_eva_cat_mult : Integer (or None), optional
        Changes the number of the evaluation points (pt_no_of_evalupoints)
        for the unordered (categorical) variables to:
        :math:`\\text{pt_eva_cat_mult} \\times \\text{pt_no_of_evalupoints}`
        (available only for the method 'policy tree').
        Default (or None) is 2.

    pt_no_of_evalupoints : Integer (or None), optional
        No of evaluation points for continuous variables. The lower this
        value, the faster the algorithm, but it may also deviate more from
        the optimal splitting rule. This parameter is closely related to
        the approximation parameter of Zhou, Athey, Wager (2022)(A) with
        :math:`\\text{pt_no_of_evalupoints} = \\text{number of observation} / \\text{A}`.
        Only relevant if gen_method is 'policy tree'.
        Default (or None) is 100.

    pt_min_leaf_size : Integer (or None), optional
        Minimum leaf size. Leaves that are smaller than pt_min_leaf_size in
        the training data will not be considered. A larger number reduces
        computation time and avoids some overfitting.
        None :

        .. math::

            min(0.1 \\times \\frac{\\text{Number of training observations}}{{\\text{Number of leaves}}}, 100)

        (if treatment shares are restricted this is multiplied by the smallest
        share allowed). Only relevant if gen_method is 'policy tree'.
        Default is None.

    pt_select_values_cat : Boolean (or None), optional
        Approximation method for larger categorical variables. Since we
        search among optimal trees, for categorical variables variables we
        need to check for all possible combinations of the different values
        that lead to binary splits. Thus number could indeed be huge.
        Therefore, we compare only
        :math:`\\text{pt_no_of_evalupoints} \\times \\text{pt_eva_cat_mult}`
        different combinations. Method 1 (pt_select_values_cat == True) does
        this by randomly drawing values from the particular categorical variable
        and forming groups only using those values. Method 2
        (pt_select_values_cat == False) sorts the values of the categorical
        variables according to a values of the policy score as one would do
        for a standard random forest. If this set is still too large, a
        random sample of the entailed combinations is drawn. Method 1 is
        only available for the method 'policy tree'.

    rnd_shares : Tuple of floats (or None), optional
        Share of treatments of a stochastic assignment as computed by the
        :meth:`~OptimalPolicy.evaluate` method. Sum of all elements must add to
        1. This used only used as a comparison in the evaluation of other
        allocations. None: Shares of treatments in the allocation under
        investigation.
        Default is None.

    var_bb_restrict_name : String (or None), optional
        Name of variable related to a restriction in case of capacity
        constraints. If there is a capacity constraint, preference will be
        given to observations with highest values of this variable.
        Only relevant if gen_method is 'best_policy_score'.
        Default is None.

    var_d_name : String (or None), optional
        Name of (discrete) treatment. Needed in training data only if
        'changers' (different treatment in allocation than observed
        treatment) are analysed and if allocation is compared to observed
        allocation (in :meth:`~OptimalPolicy.evaluate` method).
        Default is None.

    var_effect_vs_0  : List/tuple of strings (or None), optional
        Name of variables of effects of treatment relative to first
        treatment. Dimension is equal to the number of treatments minus 1.
        Default is None.

    var_effect_vs_0_se  : List/tuple of strings (or None), optional
        Name of variables of standard errors of the effects of treatment
        relative to first treatment. Dimension is equal to the number of
        treatments minus 1.
        Default is None.

    var_id_name : (or None), optional
        Name of identifier in data. Default is None.

    var_polscore_desc_name : List/tuple of tuples of strings (or None), optional
        Each tuple of dimension equal to the different treatments
        contains treatment specific variables that are used to evaluate the
        effect of the allocation with respect to those variables. This
        could be for example policy score not used in training, but which
        are relevant nevertheless.
        Default is None.

    var_polscore_name : List or tuple of strings (or None), optional
        Names of treatment specific variables to measure the value of
        individual treatments. This is usually the estimated potential
        outcome or any other score related. This is required for the
        :meth:`~OptimalPolicy.solve` method.
        Default is None.

    var_material_name_ord : List or tuple of strings (nor None), optional
        Materially relavant ordered variables: An effect of the protected
        variables on the scores is allowed, if captured by these variables
        (only). These variables may (or may not) be included among the decision
        variables. These variables must (!) not be included among the protected
        variables.
        Default is None.

    var_material_name_unord : List or tuple of strings (nor None), optional
        Materially relavant unordered variables: An effect of the protected
        variables on the scores is allowed, if captured by these variables
        (only). These variables may (or may not) be included among the decision
        variables. These variables must (!) not be included among the protected
        variables.
        Default is None.

    var_protected_ord_name : List or tuple of strings (nor None), optional
        Names of protected ordered variables. Their influence on the policy
        scores will be removed (conditional on the 'materially important'
        variables). These variables should NOT be contained in decision
        variables, i.e., var_x_name_ord. If they are included, they will be
        removed and var_x_name_ord will be adjusted accordingly.
        Default is None.

    var_protected_unord_name : List or tuple of strings (nor None), optional
        Names of protected unordered variables. Their influence on the policy
        scores will be removed (conditional on the 'materially important'
        variables). These variables should NOT be contained in decision
        variables, i.e., var_x_name_unord. If they are included, they will be
        removed andvar_x_name_unord will be adjusted accordingly.
        Default is None.

    var_vi_x_name : List or tuple of strings or None, optional
        Names of variables for which variable importance is computed.
        Default is None.

    var_vi_to_dummy_name : List or tuple of strings or None, optional
        Names of variables for which variable importance is computed.
        These variables will be broken up into dummies.
        Default is None.

    var_x_name_ord : Tuple of strings (or None), optional
        Name of ordered variables (including dummy variables) used to build
        policy tree and classifier. They are also used to characterise the
        allocation.
        Default is None.

    var_x_name_unord : Tuple of strings (or None), optional
        Name of unordered variables used to build policy tree and classifier.
        They are also used to characterise the allocation. Default is None.

    _int_dpi : Integer (or None), optional
        dpi in plots.
        Default (or None) is 500.
        Internal variable, change default only if you know what you do.

    _int_fontsize : Integer (or None), optional
        Font for legends, from 1 (very small) to 7 (very large).
        Default (or None) is 2.
        Internal variable, change default only if you know what you do.

    _int_output_no_new_dir: Boolean
        Do not create a new directory when the path already exists.
        Default (or None) is False.

    _int_report : Boolean, optional
        Provide information for McfOptPolReports to construct informative
        reports.
        Default (or None) is True.

    _int_with_numba : Boolean (or None), optional
        Use Numba to speed up computations.
        Default (or None) is True.

    _int_with_output : Boolean (or None), optional
        Print output on file and/or screen.
        Default (or None) is True.

    _int_xtr_parallel : Boolean (or None), optional.
        Parallelize to a larger degree to make sure all CPUs are busy for
        most of the time.
        Default (or None) is True.
        Only used for 'policy tree' and
        only used if _int_parallel_processing > 1 (or None)


    Attributes
    ----------

    __version__ : String
        Version of mcf module used to create the instance.

    <NOT-ON-API>

    dc_cfg : DCCfg dataclass
        Parameters used in data cleaning.

    estrisk_cfg : EstRisk dataclass
        Parameters used to account for estimation uncertainty in policy scores.

    fair_cfg : FairCfg dataclass
        Parameters used in fairness adjustment of scores.

    gen_cfg : GenCfg dataclass
        General parameters used in various parts of the programme.

    int_cfg : IntCfg dataclass
        Parameters used in many parts of the class.

    number_scores : Integer
        Number of policy scores.

    other_cfg : OtherCfg dataclass
        Contains other relevant information needed for allocation (like cost,
        constraints).

    pt_cfg : PtCfg dataclass
        Parameters used to build policy tree.

    rnd_cfg : RndCfg dataclass
        Shares for random allocation.

    time_strings : String
        Detailed information on how long the different methods needed.

    var_cfg : VarCfg dataclass
        Variable names.

    var_x_type : Dictionary
        Types of covariates (internal).

    var_x_values : Dictionary
        Values of covariates (internal).

    </NOT-ON-API>
    
    """

    def __init__(
        self, dc_check_perfectcorr=True,
        dc_clean_data=True, dc_min_dummy_obs=10, dc_screen_covariates=True,
        estrisk_value=1,
        fair_adjust_target='xvariables', fair_consistency_test=False,
        fair_cont_min_values=20,
        fair_material_disc_method='Kmeans', fair_material_max_groups=5,
        fair_regression_method='RandomForest',
        fair_protected_disc_method='Kmeans', fair_protected_max_groups=5,
        fair_type='Quantiled',
        gen_method='best_policy_score', gen_mp_parallel='None',
        gen_outfiletext='txtFileWithOutput',
        gen_outpath=None, gen_output_type=2, gen_variable_importance=True,
        other_costs_of_treat=None, other_costs_of_treat_mult=None,
        other_max_shares=None,
        pt_depth_tree_1=3, pt_depth_tree_2=1, pt_enforce_restriction=False,
        pt_eva_cat_mult=1, pt_no_of_evalupoints=100, pt_min_leaf_size=None,
        pt_select_values_cat=False,
        rnd_shares=None,
        var_bb_restrict_name=None, var_d_name=None, var_effect_vs_0=None,
        var_effect_vs_0_se=None, var_id_name=None,
        var_material_name_ord=None, var_material_name_unord=None,
        var_polscore_desc_name=None, var_polscore_name=None,
        var_polscore_se_name=None,
        var_protected_name_ord=None, var_protected_name_unord=None,
        var_vi_x_name=None, var_vi_to_dummy_name=None,
        var_x_name_ord=None, var_x_name_unord=None,
        _int_dpi=500, _int_fontsize=2, _int_output_no_new_dir=False,
        _int_report=True, _int_with_numba=True, _int_with_output=True,
        _int_xtr_parallel=True,
            ):

        self.version = '0.9.0'
        self.__version__ = '0.9.0'

        self.int_cfg = op_init.IntCfg.from_args(
            cuda=False, output_no_new_dir=_int_output_no_new_dir,
            report=_int_report, with_numba=_int_with_numba,
            with_output=_int_with_output, xtr_parallel=_int_xtr_parallel,
            dpi=_int_dpi, fontsize=_int_fontsize,
            )
        self.gen_cfg = op_init.GenCfg.from_args(
            method=gen_method, mp_parallel=gen_mp_parallel,
            outfiletext=gen_outfiletext,
            outpath=gen_outpath, output_type=gen_output_type,
            variable_importance=gen_variable_importance,
            with_output=self.int_cfg.with_output,
            new_outpath=not self.int_cfg.output_no_new_dir,
            )
        self.dc_cfg = op_init.DataCleanCfg.from_args(
            check_perfectcorr=dc_check_perfectcorr,
            clean_data=dc_clean_data,
            min_dummy_obs=dc_min_dummy_obs,
            screen_covariates=dc_screen_covariates,
            )
        self.pt_cfg = op_init.PtCfg.from_args(
            depth_tree_1=pt_depth_tree_1, depth_tree_2=pt_depth_tree_2,
            eva_cat_mult=pt_eva_cat_mult,
            enforce_restriction=pt_enforce_restriction,
            no_of_evalupoints=pt_no_of_evalupoints,
            select_values_cat=pt_select_values_cat,
            min_leaf_size=pt_min_leaf_size,
            )
        self.other_cfg = op_init.OtherCfg.from_args(
            other_costs_of_treat=other_costs_of_treat,
            other_costs_of_treat_mult=other_costs_of_treat_mult,
            other_max_shares=other_max_shares
            )
        self.rnd_cfg = op_init.RndCfg.from_args(rnd_shares=rnd_shares)

        self.var_cfg = op_init.VarCfg.from_args(
            bb_restrict_name=var_bb_restrict_name,
            d_name=var_d_name,
            effect_vs_0=var_effect_vs_0, effect_vs_0_se=var_effect_vs_0_se,
            id_name=var_id_name, polscore_desc_name=var_polscore_desc_name,
            material_ord_name=var_material_name_ord,
            material_unord_name=var_material_name_unord,
            polscore_name=var_polscore_name,
            polscore_se_name=var_polscore_se_name,
            protected_ord_name=var_protected_name_ord,
            protected_unord_name=var_protected_name_unord,
            x_ord_name=var_x_name_ord, x_unord_name=var_x_name_unord,
            vi_x_name=var_vi_x_name, vi_to_dummy_name=var_vi_to_dummy_name)

        self.fair_cfg = op_init.FairCfg.from_args(
            self.gen_cfg,
            adjust_target=fair_adjust_target,
            consistency_test=fair_consistency_test,
            cont_min_values=fair_cont_min_values,
            material_disc_method=fair_material_disc_method,
            protected_disc_method=fair_protected_disc_method,
            material_max_groups=fair_material_max_groups,
            regression_method=fair_regression_method,
            protected_max_groups=fair_protected_max_groups,
            adj_type=fair_type)

        self.estriskcfg = op_init.EstRiskCfg(value=estrisk_value)

        self.time_strings, self.var_x_type, self.var_x_values = {}, {}, {}
        self.bps_class_dict = {}
        self.report = {'fairscores': False,
                       'solvefair': False,
                       'training': False,
                       'evaluation': False,
                       'allocation': False,
                       'estriskscores': False,
                       'training_data_chcksm': 0,   # To identify training data
                       'training_alloc_chcksm': 0,  # To identify train. alloc.
                       'alloc_list': [],   # List because of possible multiple
                       'evalu_list': [],   # allocation, evaluation methods
                       }                   # might be used multiple times.
        self.number_scores = len(self.var_cfg.polscore_name)

    def allocate(self,
                 data_df,
                 data_title: str = '',
                 fair_adjust_decision_vars: bool = False
                 ):
        """
        Allocate observations to treatment state.

        Parameters
        ----------
        data_df : DataFrame
            Input data with at least features or policy scores
            (depending on algorithm).

        data_title : String, optional
            This string is used as title in outputs. The default is ''.

        fair_adjust_decision_vars : Boolean, optional
            If True, it will fairness-adjust the decision variables even when
            fairness adjustments have not been used in training.
            If False, no fairness adjustments of decision variables. However,
            if fairness adjustments of decision variables have already been used
            in training, then these variables will also be fairness adjusted in
            the allocate method, independent of the value of
            ``fair_adjust_decision_vars``.
            The default is False.

        Returns
        -------
        results : Dictionary.
            Contains the results. This dictionary has the following structure:
            'allocation_df' : DataFrame
                data_df with optimal allocation appended.
            'outpath' : Path
                Location of directory in which output is saved.

        """
        (allocation_df, self.gen_cfg.outpath
         ) = op_methods.allocate_method(
             self,
             data_df,
             data_title=data_title,
             fair_adjust_decision_vars=fair_adjust_decision_vars
             )
        results_dic = {
            'allocation_df': allocation_df,
            'outpath': self.gen_cfg.outpath
            }

        return results_dic

    def evaluate(self,
                 allocation_df,
                 data_df,
                 data_title='',
                 seed=12434
                 ):
        """
        Evaluate allocation with potential outcome data.

        Parameters
        ----------
        allocation_df : DataFrame
            Optimal allocation as outputed by the
            :meth:`~OptimalPolicy.solve`, :meth:`~OptimalPolicy.solvefair`,
            and :meth:`~OptimalPolicy.allocate` methods.

        data_df : DataFrame
            Additional information that can be linked to allocation_df.

        data_title : String, optional
            This string is used as title in outputs. The default is ''.

        seed : Integer, optional
            Seed for random number generators. The default is 12434.

        Returns
        -------
        results_all_dic : Dictory
            'results_dic': Collected results of evaluation with
                           self-explanatory keys.
            'outpath': Output path.

        """
        (results_dic, self.gen_cfg.outpath
         ) = op_methods.evaluate_method(self,
                                        allocation_df,
                                        data_df,
                                        data_title=data_title,
                                        seed=seed)
        results_all_dic = {
            'results_dic': results_dic,
            'outpath': self.gen_cfg.outpath
            }

        return results_all_dic

    def evaluate_multiple(self, allocations_dic, data_df):
        """
        Evaluate several allocations simultaneously.

        Parameters
        ----------
        allocations_dic : Dictionary
            Contains dataframes with specific allocations.

        data_df : DataFrame.
            Data with the relevant information about potential outcomes
            which
            will be used to evaluate the allocations.

        Returns
        -------
        results_dic : Dictionary.
            Contains the results. This dictionary has the following structure:
            'outpath' : Path
                Location of directory in which output is saved.

        """
        if not self.gen_cfg.with_output:
            raise ValueError('To use this method, allow output to be written.')
        potential_outcomes_np = data_df[self.var_cfg.polscore_name]
        op_eval.evaluate_multiple(self, allocations_dic, potential_outcomes_np)
        results_dic = {'outpath': self.gen_cfg.outpath
                       }

        return results_dic

    def estrisk_adjust(self, data_df, data_title=''):
        """
        Adjust policy score for estimation risk.

        Parameters
        ----------
        data_df : Dataframe
            Input data.

        data_title : String, optional
            This string is used as title in outputs. The default is ''.

        Returns
        -------
        results_dic : Dictionary.
            Contains the results. This dictionary has the following structure:
            'data_estrisk_df' : DataFrame
                Input data with additional fairness adjusted scores.
            'estrisk_scores_names' : List of strings.
                Names of adjusted scores.
            'outpath' : Path
                Location of directory in which output is saved.

        """
        (data_estrisk_df, estrisk_scores_names, self.gen_cfg.outpath
         ) = op_methods.estrisk_adjust_method(self,
                                              data_df,
                                              data_title=data_title)
        results_dic = {
            'data_estrisk_df': data_estrisk_df,
            'estrisk_scores_names': estrisk_scores_names,
            'outpath': self.gen_cfg.outpath
            }
        return results_dic

    def solvefair(self, data_df, data_title=''):
        """
        Solve for optimal allocation rule with fairness adjustments.

        Follows the suggestions of Bearth, Lechner, Muny, Mareckova (2025,
        arXiV). It has the same syntax and is used in the same way as the solve
        method.

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
                data_df with optimal allocation appended.
            'result_dic' : Dictionary
                Contains additional information about trained allocation rule.
                Only complete when keyword _int_with_output is True.
            'outpath' : Path
                Location of directory in which output is saved.

        """
        (allocation_df, result_dic, self.gen_cfg.outpath
         ) = op_methods.solvefair_method(self,
                                         data_df,
                                         data_title=data_title)
        results_all_dic = {
            'allocation_df': allocation_df,
            'result_dic': result_dic,
            'outpath': self.gen_cfg.outpath
            }
        return results_all_dic

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
                data_df with optimal allocation appended.
            'result_dic' : Dictionary
                Contains additional information about trained allocation rule.
                Only complete when keyword _int_with_output is True.
            'outpath' : Path
                Location of directory in which output is saved.

        """
        (allocation_df, result_dic, self.gen_cfg.outpath
         ) = op_methods.solve_method(self,
                                     data_df,
                                     data_title=data_title)
        results_all_dic = {
            'allocation_df': allocation_df,
            'result_dic': result_dic,
            'outpath': self.gen_cfg.outpath
            }
        return results_all_dic

    def print_time_strings_all_steps(self, title=''):
        """Print an overview over the time needed in all steps of programme."""
        txt = '\n' + '=' * 100 + '\nSummary of computation times of all steps '
        txt += title
        mcf_ps.print_mcf(self.gen_cfg, txt, summary=True)
        val_all = ''
        for _, val in self.time_strings.items():
            val_all += val
        mcf_ps.print_mcf(self.gen_cfg, val_all, summary=True)

    def winners_losers(self,
                       data_df,
                       welfare_df,
                       welfare_reference_df: int = 0,
                       outpath: None = None,
                       title: str = ''
                       ):
        """
        Compare the winners and loser.

        k-means is used to cluster groups of individuals that are similar
        in gains and losses from two user-provided allocations. The groups are
        described by the policy scores as well as the decision, protected,
        and materially relevant variables.

        Parameters
        ----------
        data_df : Dataframe
            Variables used for descriptions.

        welfare_df : DataFrame
            Welfare of the allocations.

        welfare_reference_df : DataFrame, optional
            Welfare of the reference allocation. The default is 0.

        outpath : String or None, optional
            Path used to save the outputs.

        title : String, optional
            Title used in the statistics. The default is ''.

        Returns
        -------
        results_dict : Dictionary.
            Contains the results. This dictionary has the following structure:
            'data_plus_cluster_number_df' : DataFrame
                Cluster number ('cluster_no') is added to data_df.
            'outpath' : Path
                Location of directory in which output is saved.

        """
        (data_plus_cluster_number_df, self.gen_cfg.outpath
         ) = op_methods.winners_losers_method(
             self, data_df, welfare_df,
             welfare_reference_df=welfare_reference_df,
             outpath=outpath,
             title=title
             )
        results_dic = {
            'data_plus_cluster_number_df': data_plus_cluster_number_df,
            'outpath': self.gen_cfg.outpath
            }
        return results_dic

