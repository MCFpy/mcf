"""Created on Mon May 8 2023.

Contains the class and the functions needed for running the mcf.
@author: MLechner
-*- coding: utf-8 -*-
"""
from time import time

from mcf import mcf_print_stats_functions as ps
from mcf import optpolicy_bb_functions as op_bb
from mcf import optpolicy_data_functions as op_data
from mcf import optpolicy_evaluation_functions as op_eval
from mcf import optpolicy_init_functions as op_init
from mcf import optpolicy_pt_functions as op_pt


class OptimalPolicy:
    """
    The class contains all methods necessary for the Optimal Policy module.

    Attributes
    ----------
        int_dict : Dictionary
            Parameters used in many parts of the class.
        dc_dict : Dictionary
            Parameters used in data cleaning.
            List of list containing the estimated causal forest.
        gen_dict : Dictionary
            General parameters used in various parts of the programme.
        other_dict : Dictionary
            Contains other relevant information needed for allocation (like
                cost, constraints)
        pt_dict : Dictionary
            Parameters used to build policy tree.
        rnd_dict: Dictionary
            Shares for random allocation.
        time_strings : String.
            Detailed information on how the long the different methods needed.
        var_dict : Dictionary
            Variable names.
        var_x_type : Dictionary
            Types of covariates (internal).
        var_x_values : Dictionary
            Values of covariates (internal).

    Methods
    -------
        solve : Building the assignment algorithm with training data.
        allocate : Allocate treatments given the assignment algorithm.
        evaluate : Evaluate an allocation.
        evaluate_multiple : Evaluate several allocations simultanously.
    """

    def __init__(
        self, dc_check_perfectcorr=True,
        dc_clean_data=True, dc_min_dummy_obs=10, dc_screen_covariates=True,
        gen_method='best_policy_score', gen_outfiletext='txtFileWithOutput',
        gen_outpath=None, gen_output_type=2, gen_variable_importance=True,
        other_costs_of_treat=None, other_costs_of_treat_mult=None,
        other_max_shares=None,
        pt_depth_tree_1=3, pt_depth_tree_2=1, pt_enforce_restriction=False,
        pt_eva_cat_mult=1, pt_no_of_evalupoints=100, pt_min_leaf_size=None,
        pt_select_values_cat=False,
        rnd_shares=None,
        var_bb_restrict_name=None, var_d_name=None, var_effect_vs_0=None,
        var_effect_vs_0_se=None, var_id_name=None, var_polscore_desc_name=None,
        var_polscore_name=None, var_vi_x_name=None, var_vi_to_dummy_name=None,
        var_x_name_ord=None, var_x_name_unord=None,
        _int_how_many_parallel=None, _int_output_no_new_dir=False,
        _int_parallel_processing=True, _int_report=True, _int_with_numba=True,
        _int_with_output=True, _int_xtr_parallel=True
            ):
        """Define Constructor for OptimalPolicy class.

        Args:
        ----
        dc_screen_covariates : Boolean (or None), optional
            Check features. Default (or None) is True.
        dc_check_perfectcorr : Boolean (or None), optional
            Features that are perfectly correlated are deleted (1 of them).
            Only relevant if dc_screen_covariates is True.
            Default (or None) is True.
        dc_min_dummy_obs : Integer (or None), optional
            Delete dummmy variables that have less than dc_min_dummy_obs in one
            of their categories. Only relevant if dc_screen_covariates is True.
            Default (or None) is 10.
        dc_clean_data : Boolean (or None), optional
            Remove all missing & unnecessary variables.
            Default (or None) is True.
        gen_method : String (or None), optional.
            Method to compute assignment algorithm (available methods:
            'best_policy_score', 'policy tree', 'policy tree old')
            'best_policy_score' conducts Black-Box allocations, which are
            obtained by using the scores directly (potentially subject to
            restrictions). When the Black-Box allocations are used for
            allocation of data not used for training, the respective scores
            must be available.
            The implemented 'policy tree' 's are optimal trees, i.e. all
            possible trees are checked if they lead to a better performance.
            If restrictions are specified, then this is incorparated into
            treatment specific cost parameters. Many ideas of the
            implementation follow Zhou, Athey, Wager (2022). If the provided
            policy scores fulfil their conditions (i.e., they use a doubly
            robust double machine learning like score), then they also provide
            attractive theoretical properties.
            'policy tree' is very similar to 'policy tree old' It uses
            different approximation rules and uses slightly different coding.
            In many cases it should be faster than 'policy tree old'.
            Default (or None) is 'best_policy_score'.
        gen_outfiletext : String (or None), optional
            File for text output. *.txt file extension will be automatically
            added. Default (or None) is 'txtFileWithOutput'.
        gen_outpath : String (or None), optional
            Directory to where to put text output and figures. If it does not
            exist, it will be created.
            None: *.out directory just below to the directory where the
            programme is run. Default is None.
        gen_output_type : Integer (or None), optional
            Destination of the output. 0: Terminal, 1: File,
            2: File and terminal. Default (or None) is 2.
        gen_variable_importance : Boolean
            Compute variable importance statistics based on random forest
            classifiers. Default is True.
        other_costs_of_treat : List of floats (or None), optional
            Treatment specific costs. These costs are directly substracted from
            the policy scores. Therefore, they should be measured in the same
            units as the scores.
            None (when there are no constraints): 0
            None (when are constraints): Costs will be automatically determined
                such as to enforce constraints in the training data by finding
                cost values that lead to an allocation ('best_policy_score')
                that fulfils restrictions other_max_shares.
            Default is None.
        other_costs_of_treat_mult : Float or tuple of floats (with as many
                                    elements as treatments) (or None), optional
            Multiplier of automatically determined cost values. Use only when
            automatic costs violate the constraints given by other_max_shares.
            This allows to increase (>1) or decrease (<1) the share of treated
            in particular treatment.
            None: (1, ..., 1). Default is None.
        other_max_shares : Tuple of float elements as treatments) (or None),
                           optional
            Maximum share allowed for each treatment. Default is None.
        pt_depth_tree_1 : Integer (or None), optional
            Depth of 1st optimal tree. Default is 3.
            Note that tree depth is defined such that a depth of 1 implies 2
            leaves, a depth of implies 4 leaves, a depth of 3 implies 8 leaves,
            etc.
        pt_depth_tree_2 = 2   Integer (or None), optional
            Depth of 2nd optimal tree. This set is build within the strata
            obtained from the leaves of the first tree. If set to 0, a second
            tree is not build. Default is 1 (together with the default for
            pt_depth_tree_1 this leads to a (not optimal) total tree of level
            of 4.
            Note that tree depth is defined such that a depth of 1 implies 2
            leaves, a depth of implies 4 leaves, a depth of 3 implies 8 leaves,
            etc.
        pt_enforce_restriction : Boolean (or None), optional
            Enforces the imposed restriction (to some extent) during the
            computation of the policy tree. This increases the quality of trees
            concerning obeying the restrictions, but can be very time consuming.
            It will be automatically set to False if more than 1 policy tree is
            estimated. Default (or None) is False.
        pt_eva_cat_mult : Integer (or None), optional
            Changes the number of the evaluation points (pt_no_of_evalupoints)
            for the unordered (categorical) variables to:
                pt_eva_cat_mult * pt_no_of_evalupoints
            (available only for the method 'policy tree').
            Default (or None is 1).
        pt_no_of_evalupoints : Integer (or None), optional
            No of evaluation points for continous variables. The lower this
            value, the faster the algorithm, but it may also deviate more from
            the optimal splitting rule. This parameter is closely related to
            the approximation parameter of Zhou, Athey, Wager (2022)(A) with
            pt_no_of_evalupoints = number of observation / A.
            Only relevant if gen_method is 'policy tree' or 'policy tree old'.
            Default (or None) is 100.
        pt_min_leaf_size : Integer (or None), optional
            Minimum leaf size. Leaves that are smaller than PT_MIN_LEAF_SIZE in
            the training data will not be considered. A larger number reduces
            computation time and avoids some overfitting.
            None: 0.1 x # of training observations / # of leaves (if treatment
            shares are restricted this is multiplied by the smallest share
            allowed).
            Only relevant if gen_method is 'policy tree' or 'policy tree old'.
            Default is None.
        pt_select_values_cat : Boolean (or None), optional
            Approximation method for larger categorical variables. Since we
            search among optimal trees, for catorgical variables variables we
            need to check for all possible combinations of the different values
            that lead to binary splits. Thus number could indeed be huge.
            Therefore, we compare only pt_no_of_evalupoints * 2 different
            combinations. Method 1 (pt_select_values_cat == True) does this by
            randomly drawing values from the particular categorical variable
            and forming groups only using those values. Method 2
            (pt_select_values_cat == False) sorts the values of the categorical
            variables according to a values of the policy score as one would do
            for a standard random forest. If this set is still too large, a
            random sample of the entailed combinations is drawn. Method 1 is
            only available for the method 'policy tree'.
        rnd_shares : Tuple of floats (or None), optional
            Share of treatments of a stochastic assignment as computed by the
            evaluate method. Sum of all elements must add to 1. This used only
            used as a comparison in the evaluation of other allocations.
            None: Shares of treatments in the allocation under investigation.
            Default is None.
        var_vi_x_name : List of strings or None, optional
            Names of variables for which variable importance is computed.
            Default is None.
        var_vi_to_dummy_name : List of strings or None, optional
            Names of variables for which variable importance is computed.
            These variables will be broken up into dummies. Default is None.
        var_bb_restrict_name : String (or None), optional
            Name of variable related to a restriction in case of capacity
            constraints. If there is a capacity constraint, preference will be
            given to observations with highest values of this variable.
            Only relevant if gen_method is 'best_policy_score'.
            Default is None.
        var_d_name : String (or None), optional
            Name of (discrete) treatment. Needed in training data only if
            'changers' (different treatment in alloication than observed
            treatment) are analysed and if allocation is compared to observed
            allocation (in evaluate method). Default is None.
        var_effect_vs_0  : Tuple of strings (or None), optional
            Name of variables of effects of treatment relative to first
            treatment. Dimension is equal to the number of treatments minus 1.
            Default is None.
        var_effect_vs_0_se  : Tuple of strings (or None), optional
            Name of variables of effects of treatment relative to first
            treatment. Dimension is equal to the number of treatments minus 1.
            Default is None.
        var_id_name : (or None), optional
            Name of identifier in data. Default is None.
        var_polscore_desc_name : Tuple of tuples of strings (or None), optional
            Each tuple of dimension equal to the different treatments
            contains treatment specific variables that are used to evaluate the
            effect of the allocation with respect to those variables. This
            could be for example policy score not used in training,but which
            are relevant nevertheless. Default is None.
        var_polscore_name : Tuple of strings (or None), optional
            Names of treatment specific variables to measure the value of
            individual treatments. This is ususally the estimated potential
            outcome or any other score related. This is required for the solve
            method. Default is None.
        var_x_name_ord : Tuple of strings (or None), optional
            Name of ordered variables used to build policy tree. They are also
            used to characterise the allocation. Default is None.
        var_x_name_unord : Tuple of strings (or None), optional
            Name of unordered variables used to build policy tree. They are
            also used to characterise the allocation. Default is None.
        _int_how_many_parallel : Integer (or None), optional
            Number of parallel process. None: 80% of logical cores, if this can
            be effectively implemented. Default is None.
        _int_output_no_new_dir: Boolean
            Do not create a new directory when the path already exists. Default
            is False.
        _int_parallel_processing : Boolean (or None), optional
            Multiprocessing.
            Default (or None) is True.
        _int_report : Boolean, optional
            Provide information for McfOptPolReports to construct informative
            reports. Default is True.
        _int_with_numba : Boolean (or None), optional
            Use Numba to speed up computations. Default (or None) is True.
        _int_with_output : Boolean (or None), optional
            Print output on file and/or screen. Default (or None) is True.
        _int_xtr_parallel : Boolean (or None), optional.
            Parallelize to a larger degree to make sure all CPUs are busy for
            most of the time. Default is True. Only used for 'policy tree' and
            only used if _int_parallel_processing > 1 (or None)
        """
        self.int_dict = op_init.init_int(
            cuda=False, how_many_parallel=_int_how_many_parallel,
            output_no_new_dir=_int_output_no_new_dir,
            parallel_processing=_int_parallel_processing, report=_int_report,
            with_numba=_int_with_numba, with_output=_int_with_output,
            xtr_parallel=_int_xtr_parallel)

        self.gen_dict = op_init.init_gen(
            method=gen_method, outfiletext=gen_outfiletext,
            outpath=gen_outpath, output_type=gen_output_type,
            variable_importance=gen_variable_importance,
            with_output=self.int_dict['with_output'],
            new_outpath=not self.int_dict['output_no_new_dir'])

        self.dc_dict = op_init.init_dc(
            check_perfectcorr=dc_check_perfectcorr,
            clean_data=dc_clean_data, min_dummy_obs=dc_min_dummy_obs,
            screen_covariates=dc_screen_covariates)

        self.pt_dict = op_init.init_pt(
            depth_tree_1=pt_depth_tree_1, depth_tree_2=pt_depth_tree_2,
            eva_cat_mult=pt_eva_cat_mult,
            enforce_restriction=pt_enforce_restriction,
            no_of_evalupoints=pt_no_of_evalupoints,
            select_values_cat=pt_select_values_cat,
            min_leaf_size=pt_min_leaf_size)

        self.other_dict = {'costs_of_treat': other_costs_of_treat,
                           'costs_of_treat_mult': other_costs_of_treat_mult,
                           'max_shares': other_max_shares}

        self.rnd_dict = {'shares': rnd_shares}

        self.var_dict = op_init.init_var(
            bb_restrict_name=var_bb_restrict_name, d_name=var_d_name,
            effect_vs_0=var_effect_vs_0, effect_vs_0_se=var_effect_vs_0_se,
            id_name=var_id_name, polscore_desc_name=var_polscore_desc_name,
            polscore_name=var_polscore_name, x_ord_name=var_x_name_ord,
            x_unord_name=var_x_name_unord, vi_x_name=var_vi_x_name,
            vi_to_dummy_name=var_vi_to_dummy_name)

        self.time_strings, self.var_x_type, self.var_x_values = {}, {}, {}
        self.report = {'training': False,
                       'evaluation': False,
                       'allocation': False,
                       'training_data_chcksm': 0,   # To identify training data
                       'training_alloc_chcksm': 0,  # To identify train. alloc.
                       'alloc_list': [],   # List because of possible multiple
                       'evalu_list': [],   # allocation, evaluation methods
                       }                   # might be used multiple times.

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
        allocation_df : DataFrame
            data_df with optimal allocation appended.

        """
        time_start = time()

        self.report['training'] = True
        self.report['training_data_chcksm'] = op_data.dataframe_checksum(
            data_df)

        op_init.init_gen_solve(self, data_df)
        op_init.init_other_solve(self)
        method = self.gen_dict['method']
        if method in ('policy tree', 'policy tree old'):
            op_init.init_pt_solve(self, len(data_df))
        if self.gen_dict['with_output']:
            print_dic_values_all_optp(self, summary_top=True,
                                      summary_dic=False, stage='Training')
        if method in ('policy tree', 'policy tree old', 'best_policy_score'):
            (data_new_df, bb_rest_variable) = op_data.prepare_data_bb_pt(
                self, data_df)
            if method == 'best_policy_score':
                allocation_df = op_bb.black_box_allocation(self, data_new_df,
                                                           bb_rest_variable)
            elif method in ('policy tree', 'policy tree old'):
                allocation_df = op_pt.policy_tree_allocation(self, data_new_df)
        # Timing
        time_name = [f'Time for {method:20} training:    ',]
        time_difference = [time() - time_start]
        if self.gen_dict['with_output']:
            time_str = ps.print_timing(
                self.gen_dict, f'{method:20} Training ', time_name,
                time_difference, summary=True)
        key = f'{method} training ' + data_title
        self.time_strings[key] = time_str

        self.report['training_alloc_chcksm'] = op_data.dataframe_checksum(
            allocation_df)

        return allocation_df

    def allocate(self, data_df, data_title=''):
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
        allocation_df : DataFrame
            data_df with optimal allocation appended.

        """
        time_start = time()
        method = self.gen_dict['method']

        self.report['allocation'] = True
        data_train = (op_data.dataframe_checksum(data_df)
                      == self.report['training_data_chcksm'])
        if method == 'policy tree':
            method_str = 'Policy Tree'
        elif method == 'best_policy_score':
            method_str = 'Best Policy Score'
        self.report['txt'] = ('\nAllocation of unit to treatments using '
                              f'{method_str}.'
                              '\nTraining data '
                              f'{"IS NOT" if data_train else "IS"} used.'
                              )

        if self.gen_dict['with_output']:
            print_dic_values_all_optp(self, summary_top=True,
                                      summary_dic=False, stage='Allocation')

        if method == 'best_policy_score':
            allocation_df = self.solve(data_df, data_title='Prediction data')
        elif method in ('policy tree', 'policy tree old'):
            allocation_df = op_pt.policy_tree_prediction_only(self, data_df)
        time_name = [f'Time for {method:20} allocation:  ',]
        time_difference = [time() - time_start]
        if self.gen_dict['with_output']:
            time_str = ps.print_timing(
                self.gen_dict, f'{method:20} Allocation ', time_name,
                time_difference, summary=True)
        key = f'{method} allocation ' + data_title
        self.time_strings[key] = time_str
        self.report['alloc_list'].append(self.report['txt'])
        return allocation_df

    def evaluate(self, allocation_df, data_df, data_title='', seed=12434):
        """
        Evaluate allocation with potential outcome data.

        Parameters
        ----------
        allocation_df : DataFrame
            Optimal allocation as outputed by the solve and allocate methods.
        data_df : DataFrame
            Additional information that can be linked to allocation_df.
        data_title : String, optional
            This string is used as title in outputs. The default is ''.
        seed : Integer, optional
            Seed for random number generators. The default is 12434.

        Returns
        -------
        results_dic : Dictory
            Collected results of evaluation with self-explanatory keys.

        """
        time_start = time()

        self.report['evaluation'] = True
        alloc_train = (op_data.dataframe_checksum(allocation_df)
                       == self.report['training_alloc_chcksm'])

        if self.gen_dict['with_output']:
            print_dic_values_all_optp(self, summary_top=True,
                                      summary_dic=False, stage='Evaluation')
        var_dic, gen_dic = self.var_dict, self.gen_dict
        txt = '\n' + '=' * 100 + '\nEvaluating allocation of '
        txt += f'{gen_dic["method"]} with {data_title}\n' + '-' * 100
        ps.print_mcf(gen_dic, txt, summary=True)
        (data_df, d_ok, polscore_desc_ok, desc_var
         ) = op_data.prepare_data_eval(self, data_df)
        if len(allocation_df) != len(data_df):
            d_ok = False
        op_init.init_rnd_shares(self, data_df, d_ok)
        if d_ok:
            allocation_df['observed'] = data_df[var_dic['d_name']]
        allocation_df['random'] = op_eval.get_random_allocation(
            self, len(data_df), seed)
        results_dic = op_eval.evaluate(self, data_df, allocation_df, d_ok,
                                       polscore_desc_ok, desc_var)
        if (self.gen_dict['with_output']
                and self.gen_dict['variable_importance']):
            op_eval.variable_importance(self, data_df, allocation_df, seed)
        time_name = [f'Time for Evaluation {data_title}:     ',]
        time_difference = [time() - time_start]
        if self.gen_dict['with_output']:
            time_str = ps.print_timing(
                self.gen_dict, f'Evaluation of {data_title} with '
                f'{gen_dic["method"]}', time_name, time_difference,
                summary=True)
        key = 'evaluate_' + data_title
        self.time_strings[key] = time_str

        train_str = 'the SAME as ' if alloc_train else 'DIFFERENT from '
        rep_txt = ('Allocation analysed is ' + train_str + 'the one obtained '
                   'from the training data. '
                   )
        self.report['evalu_list'].append((rep_txt, results_dic))
        return results_dic

    def evaluate_multiple(self, allocations_dic, data_df):
        """
        Evaluate several allocations simultanously.

        Parameters
        ----------
        allocations_dic : Dictionary.
            Contains dataframes with specific allocations.
        data_df : DataFrame.
            Data with the relevant information about potential outcomes which
            will be used to evaluate the allocations.

        Returns
        -------
        None.

        """
        if not self.gen_dict['with_output']:
            raise ValueError('To use this method, allow output to be written.')
        potential_outcomes_np = data_df[self.var_dict['polscore_name']]
        op_eval.evaluate_multiple(self, allocations_dic, potential_outcomes_np)

    def print_time_strings_all_steps(self):
        """Print an overview over the time needed in all steps of programme."""
        txt = '\n' + '=' * 100 + '\nSummary of computation times of all steps'
        ps.print_mcf(self.gen_dict, txt, summary=True)
        val_all = ''
        for _, val in self.time_strings.items():
            val_all += val
        ps.print_mcf(self.gen_dict, val_all, summary=True)


def print_dic_values_all_optp(optp_, summary_top=True, summary_dic=False,
                              stage=''):
    """Print the dictionaries."""
    txt = '=' * 100 + f'\nOptimal Policy Modul ({stage}) with '
    txt += f'{optp_.gen_dict["method"]}' + '\n' + '-' * 100
    ps.print_mcf(optp_.gen_dict, txt, summary=summary_top)
    print_dic_values_optp(optp_, summary=summary_dic)


def print_dic_values_optp(optp_, summary=False):
    """Print values of dictionaries that determine module."""
    dic_list = [optp_.int_dict, optp_.gen_dict, optp_.dc_dict,
                optp_.other_dict, optp_.rnd_dict, optp_.var_dict]
    dic_name_list = ['int_dict', 'gen_dict', 'dc_dict',
                     'other_dict', 'rnd_dict', 'var_dict']
    if optp_.gen_dict['method'] in ('policy tree', 'policy tree old'):
        add_list = [optp_.var_x_type, optp_.var_x_values, optp_.pt_dict]
        add_list_name = ['var_x_type', 'var_x_values', 'pt_dict']
        dic_list.extend(add_list)
        dic_name_list.extend(add_list_name)
    for dic, dic_name in zip(dic_list, dic_name_list):
        ps.print_dic(dic, dic_name, optp_.gen_dict, summary=summary)
    ps.print_mcf(optp_.gen_dict, '\n', summary=summary)
