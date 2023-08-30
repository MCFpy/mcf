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
    """

    def __init__(
        self, dc_check_perfectcorr=True,
        dc_clean_data=True, dc_min_dummy_obs=10, dc_screen_covariates=True,
        gen_method='best_policy_score', gen_outfiletext='txtFileWithOutput',
        gen_outpath=None, gen_output_type=2,
        int_how_many_parallel=None, int_parallel_processing=True,
        int_with_numba=True, int_with_output=True, other_costs_of_treat=None,
        other_costs_of_treat_mult=None, other_max_shares=None,
        pt_depth=3, pt_no_of_evalupoints=100, pt_min_leaf_size=None,
        rnd_shares=None,
        var_bb_restrict_name=None, var_d_name=None, var_effect_vs_0=None,
        var_effect_vs_0_se=None, var_id_name=None, var_polscore_desc_name=None,
        var_polscore_name=None, var_x_ord_name=None, var_x_unord_name=None
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
            'best_policy_score', 'policy tree')
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
        int_how_many_parallel : Integer (or None), optional
            Number of parallel process. None: 80% of logical cores, if this can
            be effectively implemented. Default is None.
        int_parallel_processing : Boolean (or None), optional
            Multiprocessing.
            Default (or None) is True.
        int_with_numba : Boolean (or None), optional
            Use Numba to speed up computations. Default (or None) is True.
        int_with_output : Boolean (or None), optional
            Print output on file and/or screen. Default (or None) is True.
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
            automatic costs violate the constraints given by OTHE_MAX_SHARES.
            This allows to increase (>1) or decrease (<1) the share of treated
            in particular treatment. None: (1, ..., 1). Default is None.
        other_max_shares : Tuple of float elements as treatments) (or None),
                           optional
            Maximum share allowed for each treatment. Default is None.
        pt_depth : Integer (or None), optional
            Depth of tree. Defined such that pt_depth == 1 implies 2 splits,
            pt_depth = 2 implies 4 leafs, pt_depth= 3 implies 8 leafs, etc.
            Only relevant if gen_method is 'policy tree'.
            Default (or None) is 3.
        pt_no_of_evalupoints : Integer (or None), optional
            No of evaluation points for continous variables. The lower this
            value, the faster the algorithm, but it may also deviate more from
            the optimal splitting rule. This parameter is closely related to
            the approximation parameter of Zhou, Athey, Wager (2022)(A) with
            pt_no_of_evalupoints = number of observation / A.
            Only relevant if gen_method is 'policy tree'.
            Default (or None) is 100.
        pt_min_leaf_size : Integer (or None), optional
            Minimum leaf size. Leaves that are smaller than PT_MIN_LEAF_SIZE in
            the training data will not be considered. A larger number reduces
            computation time and avoids some overfitting.
            None: 0.1 x # of training observations / # of leaves.
            Only relevant if gen_method is 'policy tree'. Default is None.
        rnd_shares : Tuple of floats (or None), optional
            Share of treatments of a stochastic assignment as computed by the
            evaluate method. Sum of all elements must add to 1. This used only
            used as a comparison in the evaluation of other allocations.
            None: Shares of treatments in the allocation under investigation.
            Default is None.
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
        var_x_ord_name : Tuple of strings (or None), optional
            Name of ordered variables used to build policy tree. They are also
            used to characterise the allocation. Default is None.
        var_x_unord_name : Tuple of strings (or None), optional
            Name of unordered variables used to build policy tree. They are
            also used to characterise the allocation. Default is None.
        """
        self.int_dict = op_init.init_int(
            how_many_parallel=int_how_many_parallel,
            parallel_processing=int_parallel_processing,
            with_numba=int_with_numba, with_output=int_with_output)

        self.gen_dict = op_init.init_gen(
            method=gen_method, outfiletext=gen_outfiletext,
            outpath=gen_outpath, output_type=gen_output_type,
            with_output=self.int_dict['with_output'])

        self.dc_dict = op_init.init_dc(
            check_perfectcorr=dc_check_perfectcorr,
            clean_data=dc_clean_data, min_dummy_obs=dc_min_dummy_obs,
            screen_covariates=dc_screen_covariates)

        self.pt_dict = op_init.init_pt(depth=pt_depth,
                                       no_of_evalupoints=pt_no_of_evalupoints,
                                       min_leaf_size=pt_min_leaf_size)

        self.other_dict = {'costs_of_treat': other_costs_of_treat,
                           'costs_of_treat_mult': other_costs_of_treat_mult,
                           'max_shares': other_max_shares}

        self.rnd_dict = {'shares': rnd_shares}

        self.var_dict = op_init.init_var(
            bb_restrict_name=var_bb_restrict_name, d_name=var_d_name,
            effect_vs_0=var_effect_vs_0, effect_vs_0_se=var_effect_vs_0_se,
            id_name=var_id_name, polscore_desc_name=var_polscore_desc_name,
            polscore_name=var_polscore_name, x_ord_name=var_x_ord_name,
            x_unord_name=var_x_unord_name)

        self.time_strings, self.var_x_type, self.var_x_values = {}, {}, {}

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
        op_init.init_gen_solve(self, data_df)
        op_init.init_other_solve(self)
        method = self.gen_dict['method']
        if method == 'policy tree':
            op_init.init_pt_solve(self, len(data_df))
        if self.gen_dict['with_output']:
            print_dic_values_all_optp(self, summary_top=True,
                                      summary_dic=False, stage='Training')
        if method in ('policy tree', 'best_policy_score'):
            (data_new_df, bb_rest_variable) = op_data.prepare_data_bb_pt(
                self, data_df, black_box=method == 'best_policy_score')
            if method == 'best_policy_score':
                allocation_df = op_bb.black_box_allocation(self, data_new_df,
                                                           bb_rest_variable)
            elif method == 'policy tree':
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
        if self.gen_dict['with_output']:
            print_dic_values_all_optp(self, summary_top=True,
                                      summary_dic=False, stage='Allocation')
        method = self.gen_dict['method']
        if method == 'best_policy_score':
            allocation_df = self.solve(data_df, data_title='Prediction data')
        elif method == 'policy tree':
            allocation_df = op_pt.policy_tree_prediction_only(self, data_df)
        time_name = [f'Time for {method:20} allocation:  ',]
        time_difference = [time() - time_start]
        if self.gen_dict['with_output']:
            time_str = ps.print_timing(
                self.gen_dict, f'{method:20} Allocation ', time_name,
                time_difference, summary=True)
        key = f'{method} allocation ' + data_title
        self.time_strings[key] = time_str
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
        # op_init.init_rnd_shares(self, allocation_df, d_ok)
        op_init.init_rnd_shares(self, data_df, d_ok)
        if d_ok:
            allocation_df['observed'] = data_df[var_dic['d_name']]
        allocation_df['random'] = op_eval.get_random_allocation(
            self, len(data_df), seed)
        results_dic = op_eval.evaluate(self, data_df, allocation_df, d_ok,
                                       polscore_desc_ok, desc_var)
        time_name = [f'Time for Evaluation {data_title}:     ',]
        time_difference = [time() - time_start]
        if self.gen_dict['with_output']:
            time_str = ps.print_timing(
                self.gen_dict, f'Evaluation of {data_title} with '
                f'{gen_dic["method"]}', time_name, time_difference,
                summary=True)
        key = 'evaluate_' + data_title
        self.time_strings[key] = time_str
        return results_dic

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
    if optp_.gen_dict['method'] == 'policy_tree':
        add_list = [optp_.var_x_type, optp_.var_x_values, optp_.pt_dict]
        add_list_name = ['var_x_type', 'var_x_values', 'pt_dict']
        dic_list.extend(add_list)
        dic_name_list.extend(add_list_name)
    for dic, dic_name in zip(dic_list, dic_name_list):
        ps.print_dic(dic, dic_name, optp_.gen_dict, summary=summary)
    ps.print_mcf(optp_.gen_dict, '\n', summary=summary)
