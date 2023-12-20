import numpy as np
import pandas as pd
import io


class OptimalPolicy:
    """
    The :class:`OptimalPolicy` contains all methods necessary for the Optimal Policy module.

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
        Delete dummmy variables that have less than dc_min_dummy_obs in one
        of their categories. Only relevant if dc_screen_covariates is True.
        Default (or None) is 10.

    dc_clean_data : Boolean (or None), optional
        Remove all missing & unnecessary variables.
        Default (or None) is True.

    """

    def __init__(
        self, dc_check_perfectcorr=True,
        dc_clean_data=True, dc_min_dummy_obs=10, dc_screen_covariates=True,
        gen_method='best_policy_score', gen_outfiletext='txtFileWithOutput',
        gen_outpath=None, gen_output_type=2, gen_variable_importance=False,
        other_costs_of_treat=None, other_costs_of_treat_mult=None,
        other_max_shares=None,
        pt_depth=3, pt_enforce_restriction=True, pt_eva_cat_mult=1,
        pt_no_of_evalupoints=100, pt_min_leaf_size=None,
        pt_select_values_cat=False,
        rnd_shares=None,
        var_bb_restrict_name=None, var_d_name=None, var_effect_vs_0=None,
        var_effect_vs_0_se=None, var_id_name=None, var_polscore_desc_name=None,
        var_polscore_name=None, var_vi_x_name=None, var_vi_to_dummy_name=None,
        var_x_ord_name=None, var_x_unord_name=None,
        _int_how_many_parallel=None, _int_output_no_new_dir=False,
        _int_parallel_processing=True, _int_with_numba=True,
        _int_with_output=True,
            ):
        """Define Constructor for OptimalPolicy class.

        """
        self.int_dict = op_init.init_int(
            how_many_parallel=_int_how_many_parallel,
            output_no_new_dir=_int_output_no_new_dir,
            parallel_processing=_int_parallel_processing,
            with_numba=_int_with_numba, with_output=_int_with_output)

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
            depth=pt_depth, eva_cat_mult=pt_eva_cat_mult,
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
            polscore_name=var_polscore_name, x_ord_name=var_x_ord_name,
            x_unord_name=var_x_unord_name, vi_x_name=var_vi_x_name,
            vi_to_dummy_name=var_vi_to_dummy_name)

        self.time_strings, self.var_x_type, self.var_x_values = {}, {}, {}

    @property
    def int_dict(self):
        """
        ``Dictionary``, parameters used in many parts of the class.
        """
        return self._blind_dict    
    

    @property
    def dc_dict(self):
        """
        ``Dictionary``, parameters used in data cleaning.
        """
        return self._dc_dict  
    
    
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
        if method in ('policy tree', 'policy tree eff'):
            op_init.init_pt_solve(self, len(data_df))
        if self.gen_dict['with_output']:
            print_dic_values_all_optp(self, summary_top=True,
                                      summary_dic=False, stage='Training')
        if method in ('policy tree', 'policy tree eff', 'best_policy_score'):
            (data_new_df, bb_rest_variable) = op_data.prepare_data_bb_pt(
                self, data_df)
            if method == 'best_policy_score':
                allocation_df = op_bb.black_box_allocation(self, data_new_df,
                                                           bb_rest_variable)
            elif method in ('policy tree', 'policy tree eff'):
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
        elif method in ('policy tree', 'policy tree eff'):
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
        op_init.init_rnd_shares(self, data_df, d_ok)
        if d_ok:
            allocation_df['observed'] = data_df[var_dic['d_name']]
        allocation_df['random'] = op_eval.get_random_allocation(
            self, len(data_df), seed)
        results_dic = op_eval.evaluate(self, data_df, allocation_df, d_ok,
                                       polscore_desc_ok, desc_var)
        if self.gen_dict['with_output']:
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
    if optp_.gen_dict['method'] in ('policy tree', 'policy tree eff'):
        add_list = [optp_.var_x_type, optp_.var_x_values, optp_.pt_dict]
        add_list_name = ['var_x_type', 'var_x_values', 'pt_dict']
        dic_list.extend(add_list)
        dic_name_list.extend(add_list_name)
    for dic, dic_name in zip(dic_list, dic_name_list):
        ps.print_dic(dic, dic_name, optp_.gen_dict, summary=summary)
    ps.print_mcf(optp_.gen_dict, '\n', summary=summary)
