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
    """Class used for all steps of the Optimal Policy Allication & Analysis."""

    def __init__(
        self, dc_check_perfectcorr=True,
        dc_clean_data=True, dc_min_dummy_obs=True, dc_screen_covariates=True,
        gen_method='best_policy_score', gen_outfiletext=None, gen_outpath=None,
        gen_output_type=None,
        int_how_many_parallel=None, int_parallel_processing=True,
        int_with_numba=True, int_with_output=True, other_costs_of_treat=None,
        other_costs_of_treat_mult=None, other_max_shares=None,
        pt_depth=3, pt_no_of_evalupoints=100, pt_min_leaf_size=None,
        rnd_shares=None,
        var_bb_restrict_name=None, var_d_name=None, var_effect_vs_0=None,
        var_effect_vs_0_se=None, var_id_name=None, var_polscore_desc_name=None,
        var_polscore_name=None, var_x_ord_name=None, var_x_unord_name=None
            ):

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
        """Train all models."""
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
        """Allocate observations based on covariates or potential outcomes."""
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
        """Evaluate allocation with potential outcome data."""
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
        """Print an overview over the timing."""
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
