"""
Methods for optimal policy analysis.

Created on Thu Aug  3 15:23:17 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
from copy import deepcopy
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats_functions as mcf_ps
from mcf.mcf_post_functions import kmeans_labels
from mcf import optpolicy_bb_functions as op_bb
from mcf import optpolicy_bb_cl_functions as op_bb_cl
from mcf import optpolicy_data_functions as op_data
from mcf import optpolicy_estrisk_functions as op_estrisk
from mcf import optpolicy_evaluation_functions as op_eval
from mcf import optpolicy_fair_functions as op_fair
from mcf import optpolicy_init_functions as op_init
from mcf import optpolicy_pt_functions as op_pt

if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicy


def estrisk_adjust_method(optp_: 'OptimalPolicy',
                          data_df: pd.DataFrame,
                          data_title: str = ''
                          ) -> tuple[pd.DataFrame, list, Path]:
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
    data_estrisk_df : DataFrame
        Input data with additional fairness adjusted scores.
    estrisk_scores_names : List of strings.
        Names of adjusted scores.
    outpath : Pathlib object
        Location of directory in which output is saved.

    """
    time_start = time()

    optp_.estrisk_dict['estrisk_used'] = True

    if optp_.gen_dict['with_output']:
        print_dic_values_all_optp(optp_, summary_top=True,
                                  summary_dic=False,
                                  title='Transform scores for estimation risk',
                                  stage='Estrisk')

    optp_.report['estriskscores'] = True
    # Check if variables are available and have correct lengths
    op_data.check_data_estrisk(optp_, data_df)
    # Transform scores
    (data_estrisk_df, estrisk_scores_names
     ) = op_estrisk.adjust_scores_for_estimation_risk(optp_, data_df)

    # Timing
    time_name = ['Time for estimation risk adjustment of scores:    ',]
    time_difference = [time() - time_start]
    if optp_.gen_dict['with_output']:
        time_str = mcf_ps.print_timing(
            optp_.gen_dict, 'Estimation risk correction ', time_name,
            time_difference, summary=True)
    else:
        time_str = ''

    key = 'estimation risk adjustment ' + data_title
    optp_.time_strings[key] = time_str

    return data_estrisk_df, estrisk_scores_names, optp_.gen_dict['outpath']


def solvefair_method(optp_: 'OptimalPolicy',
                     data_df: pd.DataFrame,
                     data_title: str = ''
                     ) -> tuple[pd.DataFrame, dict, Path]:
    """Compute fairness allocations following method suggested by BLMM2025."""
    # Step 1: General overhead - fairness dic in reporting
    # Step 2: Transform scores if needed
    # Step 3: Transform data if needed
    # Step 4: use solve method with transformed data
    # Step 5: if policy tree: transform to tree to become explainable again
    #         if other methods: fair-transform variables in the allocate method
    #                           (in the same way as for the solve method)

    # Step 1: General overhead & fairness dic in reporting
    time_start = time()
    # references, no copies
    optp_.fair_dict['solvefair_used'] = True
    optp_.report['solvefair'] = True

    if optp_.gen_dict['with_output']:
        print_dic_values_all_optp(optp_,
                                  summary_top=True,
                                  summary_dic=False,
                                  title='Training Fairness',
                                  stage='Fairness'
                                  )
    # Check if data are available, recode features
    data_new_df = op_data.prepare_data_fair(optp_, data_df)
    time0 = time()

    # Step 2: Transform scores if needed
    if optp_.fair_dict['adjust_target'] in ('scores', 'scores_xvariables',):
        (data_fair_df, scores_fair_names, _,
         optp_.fair_dict, optp_.var_dict, optp_.report['fairscores_build_stats']
         ) = op_fair.adjust_scores(deepcopy(optp_.fair_dict),
                                   deepcopy(optp_.gen_dict),
                                   deepcopy(optp_.var_dict),
                                   data_new_df,
                                   seed=1234567
                                   )
    else:    # No adjustment of scores
        data_fair_df = data_new_df
        scores_fair_names = optp_.var_dict['polscore_name'].copy()
    time1 = time()

    # Step 3: Transform data if needed
    if optp_.fair_dict['adjust_target'] in ('xvariables', 'scores_xvariables',):
        (data_fair_df, x_fair_ord_name, x_fair_unord_name, _,
         optp_.fair_dict, optp_.report['fair_decision_vars_build_stats']
         ) = op_fair.adjust_decision_variables(deepcopy(optp_.fair_dict),
                                               deepcopy(optp_.gen_dict),
                                               deepcopy(optp_.var_dict),
                                               data_fair_df,
                                               training=True,
                                               seed=1234567
                                               )

    else:  # No adjustment of variables
        x_fair_ord_name = (optp_.var_dict['x_ord_name'].copy()
                           if optp_.var_dict['x_ord_name'] else [])
        x_fair_unord_name = (optp_.var_dict['x_unord_name'].copy()
                             if optp_.var_dict['x_unord_name'] else [])
    time2 = time()

    # Step 4: use solve method with transformed data
    (optp_.fair_dict['org_name_dict'], optp_.var_dict
     ) = op_fair.update_names_for_solve_in_self(optp_.var_dict,
                                                scores_fair_names,
                                                x_fair_ord_name,
                                                x_fair_unord_name
                                                )

    solve_dict = optp_.solve(
        data_fair_df.copy(),
        data_title=f'training fair ({optp_.fair_dict["adjust_target"]})'
        )
    allocation_df = solve_dict['allocation_df']
    result_dic = solve_dict['result_dic']
    time3 = time()

    # Timing
    time_name = [
        'Time for fairness data consistency checks:      ',
        'Time for fairness score transformation:         ',
        'Time for fairness variable transformations:     ',
        'Time for building policy tree with fair inputs: ',
        'Time for fairness-related computations:         ',]
    time_difference = [time0 - time_start,
                       time1 - time0,
                       time2 - time1,
                       time() - time3,
                       time() - time_start,
                       ]
    if optp_.gen_dict['with_output']:
        time_str = mcf_ps.print_timing(
            optp_.gen_dict, 'Fairness specific computations ', time_name,
            time_difference, summary=True)
    else:
        time_str = ''
    key = 'Fairness adjustments ' + data_title
    optp_.time_strings[key] = time_str

    return allocation_df, result_dic, optp_.gen_dict['outpath']


def solve_method(optp_: 'OptimalPolicy',
                 data_df: pd.DataFrame,
                 data_title: str = ''
                 ) -> tuple[pd.DataFrame, dict, Path]:
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

    result_dic : Dictionary
        Contains additional information about trained allocation rule. Only
        complete when keyword _int_with_output is True.

    tests_dict : Dictionary
        Tests for consistency of different fairness adjustments. Empty when
        keyword fair_consistency_test is False.

    outpath : Path
        Location of directory in which output is saved.

    """
    time_start = time()

    optp_.report['training'] = True
    optp_.report['training_data_chcksm'] = op_data.dataframe_checksum(
        data_df)

    op_init.init_gen_solve(optp_, data_df)
    op_init.init_other_solve(optp_)
    result_dic = {}
    method = optp_.gen_dict['method']
    if method in ('policy_tree', 'policy tree old'):
        op_init.init_pt_solve(optp_, len(data_df))
    if optp_.gen_dict['with_output']:
        print_dic_values_all_optp(optp_, summary_top=True,
                                  summary_dic=False, title='Training',
                                  stage='Training')
    allocation_df = allocation_txt = None

    if method in ('policy_tree', 'policy tree old', 'best_policy_score',
                  'bps_classifier'):
        (data_new_df, bb_rest_variable) = op_data.prepare_data_bb_pt(
            optp_, data_df)
        if method in ('best_policy_score', 'bps_classifier'):
            allocation_df = op_bb.black_box_allocation(
                optp_, data_new_df, bb_rest_variable, seed=234356)
            if method == 'bps_classifier':
                (allocation_df, result_dic['bps_classifier_info_dic'],
                 text_report) = op_bb_cl.bps_classifier_allocation(
                     optp_, data_new_df, allocation_df, seed=234356)
                optp_.report['training_classifier'] = text_report
        elif method in ('policy_tree', 'policy tree old'):
            (allocation_df, allocation_txt, result_dic['tree_info_dic']
             ) = op_pt.policy_tree_allocation(optp_, data_new_df)
    else:
        raise ValueError('Specified method for Optimal Policy is not valid.'
                         )

    # Timing
    time_name = [f'Time for {method:20} training:    ',]
    time_difference = [time() - time_start]
    if optp_.gen_dict['with_output']:
        time_str = mcf_ps.print_timing(
            optp_.gen_dict, f'{method:20} Training ', time_name,
            time_difference, summary=True)
    else:
        time_str = ''
    key = f'{method} training ' + data_title
    optp_.time_strings[key] = time_str

    if not ((method == 'best_policy_score')
            and (data_title == 'Prediction data')):
        optp_.report['training_alloc_chcksm'] = op_data.dataframe_checksum(
            allocation_df)
    if allocation_txt is None:
        optp_.report['training_leaf_information'] = None
    else:
        txt = '\n' if data_title == '' else (
            f' (using data from {data_title})\n')
        optp_.report['training_leaf_information'] = txt + allocation_txt

    return allocation_df, result_dic, optp_.gen_dict['outpath']


def allocate_method(optp_: 'OptimalPolicy',
                    data_df: pd.DataFrame,
                    data_title: str = '',
                    fair_adjust_decision_vars: bool = False
                    ) -> tuple[pd.DataFrame, Path]:
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
    allocation_df : DataFrame
        data_df with optimal allocation appended.

    outpath : Path
        Location of directory in which output is saved.

    """
    if not isinstance(fair_adjust_decision_vars, bool):
        fair_adjust_decision_vars = False

    time_start = time()
    method = optp_.gen_dict['method']
    optp_.report['allocation'] = True
    data_train = (op_data.dataframe_checksum(data_df)
                  == optp_.report['training_data_chcksm']
                  )
    data_df.reset_index(drop=True, inplace=True)

    if method == 'policy_tree':
        method_str = 'Policy Tree'
    elif method == 'best_policy_score':
        method_str = 'Best Policy Score'
    elif method == 'bps_classifier':
        method_str = 'Classifier for Best Policy Score Allocation'
    else:
        method_str = ''

    optp_.report['txt'] = ('\nAllocation of unit to treatments using '
                           f'{method_str}.'
                           '\nTraining data '
                           f'{"is NOT" if data_train else "is"} used.'
                           )

    if optp_.gen_dict['with_output']:
        if optp_.fair_dict['solvefair_used']:
            stage, title = 'Fairness', 'Allocation - Fairness'
        else:
            stage, title = 'Allocation', 'Allocation'
        if optp_.estrisk_dict['estrisk_used']:
            title += ' Estimation risk adjustment'
        print_dic_values_all_optp(optp_,
                                  summary_top=True, summary_dic=False,
                                  title=title, stage=stage
                                  )

    if optp_.fair_dict['solvefair_used']:
        data_df, optp_.fair_dict, txt = op_fair.fair_adjust_data_for_pred(
            optp_.fair_dict, optp_.gen_dict, optp_.var_dict,
            data_df, fair_adjust_decision_vars
            )
        optp_.report['txt'] += txt

    allocation_df = allocation_txt = None
    if method == 'best_policy_score':
        solve_dict = optp_.solve(
            data_df, data_title='Prediction data')
        allocation_df = solve_dict['allocation_df']
    elif method in ('policy_tree', 'policy tree old'):
        allocation_df, allocation_txt = op_pt.policy_tree_prediction_only(
            optp_, data_df
            )
    elif method == 'bps_classifier':
        allocation_df, allocation_txt = op_bb_cl.bps_class_prediction_only(
            optp_, data_df
            )
    time_name = [f'Time for {method:20} allocation:  ',]
    time_difference = [time() - time_start]
    if optp_.gen_dict['with_output']:
        time_str = mcf_ps.print_timing(
            optp_.gen_dict, f'{method:20} Allocation ',
            time_name, time_difference, summary=True
            )
    else:
        time_str = ''
    key = f'{method} allocation ' + data_title
    optp_.time_strings[key] = time_str
    optp_.report['alloc_list'].append(optp_.report['txt'])
    if allocation_txt is None:
        optp_.report['leaf_information_allocate'] = None
    else:
        optp_.report['leaf_information_allocate'] = (
            data_title + '\n' + allocation_txt)

    return allocation_df, optp_.gen_dict['outpath']


def evaluate_method(optp_: 'OptimalPolicy',
                    allocation_df: pd.DataFrame,
                    data_df: pd.DataFrame,
                    data_title: str = '',
                    seed: int = 12434
                    ) -> tuple[dict, Path]:
    """
    Evaluate allocation with potential outcome data.

    Parameters
    ----------
    allocation_df : DataFrame
        Optimal allocation as outputed by the :meth:`~OptimalPolicy.solve`
        and :meth:`~OptimalPolicy.allocate` methods.
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

    outpath : Path
        Location of directory in which output is saved.

    """
    time_start = time()

    optp_.report['evaluation'] = True
    alloc_train = (op_data.dataframe_checksum(allocation_df)
                   == optp_.report['training_alloc_chcksm'])

    if optp_.gen_dict['with_output']:
        if optp_.fair_dict['solvefair_used']:
            stage, title = 'Fairness', 'Evaluation - Fairness'
        else:
            stage, title = 'Evaluation', 'Evaluation'
        if optp_.estrisk_dict['estrisk_used']:
            title += ' Estimation risk adjustment'
        if alloc_train:
            title += ' Training data'
        else:
            title += ' Prediction data'
        print_dic_values_all_optp(optp_, summary_top=True,
                                  summary_dic=False, title=title, stage=stage
                                  )

    var_dic, gen_dic = optp_.var_dict, optp_.gen_dict
    txt = '\n' + '=' * 100 + '\nEvaluating allocation of '
    txt += f'{gen_dic["method"]} with {data_title} data\n' + '-' * 100
    mcf_ps.print_mcf(gen_dic, txt, summary=True)
    (data_df, d_ok, polscore_ok, polscore_desc_ok, desc_var
     ) = op_data.prepare_data_eval(optp_, data_df)
    if len(allocation_df) != len(data_df):
        d_ok = False
    op_init.init_rnd_shares(optp_, data_df, d_ok)
    if d_ok:
        allocation_df['observed'] = data_df[var_dic['d_name']]
    allocation_df['random'] = op_eval.get_random_allocation(optp_,
                                                            len(data_df),
                                                            seed
                                                            )
    if polscore_ok:
        allocation_df['best ATE'] = op_eval.get_best_ate_allocation(optp_,
                                                                    data_df
                                                                    )
    results_dic = op_eval.evaluate_fct(optp_,
                                       data_df, allocation_df,
                                       d_ok, polscore_ok, polscore_desc_ok,
                                       desc_var, data_title
                                       )
    if (optp_.gen_dict['with_output']
            and optp_.gen_dict['variable_importance']):
        op_eval.variable_importance(optp_,
                                    data_df, allocation_df,
                                    seed=seed, data_title=data_title
                                    )
    time_name = [f'Time for Evaluation with {data_title} data:     ',]
    time_difference = [time() - time_start]
    if optp_.gen_dict['with_output']:
        time_str = mcf_ps.print_timing(
            optp_.gen_dict, f'Evaluation of {data_title} data with '
            f'{gen_dic["method"]}', time_name, time_difference,
            summary=True)
    else:
        time_str = ''
    key = 'evaluate_' + data_title
    optp_.time_strings[key] = time_str

    train_str = 'the SAME as' if alloc_train else 'DIFFERENT from'
    rep_txt = (f'Allocation analysed is {train_str} the one obtained '
               f'from the training data ({data_title}). '
               f'{"bb"} stands for {"black box"}. '
               )
    if optp_.estrisk_dict['estrisk_used']:
        rep_txt += ('If a value function is shown, it refers to the mean value '
                    'of the allocation adjusted for estimation risk.')

    optp_.report['evalu_list'].append((rep_txt, results_dic))

    return results_dic, optp_.gen_dict['outpath']


def evaluate_multiple_self(optp_: 'OptimalPolicy',
                           allocations_dic: dict,
                           data_df: pd.DataFrame
                           ) -> Path:
    """
    Evaluate several allocations simultaneously.

    Parameters
    ----------
    allocations_dic : Dictionary.
        Contains dataframes with specific allocations.
    data_df : DataFrame.
        Data with the relevant information about potential outcomes which
        will be used to evaluate the allocations.

    Returns
    -------
    outpath : Path
        Location of directory in which output is saved.

    """
    if not optp_.gen_dict['with_output']:
        raise ValueError('To use this method, allow output to be written.')
    potential_outcomes_np = data_df[optp_.var_dict['polscore_name']]
    op_eval.evaluate_multiple(optp_, allocations_dic, potential_outcomes_np)

    return optp_.gen_dict['outpath']


def print_dic_values_all_optp(optp_: 'OptimalPolicy',
                              summary_top: bool = True,
                              summary_dic: bool = False,
                              title: str = '',
                              stage: str = ''
                              ) -> None:
    """Print the dictionaries."""
    txt = '=' * 100 + f'\nOptimal Policy Module ({title}) with '
    txt += f'{optp_.gen_dict["method"]}' + '\n' + '-' * 100
    mcf_ps.print_mcf(optp_.gen_dict, txt, summary=summary_top)
    print_dic_values_optp(optp_, summary=summary_dic, stage=stage)


def print_dic_values_optp(optp_: 'OptimalPolicy',
                          summary: bool = False,
                          stage: bool = None
                          ) -> None:
    """Print values of dictionaries that determine module."""
    if stage == 'Estrisk':
        # Rest of info comes via a subsequent call to solve or solvefair
        dic_list = [optp_.estrisk_dict,]
        dic_name_list = ['estrisk_dict',]
    else:
        dic_list = [optp_.int_dict, optp_.gen_dict, optp_.dc_dict,
                    optp_.other_dict, optp_.rnd_dict, optp_.var_dict]
        dic_name_list = ['int_dict', 'gen_dict', 'dc_dict',
                         'other_dict', 'rnd_dict', 'var_dict']
        if optp_.gen_dict['method'] in ('policy_tree', 'policy tree old'):
            add_list = [optp_.var_x_type, optp_.var_x_values, optp_.pt_dict]
            add_list_name = ['var_x_type', 'var_x_values', 'pt_dict']
            dic_list.extend(add_list)
            dic_name_list.extend(add_list_name)
    if stage == 'Fairness':
        dic_list.append(optp_.fair_dict)
        dic_name_list.append('fair_dict')
    for dic, dic_name in zip(dic_list, dic_name_list):
        mcf_ps.print_dic(dic, dic_name, optp_.gen_dict, summary=summary)
    mcf_ps.print_mcf(optp_.gen_dict, '\n', summary=summary)


def winners_losers_method(optp_org: 'OptimalPolicy',
                          data_df: pd.DataFrame,
                          welfare_df: pd.DataFrame,
                          welfare_reference_df: pd.DataFrame | None = None,
                          outpath: Path | None = None,
                          title: str = '',
                          ) -> tuple[pd.DataFrame, Path]:
    """Describe winners and losers from different allocations."""
    optp_ = deepcopy(optp_org)

    if outpath is not None:  # Otherwise, outpath from self is taken
        if not isinstance(outpath, Path):
            outpath = Path(outpath)
        optp_.gen_dict['outpath'] = outpath
        optp_.gen_dict['outfiletext'] = (
            outpath / optp_org.gen_dict['outfiletext'].name
            )
        optp_.gen_dict['outfilesummary'] = (
            outpath / optp_org.gen_dict['outfilesummary'].name
            )

    optp_.gen_dict['print_to_file'] = True
    # Define relevant variables for descriptive stats
    variables_to_desc = optp_.var_dict['polscore_name']
    variables_to_desc.extend(optp_.var_dict['x_ord_name'])
    variables_to_desc.extend(optp_.var_dict['x_unord_name'])
    variables_to_desc.extend(optp_.var_dict['protected_ord_name'])
    variables_to_desc.extend(optp_.var_dict['protected_unord_name'])
    variables_to_desc.extend(optp_.var_dict['material_ord_name'])
    variables_to_desc.extend(optp_.var_dict['material_unord_name'])

    names_unordered = optp_.var_dict['x_unord_name']
    names_unordered.extend(optp_.var_dict['x_unord_name'])
    names_unordered.extend(optp_.var_dict['protected_unord_name'])
    names_unordered.extend(optp_.var_dict['material_unord_name'])

    variables_to_desc = mcf_gp.remove_dupl_keep_order(variables_to_desc)
    names_unordered = mcf_gp.remove_dupl_keep_order(names_unordered)
    # Take only those variable that are contained in data
    var_desc = [name for name in variables_to_desc if name in data_df.columns]

    # Check if welfare input is ok and convert to numpy
    if welfare_reference_df is None or len(welfare_reference_df) < len(data_df):
        welfare_ref_np = np.zeros(len(data_df), 1)
        name_welfare_ref = ('zero',)
        welfare_reference_df = pd.DataFrame(welfare_ref_np,
                                            columns=name_welfare_ref
                                            )
    else:
        welfare_ref_np = welfare_reference_df.to_numpy()
        if welfare_ref_np.ndim > 1 and welfare_ref_np.shape[1] > 1:
            raise ValueError('Referenz welfare contains more than 1 column.')
        if welfare_ref_np.ndim == 1:
            welfare_ref_np = welfare_ref_np.reshape(-1, 1)
        name_welfare_ref = welfare_reference_df.columns

    if not isinstance(welfare_df, pd.DataFrame):
        raise ValueError('Actual welfare variable must be DataFrame.')

    welfare_np = welfare_df.to_numpy()
    if welfare_np.ndim > 1 and welfare_np.shape[1] > 1:
        raise ValueError('Welfare contains more than 1 column.')
    if welfare_np.ndim == 1:
        welfare_np = welfare_np.reshape(-1, 1)
    name_welfare = welfare_df.columns

    # Add welfare to data and to the variables used to describe data
    data_to_desc_df = pd.concat((data_df[var_desc],
                                 welfare_df,
                                 welfare_reference_df,
                                 ),
                                axis=1)
    variables_to_desc = data_to_desc_df.columns

    welfare_diff_np = welfare_np - welfare_ref_np
    name_diff = name_welfare[0] + '_minus_' + name_welfare_ref[0]
    welfare_cluster_id_name = 'Welfare_change_cluster_' + name_diff

    if len(np.unique(welfare_diff_np)) < 2:  # Not enough variation
        data_df[welfare_cluster_id_name] = np.zero((len(data_df), 1,))

        return data_df, optp_.gen_dict['outpath']

    welfare_diff_df = pd.DataFrame(welfare_diff_np, columns=[name_diff])
    # kmeans clustering
    km_dic = {
        'kmeans_max_tries': 1000,
        'kmeans_replications': 10,
        'kmeans_no_of_groups': [2, 3, 4, 5, 6, 7, 8],
        'kmeans_min_size_share': 1,
        }

    silhouette_avg_prev, cluster_lab_np = -1, None
    txt = ('\n' + '=' * 100 + '\nK-Means++ clustering: ' + title + '\n'
           + name_diff)
    txt += '\n' + '-' * 100
    for cluster_no in km_dic['kmeans_no_of_groups']:
        (cluster_lab_tmp, silhouette_avg, merge) = kmeans_labels(
            welfare_diff_df, km_dic, cluster_no)
        txt += (f'\nNumber of clusters: {cluster_no}   '
                f'Average silhouette score: {silhouette_avg: 8.3f}')
        if merge:
            txt += (' Smallest cluster has too few observations. It was '
                    'merged with with cluster with closest centroid.'
                    )
        if silhouette_avg > silhouette_avg_prev:
            cluster_lab_np = np.copy(cluster_lab_tmp)
            silhouette_avg_prev = np.copy(silhouette_avg)
    txt += ('\n\nBest value of average silhouette score:'
            f' {silhouette_avg_prev: 8.3f}')

    # Reorder labels for better visible inspection of results
    cl_means = welfare_diff_df.groupby(by=cluster_lab_np).mean(numeric_only=True
                                                               )
    cl_means_np = cl_means.to_numpy()
    cl_means_np = np.mean(cl_means_np, axis=1)
    sort_ind = np.argsort(cl_means_np)
    cl_group = cluster_lab_np.copy()
    for cl_j, cl_old in enumerate(sort_ind):
        cl_group[cluster_lab_np == cl_old] = cl_j
    txt += ('\n' + '- ' * 50 +
            '\nWelfare changes are ordered w.r.t. to their size'
            )
    cl_values, cl_obs = np.unique(cl_group, return_counts=True)
    txt += '\n' + '-' * 100 + '\nNumber of observations in the clusters'

    txt += '\n' + '- ' * 50
    for idx, val in enumerate(cl_values):
        string = f'\nCluster {val:2}: {cl_obs[idx]:6} '
        txt += string

    txt += '\n' + '-' * 100 + '\nWelfare changes\n' + '- ' * 50

    cl_means = welfare_diff_df.groupby(by=cl_group).mean(numeric_only=True)
    txt += '\n' + cl_means.transpose().to_string()
    txt += '\n' + '-' * 100 + '\nFeatures\n' + '- ' * 50

    if names_unordered:  # List is not empty
        x_dummies_df = pd.get_dummies(data_to_desc_df, columns=names_unordered,
                                      dtype=int)
        x_km = pd.concat([data_to_desc_df, x_dummies_df], axis=1)
    else:
        x_km = data_to_desc_df
    cl_means = x_km.groupby(by=cl_group).mean(numeric_only=True)

    mcf_ps.print_mcf(optp_.gen_dict, txt, summary=True)

    txt = cl_means.transpose().to_string() + '\n' + '-' * 100
    pd.set_option('display.max_rows', 1000, 'display.max_columns', 100)
    mcf_ps.print_mcf(optp_.gen_dict, txt, summary=True)
    pd.set_option('display.max_rows', None, 'display.max_columns', None)

    # Return clusternumber together with data
    data_df[welfare_cluster_id_name] = cl_group.reshape(-1, 1)

    return data_df, optp_.gen_dict['outpath']
