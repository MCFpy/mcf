"""
Methods for optimal policy analysis.

Created on Thu Aug  3 15:23:17 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
from __future__ import annotations
from copy import copy, deepcopy
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats as mcf_ps
from mcf import optpolicy_data as op_data
from mcf import optpolicy_evaluation as op_eval
from mcf import optpolicy_fair as op_fair
from mcf import optpolicy_init as op_init
from mcf.mcf_post import kmeans_labels
from mcf.mcf_feature_selection import feature_selection_optpol
from mcf.optpolicy_bb import black_box_allocation
from mcf.optpolicy_bb_cl import bps_classifier_allocation, bps_class_prediction_only
from mcf.optpolicy_estrisk import (adjust_scores_for_estimation_risk,
                                   adjust_scores_for_estimation_risk_prediction,
                                   )
from mcf.optpolicy_pt import policy_tree_allocation, policy_tree_prediction_only
from mcf.optpolicy_print import print_dic_values_all_optp_version
from mcf.optpolicy_tree_plot import save_policy_tree_figure

if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicy, OptimalPolicyVersions


def estrisk_adjust_method(optp_: 'OptimalPolicy',
                          data_df: pd.DataFrame,
                          data_title: str = '',
                          ) -> tuple[pd.DataFrame, list[str], Path | None]:
    """Adjust the original policy scores and commit state after a successful call."""
    time_start = time()
    original = optp_
    optp_ = copy(original)
    optp_.var_cfg = deepcopy(original.var_cfg)
    optp_.estriskcfg = deepcopy(original.estriskcfg)
    optp_.report = original.report.copy()
    optp_.time_strings = original.time_strings.copy()

    if optp_.estriskcfg.var_cfg_original is None:
        optp_.estriskcfg.var_cfg_original = deepcopy(optp_.var_cfg)

    if optp_.estriskcfg.polscore_name_org is not None:
        optp_.var_cfg.polscore_name = optp_.estriskcfg.polscore_name_org.copy()

    data_df = data_df.rename(columns=str.casefold)
    op_data.check_data_estrisk(optp_, data_df)
    adjusted_names = [name + '_estrisk' for name in optp_.var_cfg.polscore_name]
    source_names = [*optp_.var_cfg.polscore_name, *optp_.var_cfg.polscore_se_name]
    conflicts = set(adjusted_names).intersection(source_names)
    if conflicts:
        raise ValueError('Adjusted score names conflict with original score or standard-error '
                         f'columns: {", ".join(sorted(conflicts))}. Rename these input columns.'
                         )
    op_data.check_generated_names(optp_.estriskcfg.var_cfg_original, adjusted_names)
    data_df = data_df.drop(columns=adjusted_names, errors='ignore')
    optp_.estriskcfg.estrisk_used = True
    optp_.report['estriskscores'] = True

    if optp_.gen_cfg.with_output:
        print_dic_values_all_optp(optp_,
                                  summary_top=True,
                                  summary_dic=False,
                                  title='Transform scores for estimation risk',
                                  stage='Estrisk',
                                  )
    data_estrisk_df, estrisk_scores_names = adjust_scores_for_estimation_risk(optp_, data_df)

    time_name = ['Time for estimation risk adjustment of scores:    ',]
    time_difference = [time() - time_start]
    if optp_.gen_cfg.with_output:
        time_str = mcf_ps.print_timing(optp_.gen_cfg, 'Estimation risk correction ', time_name,
                                       time_difference, summary=True
                                       )
    else:
        time_str = ''
    key = 'estimation risk adjustment ' + data_title
    optp_.time_strings[key] = time_str

    original.var_cfg = optp_.var_cfg
    original.estriskcfg = optp_.estriskcfg
    original.report = optp_.report
    original.time_strings = optp_.time_strings

    return data_estrisk_df, estrisk_scores_names, optp_.gen_cfg.outpath


def solvefair_method(optp_: 'OptimalPolicy',
                     data_df: pd.DataFrame,
                     data_title: str = '',
                     ) -> tuple[pd.DataFrame, dict, Path | None]:
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
    if optp_.fs_cfg.yes:
        raise NotImplementedError('Automatic feature selection is not implemented for '
                                  'fairness adjustments.'
                                  )

    fair_cfg = optp_.fair_cfg
    if fair_cfg.var_cfg_original is None:
        fair_cfg.var_cfg_original = deepcopy(optp_.var_cfg)
    else:
        optp_.var_cfg = deepcopy(fair_cfg.var_cfg_original)
    if fair_cfg.settings_original is None:
        fair_cfg.settings_original = {
            name: getattr(fair_cfg, name)
            for name in ('adj_type', 'protected_disc_method', 'material_disc_method')
            }
    for name, value in fair_cfg.settings_original.items():
        setattr(fair_cfg, name, value)
    for name in ('decision_vars_fair_org_name', 'x_ord_org_name', 'fair_strata',
                  'decision_vars_org_df', 'data_train_for_pred_df', 'protected_matrel',
                  'org_name_dict'):
        setattr(fair_cfg, name, None)
    fair_cfg.solvefair_used = True
    optp_.report['solvefair'] = True

    if optp_.gen_cfg.with_output:
        print_dic_values_all_optp(optp_,
                                  summary_top=True, summary_dic=False,
                                  title='Training Fairness', stage='Fairness'
                                  )
    # Check if data are available, recode features
    data_new_df = op_data.prepare_data_fair(optp_, data_df)
    if fair_cfg.adjust_target == 'scores_xvariables':
        shared_names = set(optp_.var_cfg.polscore_name).intersection(
            [*optp_.var_cfg.x_ord_name, *optp_.var_cfg.x_unord_name])
        if shared_names:
            raise ValueError('Score and decision-variable adjustments would share output '
                             f'columns for: {", ".join(sorted(shared_names))}. '
                             'Use separately named input columns for these roles.'
                             )
    adjustment_names = []
    if fair_cfg.adjust_target in ('scores', 'scores_xvariables'):
        adjustment_names.extend(optp_.var_cfg.polscore_name)
    if fair_cfg.adjust_target in ('xvariables', 'scores_xvariables'):
        adjustment_names.extend(optp_.var_cfg.x_ord_name)
        adjustment_names.extend(optp_.var_cfg.x_unord_name)
    op_data.check_generated_names(optp_.var_cfg,
                                   [name + '_fair' for name in adjustment_names])
    score_names_org = optp_.var_cfg.polscore_name.copy()
    time0 = time()

    # Step 2: Transform scores if needed
    if optp_.fair_cfg.adjust_target in ('scores', 'scores_xvariables',):
        (data_fair_df, scores_fair_names, _,
         optp_.fair_cfg, optp_.var_cfg, optp_.report['fairscores_build_stats']
         ) = op_fair.adjust_scores(deepcopy(optp_.fair_cfg), deepcopy(optp_.gen_cfg),
                                   deepcopy(optp_.var_cfg),
                                   data_new_df,
                                   seed=1234567
                                   )
    else:    # No adjustment of scores
        data_fair_df = data_new_df
        scores_fair_names = optp_.var_cfg.polscore_name.copy()
    time1 = time()

    # Step 3: Transform data if needed
    if optp_.fair_cfg.adjust_target in ('xvariables', 'scores_xvariables',):
        (data_fair_df, x_fair_ord_name, x_fair_unord_name, _,
         optp_.fair_cfg, optp_.report['fair_decision_vars_build_stats']
         ) = op_fair.adjust_decision_variables(deepcopy(optp_.fair_cfg),
                                               deepcopy(optp_.gen_cfg),
                                               deepcopy(optp_.var_cfg),
                                               data_fair_df,
                                               training=True,
                                               seed=1234567
                                               )
    else:  # No adjustment of variables
        x_fair_ord_name = optp_.var_cfg.x_ord_name.copy() if optp_.var_cfg.x_ord_name else []
        x_fair_unord_name = optp_.var_cfg.x_unord_name.copy() if optp_.var_cfg.x_unord_name else []

    time2 = time()

    # Step 4: use solve method with transformed data
    (optp_.fair_cfg.org_name_dict, optp_.var_cfg
     ) = op_fair.update_names_for_solve_in_self(optp_.var_cfg,
                                                scores_fair_names,
                                                x_fair_ord_name,
                                                x_fair_unord_name
                                                )
    optp_.fair_cfg.org_name_dict['polscore_name'] = score_names_org

    solve_title = f'training fair ({optp_.fair_cfg.adjust_target})'
    if data_title:
        solve_title += f': {data_title}'
    solve_dict, _ = optp_.solve(data_fair_df.copy(), data_title=solve_title)

    allocation_df = solve_dict['allocation_df']
    result_dic = solve_dict['result_dic']
    time3 = time()

    # Timing
    time_name = ['Time for fairness data consistency checks:      ',
                 'Time for fairness score transformation:         ',
                 'Time for fairness variable transformations:     ',
                 'Time for building policy tree with fair inputs: ',
                 'Time for fairness-related computations:         ',
                 ]
    time_difference = [time0 - time_start,
                       time1 - time0,
                       time2 - time1,
                       time3 - time2,
                       time() - time_start,
                       ]
    if optp_.gen_cfg.with_output:
        time_str = mcf_ps.print_timing(optp_.gen_cfg, 'Fairness specific computations ', time_name,
                                       time_difference, summary=True
                                       )
    else:
        time_str = ''
    key = 'Fairness adjustments ' + data_title
    optp_.time_strings[key] = time_str

    return allocation_df, result_dic, optp_.gen_cfg.outpath


def solve_method(optp_: 'OptimalPolicy',
                 data_df: pd.DataFrame,
                 data_title: str = '',
                 ) -> tuple[pd.DataFrame, dict, Path, pd.DataFrame]:
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
    
    data_return: DataFrame
        Training data without the observations used for feature selection.

    """
    time_start = time()

    optp_.report['training'] = True
    optp_.report['training_tree_figure'] = None
    optp_.report['training_data_chcksm'] = op_data.dataframe_checksum(data_df)
    data_df = data_df.reset_index(drop=True).rename(columns=str.casefold)

    op_init.init_gen_solve(optp_, data_df)
    result_dic = {}
    method = optp_.gen_cfg.method
    allocation_df = allocation_txt = None

    if method not in ('policy_tree', 'policy tree old', 'best_policy_score', 'bps_classifier',):
        raise ValueError('Specified method for Optimal Policy is not valid.')

    # Data preparation
    data_new_df, bb_rest_variable = op_data.prepare_data_bb_pt(optp_, data_df)

    # Feature selection
    if optp_.fs_cfg.yes and method in ('policy_tree', 'policy tree old', 'bps_classifier',):
        d_df = feature_selection_optpol(optp_, data_new_df)  # Instance to be changed inside
        # Select elements from data data that are not used for feature selection, reindex.
        data_return_df = data_df[data_df.index.isin(d_df.index)].sort_index().reset_index(drop=True)
        data_new_df = d_df.sort_index().reset_index(drop=True)
        d_df = None
    else:
        data_return_df = data_df.loc[data_new_df.index].reset_index(drop=True)
        data_new_df = data_new_df.reset_index(drop=True)

    no_obs_train = len(data_new_df)
    op_init.init_other_solve(optp_, no_obs_train)

    if method in ('policy_tree', 'policy tree old'):
        op_init.init_pt_solve(optp_, no_obs_train)

    if optp_.gen_cfg.with_output:
        print_dic_values_all_optp(optp_,
                                  summary_top=True,
                                  summary_dic=False,
                                  title='Training',
                                  stage='Training',
                                  )
    # Find allocation rules
    if method in ('best_policy_score', 'bps_classifier'):
        allocation_df = black_box_allocation(optp_, data_new_df, bb_rest_variable, seed=234356)
        if method == 'bps_classifier':
            (allocation_df, result_dic['bps_classifier_info_dic'], text_report
             ) = bps_classifier_allocation(optp_, data_new_df, allocation_df, seed=234356)
            optp_.report['training_classifier'] = text_report
    elif method in ('policy_tree', 'policy tree old'):
        (allocation_df, allocation_txt, result_dic['tree_info_dic']
         ) = policy_tree_allocation(optp_, data_new_df)
        tree_figure = save_policy_tree_figure(optp_, data_new_df, data_title=data_title)
        optp_.report['training_tree_figure'] = tree_figure

    # Timing
    time_name = [f'Time for {method:20} training:    ',]
    time_difference = [time() - time_start]
    if optp_.gen_cfg.with_output:
        time_str = mcf_ps.print_timing(
            optp_.gen_cfg, f'{method:20} Training ', time_name, time_difference, summary=True)
    else:
        time_str = ''
    key = f'{method} training ' + data_title
    optp_.time_strings[key] = time_str

    if not ((method == 'best_policy_score')
            and (data_title == 'Prediction data')):
        optp_.report['training_alloc_chcksm'] = op_data.dataframe_checksum(allocation_df)
    if allocation_txt is None:
        optp_.report['training_leaf_information'] = None
    else:
        txt = '\n' if data_title == '' else (
            f' (using data from {data_title})\n')
        optp_.report['training_leaf_information'] = txt + allocation_txt

    return allocation_df, result_dic, optp_.gen_cfg.outpath, data_return_df


def allocate_method(optp_: 'OptimalPolicy',
                    data_df: pd.DataFrame,
                    data_title: str = '',
                    fair_adjust_decision_vars: bool = False,
                    ) -> tuple[pd.DataFrame, Path | None]:
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
        After solvefair() with decision-variable adjustments, True recomputes those variables.
        False reuses adjusted variables, computing missing ones only when the allocation rule
        needs them. Best-policy-score allocation does not need decision variables.
        On an instance that has not used solvefair(), this option has no effect.
        The default is False.

    Returns
    -------
    allocation_df : DataFrame
        data_df with optimal allocation appended.

    outpath : Path or None
        Location of directory in which output is saved.

    """
    if not isinstance(fair_adjust_decision_vars, bool):
        fair_adjust_decision_vars = False

    time_start = time()
    method = optp_.gen_cfg.method
    optp_.report['allocation'] = True
    data_train = op_data.dataframe_checksum(data_df) == optp_.report['training_data_chcksm']
    data_df = data_df.reset_index(drop=True).rename(columns=str.casefold)

    match method:
        case 'policy_tree':       method_str = 'Policy Tree'
        case 'best_policy_score': method_str = 'Best Policy Score'
        case 'bps_classifier':    method_str = 'Classifier for Best Policy Score Allocation'
        case _:                   method_str = ''

    optp_.report['txt'] = (f'\nAllocation of unit to treatments using {method_str}.'
                           f'\nTraining data {"is" if data_train else "is NOT"} used.'
                           )
    if optp_.gen_cfg.with_output:
        if optp_.fair_cfg.solvefair_used:
            stage, title = 'Fairness', 'Allocation - Fairness'
        else:
            stage, title = 'Allocation', 'Allocation'
        if optp_.estriskcfg.estrisk_used:
            title += ' Estimation risk adjustment'
        print_dic_values_all_optp(optp_,
                                  summary_top=True, summary_dic=False, title=title, stage=stage
                                  )
    if (optp_.estriskcfg.estrisk_used and method == 'best_policy_score'
            and not all(name in data_df.columns
                        for name in optp_.var_cfg.polscore_name)):
        data_df, txt = adjust_scores_for_estimation_risk_prediction(optp_, data_df)
        optp_.report['txt'] += txt

    if optp_.fair_cfg.solvefair_used:
        data_df, optp_.fair_cfg, txt = op_fair.fair_adjust_data_for_pred(
            optp_.fair_cfg,
            optp_.gen_cfg,
            optp_.var_cfg,
            data_df,
            fair_adjust_decision_vars,
            )
        optp_.report['txt'] += txt

    allocation_df = allocation_txt = None
    match method:
        case 'best_policy_score':
            solve_dict, _ = optp_.solve(data_df, data_title='Prediction data')
            allocation_df = solve_dict['allocation_df']
        case 'policy_tree' | 'policy tree old':
            allocation_df, allocation_txt = policy_tree_prediction_only(optp_, data_df)
        case 'bps_classifier':
            allocation_df, allocation_txt = bps_class_prediction_only(optp_, data_df)

    time_name = [f'Time for {method:20} allocation:  ',]
    time_difference = [time() - time_start]
    if optp_.gen_cfg.with_output:
        time_str = mcf_ps.print_timing(optp_.gen_cfg, f'{method:20} Allocation ',
                                       time_name, time_difference, summary=True,
                                       )
    else:
        time_str = ''
    key = f'{method} allocation ' + data_title
    optp_.time_strings[key] = time_str
    optp_.report['alloc_list'].append(optp_.report['txt'])
    if allocation_txt is None:
        optp_.report['leaf_information_allocate'] = None
    else:
        optp_.report['leaf_information_allocate'] = data_title + '\n' + allocation_txt

    return allocation_df, optp_.gen_cfg.outpath


def evaluate_method(optp_: 'OptimalPolicy',
                    allocation_df: pd.DataFrame,
                    data_df: pd.DataFrame, *,
                    data_title: str = '',
                    seed: int = 12434,
                    optp_version: 'OptimalPolicyVersions' | None = None,
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
    alloc_train = op_data.dataframe_checksum(allocation_df) == optp_.report['training_alloc_chcksm']
    allocation_df = allocation_df.copy(deep=False)

    if optp_.gen_cfg.with_output:
        if optp_.fair_cfg.solvefair_used:
            stage, title = 'Fairness', 'Evaluation - Fairness'
        else:
            stage, title = 'Evaluation', 'Evaluation'
        if optp_.estriskcfg.estrisk_used:
            title += ' Estimation risk adjustment'
        match_str = 'matches' if alloc_train else 'differs from'
        title += f' (allocation table {match_str} stored training allocation table)'
        print_dic_values_all_optp(optp_,
                                  summary_top=True, summary_dic=False, title=title, stage=stage,
                                  )
        if optp_version is not None:
            print_dic_values_all_optp_version(optp_version, gen_cfg=optp_.gen_cfg,
                                              summary_dic=False, line_length=100,
                                              )
    var_cfg, gen_cfg = optp_.var_cfg, optp_.gen_cfg
    txt = '\n' + '=' * 100 + '\nEvaluating allocation of '
    txt += f'{gen_cfg.method} with {data_title} data\n' + '-' * 100
    mcf_ps.print_mcf(gen_cfg, txt, summary=True)
    (data_df, d_ok, polscore_ok, polscore_desc_ok, desc_var, polscore_desc_name,
     ) = op_data.prepare_data_eval(optp_, data_df)

    if polscore_ok:
        polscore_name_eval = var_cfg.polscore_name
    elif ((optp_.fair_cfg.solvefair_used or optp_.estriskcfg.estrisk_used)
          and polscore_desc_ok
          and polscore_desc_name
          ):
        polscore_name_eval = polscore_desc_name[-gen_cfg.no_of_treat:]
    else:
        polscore_name_eval = None

    if len(allocation_df) != len(data_df):
        raise ValueError('allocation_df and data_df must contain the same number of observations.')
    reference_names = {'random'}
    if d_ok:
        reference_names.add('observed')
    if polscore_name_eval is not None:
        reference_names.add('best ATE')
    conflicting_names = reference_names.intersection(allocation_df.columns)
    if conflicting_names:
        raise ValueError('Allocation column names conflict with automatic reference allocations: '
                         f'{", ".join(sorted(conflicting_names))}'
                         )
    op_init.init_rnd_shares(optp_, data_df, d_ok)

    if d_ok:
        allocation_df['observed'] = data_df[var_cfg.d_name[0]].to_numpy(copy=False)
    allocation_df['random'] = op_eval.get_random_allocation(optp_, len(data_df), seed)

    if polscore_name_eval is not None:
        allocation_df['best ATE'] = op_eval.get_best_ate_allocation(optp_,
                                                                    data_df,
                                                                    polscore_name_eval,
                                                                    )
    results_dic = op_eval.evaluate_fct(optp_,
                                       data_df,
                                       allocation_df,
                                       d_ok=d_ok,
                                       polscore_ok=polscore_ok,
                                       polscore_desc_ok=polscore_desc_ok,
                                       polscore_desc_name=polscore_desc_name,
                                       desc_var=desc_var,
                                       data_title=data_title,
                                       )
    if optp_.gen_cfg.with_output and optp_.gen_cfg.variable_importance:
        op_eval.variable_importance(optp_,
                                    data_df, allocation_df,
                                    seed=seed, data_title=data_title
                                    )
    time_name = [f'Time for Evaluation with {data_title} data:     ',]
    time_difference = [time() - time_start]
    if optp_.gen_cfg.with_output:
        time_str = mcf_ps.print_timing(optp_.gen_cfg,
                                       f'Evaluation of {data_title} data with {gen_cfg.method}',
                                       time_name, time_difference, summary=True
                                       )
    else:
        time_str = ''
    key = 'evaluate_' + data_title
    optp_.time_strings[key] = time_str

    train_str = 'matches' if alloc_train else 'differs from'
    rep_txt = (f'The allocation table {train_str} the stored training allocation table '
               f'({data_title}). This comparison does not establish whether the evaluation '
               'observations were used for training. bb stands for black box. '
               )
    if polscore_name_eval is not None:
        rep_txt += ('Welfare values are calculated from these evaluation score columns: '
                    f'{", ".join(polscore_name_eval)}. '
                    )
        if not polscore_ok:
            rep_txt += ('The configured policy scores are unavailable in these evaluation data. '
                        'Evaluation uses the last available complete block of descriptive '
                        'policy scores instead. '
                        )
    else:
        rep_txt += 'No policy scores are available, so welfare values are not reported. '

    optp_.report['evalu_list'].append((rep_txt, results_dic))

    return results_dic, optp_.gen_cfg.outpath


def evaluate_multiple_self(optp_: 'OptimalPolicy', allocations_dic: dict, data_df: pd.DataFrame
                           ) -> Path:
    """
    Evaluate several allocations simultaneously.

    Parameters
    ----------
    allocations_dic : Dictionary.
        Contains dataframes with specific allocations.
    data_df : DataFrame.
        Data with the relevant information about potential outcomes which will be used to evaluate
        the allocations.

    Returns
    -------
    outpath : Path
        Location of directory in which output is saved.

    """
    if not optp_.gen_cfg.with_output:
        raise ValueError('To use this method, allow output to be written.')
    (data_eval_df, _, polscore_ok, polscore_desc_ok, _, polscore_desc_name,
     ) = op_data.prepare_data_eval(optp_, data_df)

    if polscore_ok:
        polscore_name_eval = optp_.var_cfg.polscore_name
    elif ((optp_.fair_cfg.solvefair_used or optp_.estriskcfg.estrisk_used)
          and polscore_desc_ok
          and polscore_desc_name
          ):
        polscore_name_eval = polscore_desc_name[-optp_.gen_cfg.no_of_treat:]
    else:
        raise ValueError('No complete policy-score block is available for evaluating allocations.')

    op_eval.evaluate_multiple(optp_, allocations_dic, data_eval_df[polscore_name_eval])

    return optp_.gen_cfg.outpath


def print_dic_values_all_optp(optp_: 'OptimalPolicy',
                              summary_top: bool = True,
                              summary_dic: bool = False,
                              title: str = '',
                              stage: str = '',
                              ) -> None:
    """Print the dataclasses."""
    txt = '=' * 100 + f'\nOptimal Policy Module ({title}) with '
    txt += f'{optp_.gen_cfg.method}' + '\n' + '-' * 100
    mcf_ps.print_mcf(optp_.gen_cfg, txt, summary=summary_top)
    print_dic_values_optp(optp_, summary=summary_dic, stage=stage)


def print_dic_values_optp(optp_: 'OptimalPolicy',
                          summary: bool = False,
                          stage: str | None = None,
                          ) -> None:
    """Print values of dictionaries that determine module."""
    dic_list, dic_name_list = [], []
    if stage == 'Estrisk':
        # Rest of info comes via a subsequent call to solve or solvefair
        dc_list = [optp_.estriskcfg,]
        label_list = ['estriskcfg']
    else:
        dc_list = [optp_.int_cfg, optp_.gen_cfg, optp_.var_cfg, optp_.dc_cfg, optp_.other_cfg,
                   optp_.rnd_cfg,
                   ]
        label_list = ['int_cfg', 'gen_cfg', 'var_cfg', 'dc_cfg', 'other_cfg', 'rnd_cfg',]
        if stage == 'Fairness':
            dc_list.append(optp_.fair_cfg)
            label_list.append('fair_cfg')

        if optp_.gen_cfg.method in ('policy_tree', 'policy tree old'):
            dc_list.append(optp_.pt_cfg)
            label_list.append('pt_cfg')
            add_list = [optp_.var_x_type, optp_.var_x_values,]
            add_list_name = ['var_x_type', 'var_x_values',]
            dic_list.extend(add_list)
            dic_name_list.extend(add_list_name)

    print_str_list = [mcf_ps.string_dc(dc, label) for dc, label in zip(dc_list, label_list)]
    print_str = '\n'.join(print_str_list)
    mcf_ps.print_mcf(optp_.gen_cfg, print_str, summary=summary)

    for dic, dic_name in zip(dic_list, dic_name_list):
        mcf_ps.print_dic(dic, dic_name, optp_.gen_cfg, summary=summary)

    mcf_ps.print_mcf(optp_.gen_cfg, '\n', summary=summary)


def winners_losers_method(optp_org: 'OptimalPolicy',
                          data_df: pd.DataFrame,
                          welfare_df: pd.DataFrame,
                          *,
                          welfare_reference_df: pd.DataFrame | None = None,
                          outpath: str | Path | None = None,
                          title: str = '',
                          ) -> tuple[pd.DataFrame, Path | None]:
    """Describe winners and losers from different allocations."""
    optp_ = deepcopy(optp_org)

    if outpath is not None:  # Otherwise, outpath from self is taken
        if not isinstance(outpath, Path):
            outpath = Path(outpath)
        outpath.mkdir(parents=True, exist_ok=True)  # In case it does not exist, it is created
        optp_.gen_cfg.outpath = outpath
        optp_.gen_cfg.outfiletext = outpath / optp_org.gen_cfg.outfiletext.name
        optp_.gen_cfg.outfilesummary = outpath / optp_org.gen_cfg.outfilesummary.name

    optp_.gen_cfg.print_to_file = True
    # Define relevant variables for descriptive stats
    variables_to_desc = optp_.var_cfg.polscore_name.copy()
    variables_to_desc.extend(optp_.var_cfg.x_ord_name)
    variables_to_desc.extend(optp_.var_cfg.x_unord_name)
    variables_to_desc.extend(optp_.var_cfg.protected_ord_name)
    variables_to_desc.extend(optp_.var_cfg.protected_unord_name)
    variables_to_desc.extend(optp_.var_cfg.material_ord_name)
    variables_to_desc.extend(optp_.var_cfg.material_unord_name)

    names_unordered = optp_.var_cfg.x_unord_name.copy()
    names_unordered.extend(optp_.var_cfg.protected_unord_name)
    names_unordered.extend(optp_.var_cfg.material_unord_name)

    variables_to_desc = mcf_gp.remove_dupl_keep_order(variables_to_desc)
    names_unordered = mcf_gp.remove_dupl_keep_order(names_unordered)
    # Match configured names without changing the supplied data column names.
    column_names = {name.casefold(): name for name in data_df.columns}
    var_desc = [column_names[name] for name in variables_to_desc if name in column_names]
    names_unordered = [column_names[name] for name in names_unordered if name in column_names]

    # Check welfare inputs and convert to numpy
    if not isinstance(welfare_df, pd.DataFrame):
        raise TypeError('Actual welfare variable must be a DataFrame.')
    if len(welfare_df) != len(data_df):
        raise ValueError(
            'Actual welfare and data must contain the same number of observations.'
            )
    if welfare_df.shape[1] != 1:
        raise ValueError('Welfare must contain exactly one column.')

    welfare_np = welfare_df.to_numpy()
    name_welfare = welfare_df.columns

    if welfare_reference_df is None:
        welfare_ref_np = np.zeros((len(data_df), 1))
        name_welfare_ref = ('zero',)
        welfare_reference_df = pd.DataFrame(welfare_ref_np, columns=name_welfare_ref)
    else:
        if not isinstance(welfare_reference_df, pd.DataFrame):
            raise TypeError('Reference welfare variable must be a DataFrame or None.')
        if len(welfare_reference_df) != len(data_df):
            raise ValueError('Reference welfare and data must contain the same number of '
                             'observations.'
                             )
        if welfare_reference_df.shape[1] != 1:
            raise ValueError('Reference welfare must contain exactly one column.')

        welfare_ref_np = welfare_reference_df.to_numpy()
        name_welfare_ref = welfare_reference_df.columns

    # Add welfare to data and to the variables used to describe data
    welfare_desc_df = welfare_df.reset_index(drop=True)
    welfare_desc_df.columns = [f'Actual welfare ({name_welfare[0]})']
    reference_desc_df = welfare_reference_df.reset_index(drop=True)
    reference_desc_df.columns = [f'Reference welfare ({name_welfare_ref[0]})']
    data_to_desc_df = pd.concat((data_df[var_desc].reset_index(drop=True),
                                 welfare_desc_df,
                                 reference_desc_df,
                                 ),
                                axis=1
                                )
    try:
        welfare_np = np.asarray(welfare_np, dtype=np.float64)
        welfare_ref_np = np.asarray(welfare_ref_np, dtype=np.float64)
        if not (np.all(np.isfinite(welfare_np)) and np.all(np.isfinite(welfare_ref_np))):
            raise ValueError('Non-finite welfare values.')
        with np.errstate(over='raise', invalid='raise'):
            welfare_diff_np = np.subtract(welfare_np, welfare_ref_np, dtype=np.float64)
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise ValueError('Actual and reference welfare, and their differences, must be numeric '
                     'and finite.'
                     ) from exc

    name_diff = f'{name_welfare[0]}_minus_{name_welfare_ref[0]}'
    welfare_cluster_id_name = 'Welfare_change_cluster_' + name_diff

    if len(np.unique(welfare_diff_np)) < 2:  # Not enough variation
        data_df[welfare_cluster_id_name] = np.zeros(len(data_df), dtype=np.int64)

        return data_df, optp_.gen_cfg.outpath

    welfare_diff_df = pd.DataFrame(welfare_diff_np, columns=[name_diff])
    # kmeans clustering
    max_groups = min(8, len(welfare_diff_np) - 1, len(np.unique(welfare_diff_np)))
    if max_groups < 2:
        raise ValueError('At least three observations are required for silhouette-based '
                         'clustering of varying welfare changes.'
                         )
    km_dic = {
        'kmeans_max_tries': 1000,
        'kmeans_replications': 10,
        'kmeans_no_of_groups': list(range(2, max_groups + 1)),
        'kmeans_min_size_share': 1,
        }
    silhouette_avg_prev, cluster_lab_np = -1, None
    txt = '\n' + '=' * 100 + '\nK-Means++ clustering: ' + title + '\n' + name_diff
    txt += '\n' + '-' * 100
    for cluster_no in km_dic['kmeans_no_of_groups']:
        cluster_lab_tmp, silhouette_avg, merge = kmeans_labels(
            welfare_diff_np,
            no_of_clusters=cluster_no,
            kmeans_max_tries=km_dic['kmeans_max_tries'],
            kmeans_replications=km_dic['kmeans_replications'],
            kmeans_min_size_share=km_dic['kmeans_min_size_share'],
            )
        actual_groups = len(np.unique(cluster_lab_tmp))
        score_text = f'{silhouette_avg:8.3f}' if np.isfinite(silhouette_avg) else 'undefined'
        txt += (f'\nRequested clusters: {cluster_no}; retained clusters: {actual_groups}; '
                f'average silhouette score: {score_text}'
                )
        if merge:
            txt += '\nSmall clusters were merged with clusters having the closest centroids.'
        if silhouette_avg > silhouette_avg_prev:
            cluster_lab_np = np.copy(cluster_lab_tmp)
            silhouette_avg_prev = np.copy(silhouette_avg)
    if cluster_lab_np is None:
        txt += ('\nNo candidate retained at least two clusters after merging small clusters. '
                'A silhouette score cannot be computed.'
                )
        mcf_ps.print_mcf(optp_.gen_cfg, txt, summary=True)
        raise ValueError('No candidate clustering has a valid silhouette score.')

    txt += f'\n\nBest value of average silhouette score: {silhouette_avg_prev: 8.3f} '
    # Reorder labels for better visible inspection of results
    cl_means = welfare_diff_df.groupby(by=cluster_lab_np).mean(numeric_only=True)
    cl_means_np = cl_means.to_numpy()
    cl_means_np = np.mean(cl_means_np, axis=1)
    sort_ind = np.argsort(cl_means_np)
    cl_group = cluster_lab_np.copy()
    for cl_j, cl_old in enumerate(sort_ind):
        cl_group[cluster_lab_np == cl_old] = cl_j
    txt += '\n' + '- ' * 50 + '\nWelfare changes are ordered w.r.t. to their size'
    cl_values, cl_obs = np.unique(cl_group, return_counts=True)
    txt += '\n' + '-' * 100 + '\nNumber of observations in the clusters' + '\n' + '- ' * 50

    for idx, val in enumerate(cl_values):
        string = f'\nCluster {val:2}: {cl_obs[idx]:6} '
        txt += string

    txt += '\n' + '-' * 100 + '\nWelfare changes\n' + '- ' * 50

    cl_means = welfare_diff_df.groupby(by=cl_group).mean(numeric_only=True)
    txt += '\n' + cl_means.transpose().to_string()
    txt += '\n' + '-' * 100 + '\nFeatures\n' + '- ' * 50

    if names_unordered:  # List is not empty
        x_dummies_df = pd.get_dummies(data_to_desc_df[names_unordered],
                                      columns=names_unordered, dtype=int,
                                      )
        x_km = pd.concat([data_to_desc_df, x_dummies_df], axis=1)
    else:
        x_km = data_to_desc_df
    cl_means = x_km.groupby(by=cl_group).mean(numeric_only=True)

    mcf_ps.print_mcf(optp_.gen_cfg, txt, summary=True)

    txt = cl_means.transpose().to_string() + '\n' + '-' * 100
    mcf_ps.print_mcf(optp_.gen_cfg, txt, summary=True)

    # Return clusternumber together with data
    data_df[welfare_cluster_id_name] = cl_group.reshape(-1, 1)

    return data_df, optp_.gen_cfg.outpath
