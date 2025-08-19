"""Created on Mon Feb 24 10:08:06 2025.

Contains specific functions for unconfoundedness.

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from time import time
from typing import TYPE_CHECKING
import warnings

import pandas as pd
from ray import is_initialized, shutdown

from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_common_support_functions as mcf_cs
from mcf import mcf_data_functions as mcf_data
from mcf import mcf_estimation_functions as mcf_est
from mcf import mcf_fair_iate_functions as mcf_fair
from mcf import mcf_feature_selection_functions as mcf_fs
from mcf import mcf_forest_functions as mcf_fo
from mcf import mcf_gate_functions as mcf_gate
from mcf import mcf_gateout_functions as mcf_gateout
from mcf import mcf_general as mcf_gp
from mcf import mcf_iate_functions as mcf_iate
from mcf import mcf_init_update_helper_functions as mcf_init_update
from mcf import mcf_init_functions as mcf_init
from mcf import mcf_local_centering_functions as mcf_lc
from mcf import mcf_post_functions as mcf_post
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_qiate_functions as mcf_qiate
from mcf import mcf_weight_functions as mcf_w

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def train_main(mcf_: 'ModifiedCausalForest', data_df: pd.DataFrame) -> dict:
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
        'outpath' : Pathlib object
            Location of directory in which output is saved.

    """
    time_start = time()

    # Reduce sample size to upper limit
    data_df, rnd_reduce, txt_red = mcf_gp.check_reduce_dataframe(
        data_df, title='Training',
        max_obs=mcf_.int_dict['max_obs_training'],
        seed=124535, ignore_index=True)
    if rnd_reduce and mcf_.int_dict['with_output']:
        mcf_ps.print_mcf(mcf_.gen_dict, txt_red, summary=True)

    # Check treatment data
    data_df, _ = mcf_data.data_frame_vars_lower(data_df)
    data_df = mcf_data.check_recode_treat_variable(mcf_, data_df)
    # Initialise again with data information. Order of the following
    # init functions important. Change only if you know what you do.
    mcf_init_update.var_update_train(mcf_, data_df)
    mcf_init_update.gen_update_train(mcf_, data_df)
    mcf_init_update.ct_update_train(mcf_, data_df)
    mcf_init_update.cs_update_train(mcf_)           # Used updated gen_dict info
    mcf_init_update.cf_update_train(mcf_, data_df)
    mcf_init_update.lc_update_train(mcf_, data_df)
    mcf_init_update.int_update_train(mcf_)
    mcf_init_update.p_update_train(mcf_)

    if mcf_.int_dict['with_output']:
        mcf_ps.print_dic_values_all(mcf_, summary_top=True, summary_dic=False)

    # Prepare data: Add and recode variables for GATES (Z)
    #             Recode categorical variables to prime numbers, cont. vars
    data_df = mcf_data.create_xz_variables(mcf_, data_df, train=True)
    if mcf_.int_dict['with_output'] and mcf_.int_dict['verbose']:
        mcf_data.print_prime_value_corr(mcf_.data_train_dict,
                                        mcf_.gen_dict, summary=False)

    # Clean data and remove missings and unncessary variables
    if mcf_.dc_dict['clean_data']:
        data_df, report = mcf_data.clean_data(
            mcf_, data_df, train=True)
        if mcf_.gen_dict['with_output']:
            mcf_.report['training_obs'] = report
    if mcf_.dc_dict['screen_covariates']:   # Only training
        (mcf_.gen_dict, mcf_.var_dict, mcf_.var_x_type,
         mcf_.var_x_values, report
         ) = mcf_data.screen_adjust_variables(mcf_, data_df)
        if mcf_.gen_dict['with_output']:
            mcf_.report['removed_vars'] = report
    time_1 = time()

    # Descriptives by treatment
    if mcf_.int_dict['descriptive_stats'] and mcf_.int_dict['with_output']:
        mcf_ps.desc_by_treatment(mcf_, data_df, summary=False, stage=1)

    # Feature selection
    if mcf_.fs_dict['yes']:
        data_df, report = mcf_fs.feature_selection(mcf_, data_df)
        if mcf_.gen_dict['with_output']:
            mcf_.report['fs_vars_deleted'] = report
    # Split sample for tree building and tree-filling-with-y
    tree_df, fill_y_df = mcf_data.split_sample_for_mcf(mcf_, data_df)

    del data_df
    time_2 = time()

    # Compute Common support
    if mcf_.cs_dict['type']:
        (tree_df, fill_y_df, rep_sh_del, obs_remain, rep_fig,
         cs_tree_prob, cs_fill_y_prob, _
         ) = mcf_cs.common_support(mcf_, tree_df, fill_y_df, train=True)
        if mcf_.gen_dict['with_output']:
            mcf_.report['cs_t_share_deleted'] = rep_sh_del
            mcf_.report['cs_t_obs_remain'] = obs_remain
            mcf_.report['cs_t_figs'] = rep_fig
    else:
        cs_tree_prob = cs_fill_y_prob = None

    # Descriptives by treatment on common support
    if mcf_.int_dict['descriptive_stats'] and mcf_.int_dict['with_output']:
        mcf_ps.desc_by_treatment(mcf_, pd.concat([tree_df, fill_y_df], axis=0),
                                 summary=True, stage=1)
    time_3 = time()

    # Local centering
    if mcf_.lc_dict['yes']:
        (tree_df, fill_y_df, _, report) = mcf_lc.local_centering(
            mcf_, tree_df, fill_y_df)
        if mcf_.gen_dict['with_output']:
            mcf_.report["lc_r2"] = report
    time_4 = time()

    # Train forest
    if mcf_.int_dict['with_output']:
        mcf_ps.variable_features(mcf_, summary=False)
    (mcf_.cf_dict, mcf_.forest, time_vi, report) = mcf_fo.train_forest(
        mcf_, tree_df, fill_y_df)
    if mcf_.gen_dict['with_output']:
        mcf_.report['cf'] = report
    time_end = time()
    time_string = ['Data preparation and stats I:                   ',
                   'Feature preselection:                           ',
                   'Common support:                                 ',
                   'Local centering (recoding of Y):                ',
                   'Training the causal forest:                     ',
                   '  ... of which is time for variable importance: ',
                   '\nTotal time training:                            ']
    time_difference = [time_1 - time_start, time_2 - time_1,
                       time_3 - time_2, time_4 - time_3,
                       time_end - time_4, time_vi,
                       time_end - time_start]
    if mcf_.int_dict['with_output']:
        time_train = mcf_ps.print_timing(
            mcf_.gen_dict, 'Training', time_string, time_difference,
            summary=True)
        mcf_.time_strings['time_train'] = time_train

    results = {'tree_df': tree_df,
               'fill_y_df': fill_y_df,
               'common_support_probabilities_tree': cs_tree_prob,
               'common_support_probabilities_fill_y': cs_fill_y_prob,
               'path_output': mcf_.gen_dict['outpath']
               }

    return results


def predict_main(mcf_: 'ModifiedCausalForest', data_df: pd.DataFrame) -> dict:
    """
    Compute all effects given a causal forest estimated with mcf.

    Parameters
    ----------
    data_df : DataFrame
        Data used to compute the predictions. It must contain information
        about features (and treatment if effects for treatment specific
        subpopulations are desired as well).

    Returns
    -------
    results : Dictionary.
        Results. This dictionary has the following structure:
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
        'common_support_probabilities: ': pd.DataFrame containing treatment
        probabilities for all treatments, the identifier of the observation, and
        a dummy variable indicating whether the observation is inside or outside
        the common support.
        'path_output': Pathlib object, location of directory in which output is
        saved.
    """
    time_start = time()
    report = {}

    # Reduce sample size to upper limit
    data_df, rnd_reduce, txt_red = mcf_gp.check_reduce_dataframe(
        data_df, title='Prediction',
        max_obs=mcf_.int_dict['max_obs_prediction'],
        seed=124535, ignore_index=True)
    if rnd_reduce and mcf_.int_dict['with_output']:
        mcf_ps.print_mcf(mcf_.gen_dict, txt_red, summary=True)

    data_df, _ = mcf_data.data_frame_vars_lower(data_df)
    # Initialise again with data information
    data_df = mcf_init_update.p_update_pred(mcf_, data_df)
    # Check treatment data
    if mcf_.p_dict['d_in_pred']:
        data_df = mcf_data.check_recode_treat_variable(mcf_, data_df)
    mcf_init_update.int_update_pred(mcf_, len(data_df))
    mcf_init_update.post_update_pred(mcf_, data_df)
    if mcf_.int_dict['with_output']:
        mcf_ps.print_dic_values_all(mcf_, summary_top=True, summary_dic=False,
                                    train=False)

    # Prepare data: Add and recode variables for GATES (Z)
    #             Recode categorical variables to prime numbers, cont. vars
    data_df = mcf_data.create_xz_variables(mcf_, data_df, train=False)
    if mcf_.int_dict['with_output'] and mcf_.int_dict['verbose']:
        mcf_data.print_prime_value_corr(mcf_.data_train_dict,
                                        mcf_.gen_dict, summary=False)
    # Clean data and remove missings and unncessary variables
    if mcf_.dc_dict['clean_data']:
        data_df, report['prediction_obs'] = mcf_data.clean_data(
            mcf_, data_df, train=False)
    # Descriptives by treatment on common support
    if (mcf_.p_dict['d_in_pred'] and mcf_.int_dict['descriptive_stats']
            and mcf_.int_dict['with_output']):
        mcf_ps.desc_by_treatment(mcf_, data_df, summary=False, stage=3)
    time_1 = time()

    # Common support
    if mcf_.cs_dict['type']:
        (data_df, _, report['cs_p_share_deleted'],
         report['cs_p_obs_remain'], _, _, _, cs_pred_prob
         ) = mcf_cs.common_support(
             mcf_, data_df, None, train=False)
    else:
        cs_pred_prob = None

    data_df = data_df.copy().reset_index(drop=True)
    if (mcf_.p_dict['d_in_pred'] and mcf_.int_dict['descriptive_stats']
            and mcf_.int_dict['with_output']):
        mcf_ps.desc_by_treatment(mcf_, data_df, summary=True, stage=3)
    time_2 = time()

    # Local centering for IATE
    if mcf_.lc_dict['yes'] and mcf_.lc_dict['uncenter_po']:
        (_, _, y_pred_x_df, _) = mcf_lc.local_centering(mcf_, data_df,
                                                        None, train=False)
    else:
        y_pred_x_df = 0
    time_3 = time()
    time_delta_weight = time_delta_ate = time_delta_bala = 0
    time_delta_iate = time_delta_gate = time_delta_cbgate = 0
    time_delta_bgate = time_delta_qiate = 0
    ate_dic = bala_dic = iate_dic = iate_m_ate_dic = iate_eff_dic = None
    gate_dic = gate_m_ate_dic = cbgate_dic = cbgate_m_ate_dic = None
    gate_est_dic = bgate_est_dic = cbgate_est_dic = None
    bgate_dic = bgate_m_ate_dic = qiate_dic = qiate_est_dic = None
    qiate_m_med_dic = qiate_m_opp_dic = None
    txt_b = txt_am = ''
    only_one_fold_one_round = (mcf_.cf_dict['folds'] == 1
                               and len(mcf_.cf_dict['est_rounds']) == 1)
    for fold in range(mcf_.cf_dict['folds']):
        for round_ in mcf_.cf_dict['est_rounds']:
            time_w_start = time()
            if only_one_fold_one_round:
                forest_dic = mcf_.forest[fold][0]
            else:
                forest_dic = deepcopy(
                    mcf_.forest[fold][0 if round_ == 'regular' else 1])
            if mcf_.int_dict['with_output'] and mcf_.int_dict['verbose']:
                print(f'\n\nWeight maxtrix {fold+1} /',
                      f'{mcf_.cf_dict["folds"]} forests, {round_}')
            weights_dic = mcf_w.get_weights_mp(
                mcf_, data_df, forest_dic, round_ == 'regular')
            time_delta_weight += time() - time_w_start
            time_a_start = time()
            if round_ == 'regular':
                # Estimate ATE n fold
                (w_ate, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
                    mcf_, data_df, weights_dic)
                # Aggregate ATEs over folds
                ate_dic = mcf_est.aggregate_pots(
                    mcf_, y_pot_f, y_pot_var_f, txt_w_f, ate_dic, fold,
                    title='ATE')
            else:
                w_ate = None
            time_delta_ate += time() - time_a_start
            # Compute balancing tests
            time_b_start = time()
            if round_ == 'regular':
                if mcf_.p_dict['bt_yes']:
                    (_, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
                        mcf_, data_df, weights_dic, balancing_test=True)
                    # Aggregate Balancing results over folds
                    bala_dic = mcf_est.aggregate_pots(
                        mcf_, y_pot_f, y_pot_var_f, txt_w_f, bala_dic,
                        fold, title='Balancing check: ')
            time_delta_bala += time() - time_b_start

            # BGATE
            time_bgate_start = time()
            if round_ == 'regular' and mcf_.p_dict['bgate']:
                (y_pot_bgate_f, y_pot_var_bgate_f, y_pot_mate_bgate_f,
                 y_pot_mate_var_bgate_f, bgate_est_dic, txt_w_f, txt_b,
                 ) = mcf_gate.bgate_est(mcf_, data_df, weights_dic,
                                        w_ate, forest_dic,
                                        gate_type='BGATE')
                bgate_dic = mcf_est.aggregate_pots(
                    mcf_, y_pot_bgate_f, y_pot_var_bgate_f, txt_w_f,
                    bgate_dic, fold, pot_is_list=True, title='BGATE')
                if y_pot_mate_bgate_f is not None:
                    bgate_m_ate_dic = mcf_est.aggregate_pots(
                        mcf_, y_pot_mate_bgate_f, y_pot_mate_var_bgate_f,
                        txt_w_f, bgate_m_ate_dic, fold, pot_is_list=True,
                        title='BGATE minus ATE')
            time_delta_bgate += time() - time_bgate_start

            # CBGATE
            time_cbg_start = time()
            if round_ == 'regular' and mcf_.p_dict['cbgate']:
                (y_pot_cbgate_f, y_pot_var_cbgate_f, y_pot_mate_cbgate_f,
                 y_pot_mate_var_cbgate_f, cbgate_est_dic, txt_w_f, txt_am,
                 ) = mcf_gate.bgate_est(mcf_, data_df, weights_dic,
                                        w_ate, forest_dic,
                                        gate_type='CBGATE')
                cbgate_dic = mcf_est.aggregate_pots(
                    mcf_, y_pot_cbgate_f, y_pot_var_cbgate_f, txt_w_f,
                    cbgate_dic, fold, pot_is_list=True, title='CBGATE')
                if y_pot_mate_cbgate_f is not None:
                    cbgate_m_ate_dic = mcf_est.aggregate_pots(
                        mcf_, y_pot_mate_cbgate_f, y_pot_mate_var_cbgate_f,
                        txt_w_f, cbgate_m_ate_dic, fold, pot_is_list=True,
                        title='CBGATE minus ATE')
            time_delta_cbgate += time() - time_cbg_start
            if mcf_.int_dict['del_forest']:
                del forest_dic['forest']

            # IATE
            time_i_start = time()
            if mcf_.p_dict['iate']:
                y_pot_eff = None
                (y_pot_f, y_pot_var_f, y_pot_m_ate_f, y_pot_m_ate_var_f,
                 txt_w_f) = mcf_iate.iate_est_mp(
                     mcf_, weights_dic, w_ate, round_ == 'regular')
                if round_ == 'regular':
                    y_pot_iate_f = y_pot_f.copy()
                    y_pot_varf = (None if y_pot_var_f is None
                                  else y_pot_var_f.copy())
                    iate_dic = mcf_est.aggregate_pots(
                        mcf_, y_pot_iate_f, y_pot_varf, txt_w_f, iate_dic,
                        fold, title='IATE')
                    if y_pot_m_ate_f is not None:
                        iate_m_ate_dic = mcf_est.aggregate_pots(
                            mcf_, y_pot_m_ate_f, y_pot_m_ate_var_f,
                            txt_w_f, iate_m_ate_dic, fold,
                            title='IATE minus ATE')
                else:
                    y_pot_eff = (y_pot_iate_f + y_pot_f) / 2
                    iate_eff_dic = mcf_est.aggregate_pots(
                        mcf_, y_pot_eff, None, txt_w_f, iate_eff_dic, fold,
                        title='IATE eff')
            time_delta_iate += time() - time_i_start

            # QIATE
            time_q_start = time()
            if round_ == 'regular' and mcf_.p_dict['qiate']:
                (y_pot_qiate_f, y_pot_var_qiate_f,
                 y_pot_mmed_qiate_f, y_pot_mmed_var_qiate_f,
                 y_pot_mopp_qiate_f, y_pot_mopp_var_qiate_f,
                 qiate_est_dic, txt_w_f
                 ) = mcf_qiate.qiate_est(mcf_, data_df, weights_dic,
                                         y_pot_f, y_pot_var=y_pot_var_f
                                         )
                qiate_dic = mcf_est.aggregate_pots(
                    mcf_, y_pot_qiate_f, y_pot_var_qiate_f, txt_w_f,
                    qiate_dic, fold, pot_is_list=True, title='QIATE')
                if y_pot_mmed_qiate_f is not None:
                    qiate_m_med_dic = mcf_est.aggregate_pots(
                        mcf_, y_pot_mmed_qiate_f, y_pot_mmed_var_qiate_f,
                        txt_w_f, qiate_m_med_dic, fold, pot_is_list=True,
                        title='QIATE minus QIATE(median))')
                if y_pot_mopp_qiate_f is not None:
                    qiate_m_opp_dic = mcf_est.aggregate_pots(
                        mcf_, y_pot_mopp_qiate_f, y_pot_mopp_var_qiate_f,
                        txt_w_f, qiate_m_opp_dic, fold, pot_is_list=True,
                        title='QIATE(q) minus QIATE(1-q))')
            time_delta_qiate += time() - time_q_start  # QIATE

            # GATE
            time_g_start = time()
            if round_ == 'regular' and mcf_.p_dict['gate']:
                (y_pot_gate_f, y_pot_var_gate_f, y_pot_mate_gate_f,
                 y_pot_mate_var_gate_f, gate_est_dic, txt_w_f
                 ) = mcf_gate.gate_est(mcf_, data_df, weights_dic, w_ate)
                gate_dic = mcf_est.aggregate_pots(
                    mcf_, y_pot_gate_f, y_pot_var_gate_f, txt_w_f,
                    gate_dic, fold, pot_is_list=True, title='GATE')
                if y_pot_mate_gate_f is not None:
                    gate_m_ate_dic = mcf_est.aggregate_pots(
                        mcf_, y_pot_mate_gate_f, y_pot_mate_var_gate_f,
                        txt_w_f, gate_m_ate_dic, fold, pot_is_list=True,
                        title='GATE minus ATE')
            time_delta_gate += time() - time_g_start

        if (mcf_.int_dict['mp_ray_shutdown']
            and mcf_.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        if not only_one_fold_one_round and mcf_.int_dict['del_forest']:
            mcf_.forest[fold] = None
            # Without this and the next delete, it becomes impossible to reuse
            # the same forest for several data sets, which is bad.

    if mcf_.int_dict['del_forest']:
        mcf_.forest = None

    del weights_dic

    # ATE
    time_a_start = time()
    ate, ate_se, ate_effect_list = mcf_ate.ate_effects_print(
        mcf_, ate_dic, y_pred_x_df, balancing_test=False)
    time_delta_ate += time() - time_a_start

    # GATE
    time_g_start = time()
    if mcf_.p_dict['gate']:
        (gate, gate_se, gate_diff, gate_diff_se, report['fig_gate']
         ) = mcf_gateout.gate_effects_print(mcf_, gate_dic, gate_m_ate_dic,
                                            gate_est_dic, ate, ate_se,
                                            gate_type='GATE',
                                            iv=False)
    else:
        gate = gate_se = gate_diff = gate_diff_se = gate_est_dic = None
    time_delta_gate += time() - time_g_start

    # BGATE
    time_bgate_start = time()
    if mcf_.p_dict['bgate']:
        (bgate, bgate_se, bgate_diff, bgate_diff_se, report['fig_bgate']
         ) = mcf_gateout.gate_effects_print(
             mcf_, bgate_dic, bgate_m_ate_dic, bgate_est_dic, ate,
             ate_se, gate_type='BGATE', iv=False, special_txt=txt_b)
    else:
        bgate = bgate_se = bgate_diff = bgate_diff_se = None
        bgate_est_dic = None
    time_delta_bgate += time() - time_bgate_start

    # CBGATE
    time_cbg_start = time()
    if mcf_.p_dict['cbgate']:
        (cbgate, cbgate_se, cbgate_diff, cbgate_diff_se,
         report['fig_cbgate']) = mcf_gateout.gate_effects_print(
             mcf_, cbgate_dic, cbgate_m_ate_dic, cbgate_est_dic, ate,
             ate_se, gate_type='CBGATE', iv=False, special_txt=txt_am)
    else:
        cbgate = cbgate_se = cbgate_diff = cbgate_diff_se = None
        cbgate_est_dic = None
    time_delta_cbgate += time() - time_cbg_start
    # Collect some information for results_dic
    if (mcf_.p_dict['gate'] or mcf_.p_dict['bgate']
            or mcf_.p_dict['cbgate']):
        gate_names_values = mcf_gateout.get_names_values(
            mcf_, gate_est_dic, bgate_est_dic, cbgate_est_dic)
    else:
        gate_names_values = None

    # QIATE
    time_q_start = time()
    if mcf_.p_dict['qiate']:
        (qiate, qiate_se, qiate_mmed, qiate_mmed_se,
         qiate_mopp, qiate_mopp_se, report['fig_qiate']
         ) = mcf_qiate.qiate_effects_print(mcf_, qiate_dic,
                                           qiate_m_med_dic,
                                           qiate_m_opp_dic,
                                           qiate_est_dic)
    else:
        qiate = qiate_se = qiate_est_dic = None
        qiate_mmed = qiate_mmed_se = qiate_mopp = qiate_mopp_se = None

    time_delta_qiate += time() - time_q_start  # QIATE

    # IATE
    time_i_start = time()
    if mcf_.p_dict['iate']:
        (iate, iate_se, iate_eff, iate_names_dic, iate_df,
         report['iate_text']) = mcf_iate.iate_effects_print(
             mcf_, iate_dic, iate_m_ate_dic, iate_eff_dic, y_pred_x_df)
        data_df.reset_index(drop=True, inplace=True)
        iate_df.reset_index(drop=True, inplace=True)
        iate_pred_df = pd.concat([data_df, iate_df], axis=1)
    else:
        iate_eff = iate = iate_se = iate_df = iate_pred_df = None
        iate_names_dic = None
    time_delta_iate += time() - time_i_start

    # Balancing test
    time_b_start = time()
    if mcf_.p_dict['bt_yes']:
        bala, bala_se, bala_effect_list = mcf_ate.ate_effects_print(
            mcf_, bala_dic, None, balancing_test=True)
    else:
        bala = bala_se = bala_effect_list = None
    time_delta_bala += time() - time_b_start

    # Collect results
    results = {
        'ate': ate, 'ate_se': ate_se, 'ate_effect_list': ate_effect_list,
        'gate': gate, 'gate_se': gate_se,
        'gate_diff': gate_diff, 'gate_diff_se': gate_diff_se,
        'gate_names_values': gate_names_values,
        'cbgate': cbgate, 'cbgate_se': cbgate_se,
        'cbgate_diff': cbgate_diff, 'cbgate_diff_se': cbgate_diff_se,
        'bgate': bgate, 'bgate_se': bgate_se,
        'bgate_diff': bgate_diff, 'bgate_diff_se': bgate_diff_se,
        'qiate': qiate, 'qiate_se': qiate_se,
        'qiate_mmed': qiate_mmed, 'qiate_mmed_se': qiate_mmed_se,
        'qiate_mopp': qiate_mopp, 'qiate_mopp_se': qiate_mopp_se,
        'iate': iate, 'iate_se': iate_se, 'iate_eff': iate_eff,
        'iate_data_df': iate_pred_df, 'iate_names_dic': iate_names_dic,
        'bala': bala, 'bala_se': bala_se,
        'bala_effect_list': bala_effect_list,
        'common_support_probabilities': cs_pred_prob,
        'path_output': mcf_.gen_dict['outpath']
               }
    if mcf_.int_dict['with_output']:
        report['mcf_pred_results'] = results
    mcf_.report['predict_list'].append(report.copy())
    time_end = time()
    if mcf_.int_dict['with_output']:
        time_string = [
            'Data preparation and stats II:                  ',
            'Common support:                                 ',
            'Local centering (recoding of Y):                ',
            'Weights:                                        ',
            'ATEs:                                           ',
            'GATEs:                                          ',
            'BGATEs:                                         ',
            'CBGATEs:                                        ',
            'QIATEs:                                         ',
            'IATEs:                                          ',
            'Balancing test:                                 ',
            '\nTotal time prediction:                          ']
        time_difference = [
            time_1 - time_start, time_2 - time_1, time_3 - time_2,
            time_delta_weight, time_delta_ate, time_delta_gate,
            time_delta_bgate, time_delta_cbgate, time_delta_qiate,
            time_delta_iate, time_delta_bala, time_end - time_start]
        mcf_ps.print_mcf(mcf_.gen_dict, mcf_.time_strings['time_train'])
        time_pred = mcf_ps.print_timing(
            mcf_.gen_dict, 'Prediction', time_string, time_difference,
            summary=True)
        mcf_.time_strings['time_pred'] = time_pred

    return results


def analyse_main(mcf_: 'ModifiedCausalForest', results: dict) -> dict:
    """
    Analyse estimated IATE with various descriptive tools.

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
    report = {}

    # Identify if results come from IV estimation or not.
    iv = mcf_.iv_mcf['firststage'] is not None

    if (mcf_.int_dict['with_output'] and mcf_.post_dict['est_stats'] and
            mcf_.int_dict['return_iate_sp']):
        time_start = time()
        report['fig_iate'] = mcf_post.post_estimation_iate(mcf_, results, iv)
        time_end_corr = time()
        if mcf_.post_dict['kmeans_yes']:
            (results_plus_cluster, report['knn_table']
             ) = mcf_post.k_means_of_x_iate(mcf_, results)
        else:
            results_plus_cluster = report['knn_table'] = None
        time_end_km = time()
        if mcf_.post_dict['random_forest_vi'] or mcf_.post_dict['tree']:
            mcf_post.random_forest_tree_of_iate(mcf_, results)

        time_string = [
            'Correlational analysis and plots of IATE:       ',
            'K-means clustering of IATE:                     ',
            'Random forest / tree analysis of IATE:          ',
            '\nTotal time post estimation analysis:            ']
        time_difference = [
            time_end_corr - time_start, time_end_km - time_end_corr,
            time() - time_end_km, time() - time_start]
        mcf_ps.print_mcf(mcf_.gen_dict, mcf_.time_strings['time_train'],
                         summary=True)
        mcf_ps.print_mcf(mcf_.gen_dict, mcf_.time_strings['time_pred'],
                         summary=True)
        mcf_ps.print_timing(mcf_.gen_dict, 'Analysis of IATE', time_string,
                            time_difference, summary=True)
        mcf_.report['analyse_list'].append(report.copy())
    else:
        raise ValueError(
            '"Analyse" method produces output only if all of the following'
            ' parameters are True:'
            f'\nint_with_output: {mcf_.int_dict["with_output"]}'
            f'\npos_test_stats: {mcf_.post_dict["est_stats"]}'
            f'\nint_return_iate_sp: {mcf_.int_dict["return_iate_sp"]}')

    return results_plus_cluster


def blinder_iates_main(
    mcf_, data_df, blind_var_x_protected_name=None,
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
    warnings.simplefilter("always", DeprecationWarning)
    warnings.warn('This method of reducing dependence on protected '
                  'variables is deprecated. Use the method '
                  'fairscores of the OptimalPolicy class instead.',
                  DeprecationWarning)

    mcf_.blind_dict = mcf_init.blind_init(
        var_x_protected_name=blind_var_x_protected_name,
        var_x_policy_name=blind_var_x_policy_name,
        var_x_unrestricted_name=blind_var_x_unrestricted_name,
        weights_of_blind=blind_weights_of_blind,
        obs_ref_data=blind_obs_ref_data,
        seed=blind_seed)

    if mcf_.int_dict['with_output']:
        time_start = time()
    with_output = mcf_.int_dict['with_output']

    (blinded_dic, data_on_support_df, var_x_policy_ord_name,
     var_x_policy_unord_name, var_x_blind_ord_name, var_x_blind_unord_name
     ) = mcf_fair.make_fair_iates(mcf_, data_df, with_output=with_output)

    mcf_.int_dict['with_output'] = with_output
    if mcf_.int_dict['with_output']:
        time_difference = [time() - time_start]
        time_string = ['Total time for blinding IATEs:                  ']
        mcf_ps.print_timing(mcf_.gen_dict, 'Blinding IATEs', time_string,
                            time_difference, summary=True)

    return (blinded_dic, data_on_support_df, var_x_policy_ord_name,
            var_x_policy_unord_name, var_x_blind_ord_name,
            var_x_blind_unord_name, mcf_.gen_dict['outpath'])
