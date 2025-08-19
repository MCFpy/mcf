"""Created on Thu Oct 31 10:08:06 2024.

Contains IV specific functions.

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from time import time
from typing import TYPE_CHECKING

import pandas as pd
from ray import is_initialized, shutdown

from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_common_support_functions as mcf_cs
from mcf import mcf_data_functions as mcf_data
from mcf import mcf_feature_selection_functions as mcf_fs
from mcf import mcf_forest_functions as mcf_fo
from mcf import mcf_gateout_functions as mcf_gateout
from mcf import mcf_iate_functions as mcf_iate
from mcf import mcf_init_update_helper_functions as mcf_init_update
from mcf import mcf_local_centering_functions as mcf_lc
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_iv_functions_add as mcf_iv_add

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def train_iv_main(mcf_: 'ModifiedCausalForest',
                  data_df: pd.DataFrame
                  ) -> dict:
    """
    Train the modified causal forest IV on the training data.

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
    # Check treatment data
    data_df, _ = mcf_data.data_frame_vars_lower(data_df)
    data_df = mcf_data.check_recode_treat_variable(mcf_, data_df)
    # Initialise again with data information. Order of the following
    # init functions important. Change only if you know what you do.

    mcf_.int_dict['iv'] = True
    if not mcf_.var_dict['iv_name']:
        raise ValueError('Instrument must be specified for IV estimation.')
    if not mcf_.int_dict['weight_as_sparse']:
        raise ValueError('Weights must be handled using the sparse format for '
                         'IV estimation. '
                         'Change "_int_weight_as_sparse" to "False".'
                         )

    # These standard updates remain as some are important for the predict method
    mcf_init_update.var_update_train(mcf_, data_df)
    mcf_init_update.gen_update_train(mcf_, data_df)
    mcf_init_update.ct_update_train(mcf_, data_df)
    mcf_init_update.cs_update_train(mcf_)           # Used updated gen_dict info
    mcf_init_update.cf_update_train(mcf_, data_df)
    mcf_init_update.int_update_train(mcf_)
    mcf_init_update.lc_update_train(mcf_, data_df)
    mcf_init_update.p_update_train(mcf_)

    if mcf_.int_dict['with_output']:
        mcf_ps.print_dic_values_all(mcf_, summary_top=True, summary_dic=False,
                                    title='(IV training)'
                                    )
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

    # Ensure that treatment and instrument are binary (and coded as integer)
    data_df = ensure_binary_binary(mcf_, data_df)

    # Descriptives by treatment
    if mcf_.int_dict['descriptive_stats'] and mcf_.int_dict['with_output']:
        mcf_ps.desc_by_treatment(mcf_, data_df, summary=False, stage=1)
    time_1 = time()

    # Feature selection on the first stage only
    if mcf_.fs_dict['yes']:
        mcf_inst_fs = deepcopy(mcf_)
        mcf_inst_fs.var_dict['d_name'] = mcf_.var_dict['iv_name']
        mcf_inst_fs.var_dict['y_name'] = mcf_.var_dict['d_name']
        mcf_inst_fs.var_dict['y_tree_name'] = mcf_.var_dict['d_name']
        data_df, report = mcf_fs.feature_selection(mcf_inst_fs, data_df)
        mcf_.var_dict['x_name'] = mcf_inst_fs.var_dict['x_name'].copy()
        mcf_.var_x_type = mcf_inst_fs.var_x_type.copy()
        mcf_.var_x_values = mcf_inst_fs.var_x_values.copy()
        if mcf_.gen_dict['with_output']:
            mcf_.report['fs_vars_deleted'] = report
        del mcf_inst_fs

    # Split sample for tree building and tree-filling-with-y
    tree_df, fill_y_df = mcf_data.split_sample_for_mcf(mcf_, data_df)

    del data_df
    time_2 = time()

    # Compute Common support on full sample
    if mcf_.cs_dict['type']:
        mcf_inst_cs = deepcopy(mcf_)
        mcf_inst_cs.var_dict['d_name'] = mcf_.var_dict['iv_name']
        (tree_df, fill_y_df, rep_sh_del, obs_remain, rep_fig,
         cs_tree_prob, cs_fill_y_prob, _
         ) = mcf_cs.common_support(mcf_inst_cs, tree_df, fill_y_df, train=True)
        if mcf_.gen_dict['with_output']:
            mcf_.report['cs_t_share_deleted'] = rep_sh_del
            mcf_.report['cs_t_obs_remain'] = obs_remain
            mcf_.report['cs_t_figs'] = rep_fig
        mcf_.cs_dict = deepcopy(mcf_inst_cs.cs_dict)
        del mcf_inst_cs
    else:
        cs_tree_prob = cs_fill_y_prob = None

    # Descriptives by treatment on common support
    if mcf_.int_dict['descriptive_stats'] and mcf_.int_dict['with_output']:
        mcf_ps.desc_by_treatment(mcf_, pd.concat([tree_df, fill_y_df], axis=0),
                                 summary=True, stage=1)
    time_3 = time()

    # Create new instances for 1st stage and reduced form
    mcf_1st, mcf_redf = new_instances_iv_train(mcf_)

    # Local centering
    if mcf_.lc_dict['yes']:
        (tree_df, _, _, report_1st) = mcf_lc.local_centering(
            mcf_1st, tree_df, fill_y_df=None, title='(1st stage)')

        (tree_df, fill_y_df, _, report_redf) = mcf_lc.local_centering(
            mcf_redf, tree_df, fill_y_df=fill_y_df, title='(reduced form)'
            )

        if mcf_.gen_dict['with_output']:
            mcf_.report["lc_r2_1st"] = report_1st
            mcf_.report["lc_r2_redform"] = report_redf
    time_4 = time()

    # Train forests
    # 1st stage: Train forest on tree data only (not honest)
    if mcf_.int_dict['with_output']:
        mcf_ps.variable_features(mcf_, summary=False)
        txt = ('\n' * 2 + '-' * 100 + '\n' + '1st stage forest' + '\n'
               + '-' * 100)
        mcf_ps.print_mcf(mcf_.gen_dict, txt, summary=True)
    (mcf_1st.cf_dict, mcf_1st.forest, time_vi_1st, report_1st
     ) = mcf_fo.train_forest(mcf_1st, tree_df, tree_df, ' (1st stage)')
    time_5 = time()

    if (mcf_.int_dict['mp_ray_shutdown']
        and mcf_.gen_dict['mp_parallel'] > 1
            and is_initialized()):
        shutdown()

    # Reduced form (honest)
    if mcf_.int_dict['with_output']:
        txt = ('\n' * 2 + '-' * 100 + '\n' + 'Reduced form forest' + '\n'
               + '-' * 100)
        mcf_ps.print_mcf(mcf_.gen_dict, txt, summary=True)

    (mcf_redf.cf_dict, mcf_redf.forest, time_vi_redf, report_redf
     ) = mcf_fo.train_forest(mcf_redf, tree_df, fill_y_df, ' (reduced form)')

    if mcf_.gen_dict['with_output']:
        mcf_.report['cf_1st'] = report_1st
        mcf_.report['cf_redf'] = report_redf

    mcf_.iv_mcf['firststage'] = mcf_1st
    mcf_.iv_mcf['reducedform'] = mcf_redf

    time_end = time()
    time_string = ['Data preparation and stats I:                   ',
                   'Feature preselection:                           ',
                   'Common support:                                 ',
                   'Local centering (recoding of Y & D):            ',
                   'Training the causal forest (1st stage):         ',
                   '  ... of which is time for variable importance: ',
                   'Training the causal forest (reduced form):      ',
                   '  ... of which is time for variable importance: ',
                   '\nTotal time training:                            ']
    time_difference = [time_1 - time_start, time_2 - time_1,
                       time_3 - time_2, time_4 - time_3,
                       time_5 - time_4, time_vi_1st,
                       time_end - time_5, time_vi_redf,
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


def predict_iv_main(mcf_: 'ModifiedCausalForest',
                    data_df: pd.DataFrame
                    ) -> tuple[dict, dict]:
    """
    Compute all effects based on causal forests estimated by the train_iv.

    Parameters
    ----------
    data_df : DataFrame
        Data used to compute the predictions. It must contain information
        about features (and treatment if effects for treatment specific
        subpopulations are desired as well).

    Returns
    -------
    results : Dictionary.
        Results.

    """
    time_start = time()
    report = {}

    # Get instances that contain the results from training
    mcf_1st = mcf_.iv_mcf['firststage']
    mcf_redf = mcf_.iv_mcf['reducedform']
    update_mcf_(mcf_, mcf_1st)

    # Initialise again with data information
    data_df = update_inst_data(mcf_, mcf_1st, mcf_redf, data_df)

    # Print information of dictionaries
    if mcf_.int_dict['with_output']:
        print_dics(mcf_, mcf_1st, mcf_redf)

    # Prepare data and some descriptive stats
    data_df, report = prepare_data_pred(mcf_, data_df, report)
    time_1 = time()

    # Common support
    data_df, report, cs_pred_prob = common_support_pred(mcf_, data_df, report)
    time_2 = time()

    # Local centering for IATE
    if mcf_.lc_dict['yes'] and mcf_.lc_dict['uncenter_po']:
        y_pred_x_df, d_pred_x_df = local_center(mcf_1st, mcf_redf, data_df)
    else:
        y_pred_x_df = d_pred_x_df = 0
    time_3 = time()

    # Initialise time related variables
    time_delta_weight = time_delta_late = time_delta_lbala = 0
    time_delta_lgate = time_delta_lcbgate = time_delta_lbgate = 0
    time_delta_liate = time_delta_lqiate = 0

    # Initialise dictionaries that contain the various results
    results_global, results_local = {}, {}
    # Initialise variables containing results of 1st stages and reduced forms
    ate_1st_dic = ate_redf_dic = bala_1st_dic = bala_redf_dic = None
    iate_1st_dic = iate_redf_dic = None
    iate_m_ate_1st_dic = iate_m_ate_redf_dic = None
    iate_eff_1st_dic = iate_eff_redf_dic = None

    # Initialise variables containing results for LIATEs
    liate_dic = liate_m_late_local_dic = liate_eff_dic = None
    # qliate_est_dic = qliate_dic = None
    # qliate_m_med_dic = qliate_m_opp_dic = None

    # Initialise variables containing results of local and global aggregations
    late_global_dic = None
    # lgate_global_dic = lgate_m_late_global_dic = None
    # lbgate_global_dic = lbgate_m_late_global_dic = None
    # lcbgate_global_dic = lcbgate_m_late_global_dic = None

    late_local_dic = lgate_local_dic = lgate_m_late_local_dic = None
    lbgate_local_dic = lbgate_m_late_local_dic = None
    lcbgate_local_dic = lcbgate_m_late_local_dic = None

    global_effects = 'global' in mcf_.p_dict['iv_aggregation_method']
    local_effects = 'local' in mcf_.p_dict['iv_aggregation_method']
    only_one_fold_one_round = (mcf_.cf_dict['folds'] == 1
                               and len(mcf_.cf_dict['est_rounds']) == 1)

    # Start with 1st stage only (as IATEs of 1st stage are needed for scaling)
    iate_1st_dic, iate_eff_1st_dic = mcf_iv_add.iate_1st_stage_all_folds_rounds(
        mcf_, mcf_1st, data_df, only_one_fold_one_round)

    time_1stscale = time()
    # All other parameters
    # Estimate effects fold by fold and then average across folds
    for fold in range(mcf_.cf_dict['folds']):
        for round_ in mcf_.cf_dict['est_rounds']:
            time_w_start = time()

            # Get relevant forests
            forest_1st_dic, forest_redf_dic = get_forests(
                mcf_, mcf_1st, mcf_redf, only_one_fold_one_round, fold, round_)

            # Compute weights of reduced form & 1st stage & local estimation
            (weights_1st_dic, weights_redf_dic, weights_local_dic
             ) = mcf_iv_add.get_weights_iv_local(
                mcf_, mcf_1st, mcf_redf, forest_1st_dic, forest_redf_dic,
                iate_1st_dic, iate_eff_1st_dic, data_df, round_,
                local_effects=local_effects
                )

            time_delta_weight += time() - time_w_start
            time_a_start = time()
            # Compute effects
            weights = (weights_local_dic, weights_1st_dic, weights_redf_dic,)
            instances = (mcf_, mcf_1st, mcf_redf,)
            # LATE
            if round_ == 'regular':
                (_, w_late_local, late_global_dic, late_local_dic,
                 w_ate_1st, ate_1st_dic, w_ate_redf, ate_redf_dic
                 ) = mcf_iv_add.ate_iv(instances,
                                       weights,
                                       late_global_dic, late_local_dic,
                                       ate_1st_dic, ate_redf_dic,
                                       data_df,
                                       fold,
                                       global_effects=global_effects,
                                       local_effects=local_effects
                                       )
            else:
                w_late_local = w_ate_1st = w_ate_redf = None
            time_delta_late += time() - time_a_start

            # Balancing tests for reduced form and first stage
            time_b_start = time()
            if round_ == 'regular' and mcf_.p_dict['bt_yes']:
                bala_1st_dic, bala_redf_dic = mcf_iv_add.bala_1st_redf(
                    instances, weights, bala_1st_dic, bala_redf_dic,
                    data_df, fold)
            time_delta_lbala += time() - time_b_start

            # BGATE & CBGATE
            time_lbgate_start = time()
            if round_ == 'regular' and (mcf_.p_dict['bgate']
                                        or mcf_.p_dict['cbgate']):
                iv_tuple = (mcf_1st, mcf_redf, forest_1st_dic, forest_redf_dic,)
            else:
                iv_tuple = None

            # BGATE
            if round_ == 'regular' and mcf_.p_dict['bgate']:
                (_, lbgate_local_dic, _, lbgate_m_late_local_dic, _,
                 lbgate_est_local_dic, txt_lbg) = mcf_iv_add.bgate_iv(
                    mcf_,
                    data_df,
                    weights,
                    w_late_local,
                    None, lbgate_local_dic,
                    None, lbgate_m_late_local_dic,
                    fold,
                    iv_tuple=iv_tuple,
                    gate_type='BGATE', title='LBGATE',
                    global_effects=False, local_effects=local_effects
                    )
            else:
                lbgate_local_dic = txt_lbg = None
                lbgate_m_late_local_dic = lbgate_est_local_dic = None
            time_delta_lbgate += time() - time_lbgate_start

            # CBGATE
            time_cbg_start = time()
            if round_ == 'regular' and mcf_.p_dict['cbgate']:
                (_, lcbgate_local_dic, _, lcbgate_m_late_local_dic, _,
                 lcbgate_est_local_dic, txt_lcbg
                 ) = mcf_iv_add.bgate_iv(
                     mcf_,
                     data_df,
                     weights,
                     w_late_local,
                     None, lcbgate_local_dic,
                     None, lcbgate_m_late_local_dic,
                     fold,
                     iv_tuple=iv_tuple,
                     gate_type='CBGATE', title='LCBGATE',
                     global_effects=False, local_effects=local_effects
                     )
            else:
                lcbgate_local_dic = txt_lcbg = None
            time_delta_lcbgate += time() - time_cbg_start
            if mcf_.int_dict['del_forest']:
                del forest_1st_dic['forest'], forest_redf_dic['forest']

            # IATE
            time_i_start = time()
            if mcf_.p_dict['iate']:
                y_pot_liate_f = y_pot_liate_1st_f = y_pot_liate_redf_f = None
                # LIATE
                (liate_dic, liate_m_late_dic, liate_eff_dic, y_pot_liate_f
                 ) = mcf_iv_add.iate_iv(
                    mcf_,
                    weights_local_dic,
                    liate_dic, liate_m_late_local_dic, liate_eff_dic,
                    w_late_local,
                    y_pot_liate_f,
                    round_, fold, iv=True, title='LIATE'
                    )

                # IATE (1st stage)
                iate_1st_dic, _, iate_eff_1st_dic, _ = mcf_iv_add.iate_iv(
                     mcf_1st,
                     weights_1st_dic,
                     iate_1st_dic, iate_m_ate_1st_dic, iate_eff_1st_dic,
                     w_ate_1st,
                     y_pot_liate_1st_f,
                     round_, fold, iv=False, title='IATE (1st stage)'
                     )
                # IATE (reduced form)
                (iate_redf_dic, _, iate_eff_redf_dic, _) = mcf_iv_add.iate_iv(
                     mcf_redf,
                     weights_redf_dic,
                     iate_redf_dic, iate_m_ate_redf_dic, iate_eff_redf_dic,
                     w_ate_redf,
                     y_pot_liate_redf_f,
                     round_, fold, iv=False, title='IATE (reduced form)'
                     )
            time_delta_liate += time() - time_i_start

            # No QIATE !
            time_q_start = time()
            mcf_.p_dict['qiate'] = False
# =============================================================================
#             Not implemented for QIATE (if implementing, update before)
#             if round_ == 'regular' and mcf_.p_dict['qiate']:
#                 (y_pot_qiate_f, y_pot_var_qiate_f, y_pot_mmed_qiate_f,
#                  y_pot_mmed_var_qiate_f, qiate_est_dic, txt_w_f
#                  ) = mcf_qiate.qiate_est(mcf_, data_df, weights_dic, y_pot_f,
#                                          iv=True)
#                 qiate_dic = mcf_est.aggregate_pots(
#                     mcf_, y_pot_qiate_f, y_pot_var_qiate_f, txt_w_f,
#                     qiate_dic, fold, pot_is_list=True, title='QLIATE')
#                 if y_pot_mmed_qiate_f is not None:
#                     qiate_m_med_dic = mcf_est.aggregate_pots(
#                         mcf_, y_pot_mmed_qiate_f, y_pot_mmed_var_qiate_f,
#                         txt_w_f, qiate_m_med_dic, fold, pot_is_list=True,
#                         title='QLIATE minus LIATE(median)')
# =============================================================================
            time_delta_lqiate += time() - time_q_start  # QIATE

            # GATE
            time_g_start = time()
            if round_ == 'regular' and mcf_.p_dict['gate']:
                (_, lgate_local_dic, _, lgate_m_late_local_dic, _,
                 lgate_est_local_dic) = mcf_iv_add.gate_iv(
                    mcf_,
                    data_df,
                    weights,
                    w_late_local,
                    None, lgate_local_dic,
                    None, lgate_m_late_local_dic,
                    fold,
                    global_effects=False, local_effects=True,
                    )
            else:
                lgate_est_local_dic = None
            time_delta_lgate += time() - time_g_start

        if not only_one_fold_one_round and mcf_.int_dict['del_forest']:
            mcf_.forest[fold] = mcf_1st.forest[fold] = None
            mcf_redf.forest[fold] = None
            # Without this and the next delete, it becomes impossible to reuse
            # the same forest for several data sets, which is bad.

        if (mcf_.int_dict['mp_ray_shutdown']
            and mcf_.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

    if mcf_.int_dict['del_forest']:
        mcf_.forest = mcf_1st.forest = mcf_redf.forest = None

    del weights, weights_local_dic, weights_1st_dic, weights_redf_dic

    # ATE
    time_a_start = time()
    # Initialise variable
    late_global = late_se_global = late_local = late_se_local = None
    late_effect_list = None

    if global_effects:
        (late_global, late_se_global, late_effect_list
         ) = mcf_ate.ate_effects_print(
             mcf_, late_global_dic, y_pred_x_df, balancing_test=False,
             extra_title='(LATE-global)')

    if local_effects:
        late_local, late_se_local, late_effect_list = mcf_ate.ate_effects_print(
            mcf_, late_local_dic, y_pred_x_df, balancing_test=False,
            extra_title='(LATE-local)')

    ate_1st, ate_1st_se, ate_1st_effect_list = mcf_ate.ate_effects_print(
        mcf_1st, ate_1st_dic, d_pred_x_df, balancing_test=False,
        extra_title='(1st stage)')

    ate_redf, ate_redf_se, ate_redf_effect_list = mcf_ate.ate_effects_print(
        mcf_redf, ate_redf_dic, y_pred_x_df, balancing_test=False,
        extra_title='(reduced form)')

    time_delta_late += time() - time_a_start

    # GATE
    time_g_start = time()
    # Initialise variables
    # lgate_global = lgate_se_global = lgate_diff_global = None
    # lgate_diff_se_global = lgate_est_global_dic = None
    lgate_local = lgate_se_local = lgate_diff_local = None
    lgate_diff_se_local = None

    if mcf_.p_dict['gate']:
        # if global_effects:
        #     (lgate_global, lgate_se_global,
        #      lgate_diff_global, lgate_diff_se_global,
        #      report['fig_gate_global']
        #      ) = mcf_gateout.gate_effects_print(mcf_, lgate_global_dic,
        #                                         lgate_m_late_global_dic,
        #                                         lgate_est_global_dic,
        #                                         late_global, late_se_global,
        #                                         gate_type='GATE',
        #                                         iv=True
        #                                         )

        (lgate_local, lgate_se_local,
         lgate_diff_local, lgate_diff_se_local,
         report['fig_gate_local']
         ) = mcf_gateout.gate_effects_print(mcf_, lgate_local_dic,
                                            lgate_m_late_local_dic,
                                            lgate_est_local_dic,
                                            late_local, late_se_local,
                                            gate_type='GATE',
                                            iv=True,
                                            )
    time_delta_lgate += time() - time_g_start

    # BGATE
    time_lbgate_start = time()
    # Initialise variables
    # lbgate_global = lbgate_se_global = lbgate_diff_global = None
    # lbgate_diff_se_global = lbgate_est_global_dic = None
    lbgate_local = lbgate_se_local = lbgate_diff_local = None
    lbgate_diff_se_local = None

    if mcf_.p_dict['bgate']:
        # if global_effects:
        #     (lbgate_global, lbgate_se_global,
        #      lbgate_diff_global, lbgate_diff_se_global,
        #      report['fig_bgate_global']
        #      ) = mcf_gateout.gate_effects_print(mcf_,
        #                                         lbgate_global_dic,
        #                                         lbgate_m_late_global_dic,
        #                                         lbgate_est_global_dic,
        #                                         late_global, late_se_global,
        #                                         gate_type='BGATE',
        #                                         iv=True,
        #                                         special_txt=txt_lbg
        #                                         )  Add extra title
        (lbgate_local, lbgate_se_local,
         lbgate_diff_local, lbgate_diff_se_local,
         report['fig_bgate_local']
         ) = mcf_gateout.gate_effects_print(mcf_,
                                            lbgate_local_dic,
                                            lbgate_m_late_local_dic,
                                            lbgate_est_local_dic,
                                            late_local, late_se_local,
                                            gate_type='BGATE',
                                            iv=True,
                                            special_txt=txt_lbg
                                            )
    time_delta_lbgate += time() - time_lbgate_start

    # CBGATE
    time_cbg_start = time()
    # Initialise variables
    # lcbgate_global = lcbgate_se_global = lcbgate_diff_global = None
    # lcbgate_diff_se_global = lcbgate_est_global_dic = None
    lcbgate_local = lcbgate_se_local = lcbgate_diff_local = None
    lcbgate_diff_se_local = lcbgate_est_local_dic = None
    if mcf_.p_dict['cbgate']:
        # if global_effects:
        #     (lcbgate_global, lcbgate_se_global,
        #      lcbgate_diff_global, lcbgate_diff_se_global,
        #      report['fig_cbgate_global']
        #      ) = mcf_gateout.gate_effects_print(mcf_,
        #                                         lcbgate_global_dic,
        #                                         lcbgate_m_late_global_dic,
        #                                         lcbgate_est_global_dic,
        #                                         late_global, late_se_global,
        #                                         gate_type='CBGATE',
        #                                         iv=True,
        #                                         special_txt=txt_lcbg
        #                                         )

        (lcbgate_local, lcbgate_se_local,
         lcbgate_diff_local, lcbgate_diff_se_local,
         report['fig_cbgate_local']
         ) = mcf_gateout.gate_effects_print(mcf_,
                                            lcbgate_local_dic,
                                            lcbgate_m_late_local_dic,
                                            lcbgate_est_local_dic,
                                            late_local, late_se_local,
                                            gate_type='CBGATE',
                                            iv=True,
                                            special_txt=txt_lcbg
                                            )

    time_delta_lcbgate += time() - time_cbg_start
    # Collect some information for results_dic
    if (mcf_.p_dict['gate'] or mcf_.p_dict['bgate']
            or mcf_.p_dict['cbgate']):
        # if global_effects:
        #     gate_names_values = mcf_gateout.get_names_values(
        #         mcf_,
        #         lgate_est_global_dic, lbgate_est_global_dic,
        #         lcbgate_est_global_dic
        #         )
        # else:
        gate_names_values = mcf_gateout.get_names_values(
            mcf_,
            lgate_est_local_dic, lbgate_est_local_dic,
            lcbgate_est_local_dic
            )
    else:
        gate_names_values = None

    # QIATE (not yet(?) available)
    time_q_start = time()
    # if mcf_.p_dict['qiate']: Not yet implemented
    #     (qliate, qliate_se, qliate_diff, qliate_diff_se, qliate_mopp,
    #      qliate_mopp_se, report['fig_qiate']) = mcf_qiate.qiate_effects_print(
    #          mcf_, qliate_dic, qliate_m_med_dic, qliate_m_opp_dic,
    #          qliate_est_dic)

    # else:
    qliate = qliate_se = qliate_diff = qliate_diff_se = None
    # qliate_est_dic = None
    qliate_mopp = qliate_mopp_se = None
    time_delta_lqiate += time() - time_q_start  # QIATE

    # IATE
    time_i_start = time()
    if mcf_.p_dict['iate']:
        (liate, liate_se, liate_eff, liate_names_dic, liate_df,
         report['iate_text']) = mcf_iate.iate_effects_print(
             mcf_, liate_dic, liate_m_late_dic, liate_eff_dic, y_pred_x_df,
             extra_title='(LIATE)', iv=True)
        data_df.reset_index(drop=True, inplace=True)
        liate_df.reset_index(drop=True, inplace=True)
        liate_pred_df = pd.concat([data_df, liate_df], axis=1)
    else:
        liate_eff = liate = liate_se = liate_df = liate_pred_df = None
        liate_names_dic = None

    # IATE (1st stage)
    time_i_start = time()
    if mcf_.p_dict['iate']:
        (iate_1st, iate_1st_se, iate_1st_eff, iate_1st_names_dic, iate_1st_df,
         report['iate_1st_text']) = mcf_iate.iate_effects_print(
             mcf_1st, iate_1st_dic, iate_m_ate_1st_dic, iate_eff_1st_dic,
             d_pred_x_df, extra_title='(1st stage)', iv=False)
        data_df.reset_index(drop=True, inplace=True)
        iate_1st_df.reset_index(drop=True, inplace=True)
        iate_1st_pred_df = pd.concat([data_df, iate_1st_df], axis=1)
    else:
        iate_1st_eff = iate_1st = iate_1st_se = iate_1st_df = None
        iate_1st_pred_df = iate_1st_names_dic = None

    # IATE (reduced form)
    time_i_start = time()
    if mcf_.p_dict['iate']:
        (iate_redf, iate_redf_se, iate_redf_eff, iate_redf_names_dic,
         iate_redf_df, report['iate_redf_text']) = mcf_iate.iate_effects_print(
             mcf_redf, iate_redf_dic, iate_m_ate_redf_dic, iate_eff_redf_dic,
             y_pred_x_df, extra_title='(reduced form)', iv=False)
        data_df.reset_index(drop=True, inplace=True)
        iate_redf_df.reset_index(drop=True, inplace=True)
        iate_redf_pred_df = pd.concat([data_df, iate_redf_df], axis=1)
    else:
        iate_redf_eff = iate_redf = iate_redf_se = iate_redf_df = None
        iate_redf_pred_df = iate_redf_names_dic = None

    time_delta_liate += time() - time_i_start

    # Balancing test
    time_b_start = time()
    if mcf_.p_dict['bt_yes']:
        (bala_1st, bala_1st_se, bala_1st_effect_list
         ) = mcf_ate.ate_effects_print(mcf_, bala_1st_dic, None,
                                       balancing_test=True)
        (bala_redf, bala_redf_se, bala_redf_effect_list
         ) = mcf_ate.ate_effects_print(mcf_, bala_redf_dic, None,
                                       balancing_test=True)
    else:
        bala_1st = bala_1st_se = bala_1st_effect_list = None
        bala_redf = bala_redf_se = bala_redf_effect_list = None
    time_delta_lbala += time() - time_b_start

    results_both = {
        'ate_1st': ate_1st, 'ate_1st_se': ate_1st_se,
        'ate 1st_effect_list': ate_1st_effect_list,
        'ate_redf': ate_redf, 'ate_redf_se': ate_redf_se,
        'ate redf_effect_list': ate_redf_effect_list,
        'qiate': qliate, 'qiate_se': qliate_se,
        'qiate_diff': qliate_diff, 'qiate_diff_se': qliate_diff_se,
        'qiate_mopp': qliate_mopp, 'qiate_mopp_se': qliate_mopp_se,
        'iate': liate, 'iate_se': liate_se, 'iate_eff': liate_eff,
        'iate_1st': iate_1st, 'iate_1st_se': iate_1st_se,
        'iate_redf': iate_redf, 'iate_redf_se': iate_redf_se,
        'iate_1st_eff': iate_1st_eff, 'iate_redf_eff': iate_redf_eff,
        'iate_data_df': liate_pred_df, 'iate_1st_pred_df': iate_1st_pred_df,
        'iate_redf_pred_df': iate_redf_pred_df,
        'iate_names_dic': liate_names_dic,
        'iate_1st_names_dic': iate_1st_names_dic,
        'iate_redf_names_dic': iate_redf_names_dic,
        'bala_1st': bala_1st, 'bala_1st_se': bala_1st_se,
        'bala_1st_effect_list': bala_1st_effect_list,
        'bala_redf': bala_redf, 'bala_redf_se': bala_redf_se,
        'bala_redf_effect_list': bala_redf_effect_list,
        'ate_effect_list': late_effect_list,
        'gate_names_values': gate_names_values,
        'common_support_probabilities: ': cs_pred_prob,
        'path_output': mcf_.gen_dict['outpath']
        }

    if global_effects:
        results_global = {
            'ate': late_global, 'ate_se': late_se_global,
            'gate': None,  # lgate_global,
            'gate_se': None,  # lgate_se_global,
            'gate_diff': None,  # lgate_diff_global,
            'gate_diff_se': None,  # lgate_diff_se_global,
            'cbgate': None,  # lcbgate_global, 'cbgate_se': lcbgate_se_global,
            'cbgate_diff': None,  # lcbgate_diff_global,
            'cbgate_diff_se': None,  # lcbgate_diff_se_global,
            'bgate': None,   # lbgate_global,
            'bgate_se': None,  # lbgate_se_global,
            'bgate_diff': None,  # lbgate_diff_global,
            'bgate_diff_se': None,  # lbgate_diff_se_global,
            }
        results_global.update(deepcopy(results_both))

    if local_effects:
        results_local = {
            'ate': late_local, 'ate_se': late_se_local,
            'gate': lgate_local, 'gate_se': lgate_se_local,
            'gate_diff': lgate_diff_local, 'gate_diff_se': lgate_diff_se_local,
            'cbgate': lcbgate_local, 'cbgate_se': lcbgate_se_local,
            'cbgate_diff': lcbgate_diff_local,
            'cbgate_diff_se': lcbgate_diff_se_local,
            'bgate': lbgate_local, 'bgate_se': lbgate_se_local,
            'bgate_diff': lbgate_diff_local,
            'bgate_diff_se': lbgate_diff_se_local,
            }
        results_local.update(deepcopy(results_both))

    if mcf_.int_dict['with_output']:
        if global_effects:
            results_global_dic = deepcopy(results_global)
            del results_global_dic['iate_data_df']
            report['mcf_pred_results_global'] = results_global_dic

        if local_effects:
            results_local_dic = deepcopy(results_local)
            del results_local_dic['iate_data_df']
            report['mcf_pred_results_local'] = results_local_dic  # in report
        mcf_.report['predict_list'].append(report.copy())

    time_end = time()

    if mcf_.int_dict['with_output']:
        time_string = [
            'Data preparation and stats II:                  ',
            'Common support:                                 ',
            'Local centering (recoding of Y):                ',
            'First stage scaling :                           ',
            'Weights (for IATE and LIATE):                   ',
            'LATEs:                                          ',
            'LGATEs:                                         ',
            'LBGATEs:                                        ',
            'LCBGATEs:                                       ',
            'LQIATEs:                                        ',
            'LIATEs:                                         ',
            'Balancing test:                                 ',
            '\nTotal time prediction:                          ',
            ]
        time_difference = [
            time_1 - time_start, time_2 - time_1, time_3 - time_2,
            time_1stscale - time_3,
            time_delta_weight, time_delta_late, time_delta_lgate,
            time_delta_lbgate, time_delta_lcbgate, time_delta_lqiate,
            time_delta_liate, time_delta_lbala, time_end - time_start
            ]
        mcf_ps.print_mcf(mcf_.gen_dict, mcf_.time_strings['time_train'])
        time_pred = mcf_ps.print_timing(
            mcf_.gen_dict, 'Prediction', time_string, time_difference,
            summary=True
            )
        mcf_.time_strings['time_pred'] = time_pred

    return results_global, results_local


def ensure_binary_binary(mcf_: 'ModifiedCausalForest',
                         data_df: pd.DataFrame
                         ) -> pd.DataFrame:
    """Check if treatment and instrument are binary."""
    # Check data is all numeric and convert it integers
    int_series = pd.to_numeric(
        data_df[mcf_.var_dict['iv_name']].squeeze(), errors='raise').astype(int)

    # Check if instrument is binary
    if not set(int_series.unique()).issubset({0, 1}):
        raise ValueError('Instrument must be binary.')
    data_df[mcf_.var_dict['iv_name']] = int_series.to_frame()

    # Check if treatment is binary
    d_series = data_df[mcf_.var_dict['d_name']].squeeze()
    if not set(d_series.unique()).issubset({0, 1}):
        raise ValueError('Treatment must be binary.')

    return data_df


def print_dics(mcf_: 'ModifiedCausalForest',
               mcf_1st: 'ModifiedCausalForest',
               mcf_redf: 'ModifiedCausalForest'
               ):
    """Print values of dictionaries."""
    mcf_ps.print_dic_values_all(mcf_, summary_top=True, summary_dic=False,
                                train=False, title='IV')
    mcf_ps.print_dic_values_all(mcf_1st, summary_top=False,
                                summary_dic=False, train=False,
                                title='IV (1st stage)'
                                )
    mcf_ps.print_dic_values_all(mcf_redf, summary_top=True,
                                summary_dic=False, train=False,
                                title='IV (reduced form)')


def update_mcf_(mcf_: 'ModifiedCausalForest', mcf_1st: 'ModifiedCausalForest'):
    """Update mcf_ instance."""
    # Copy some attributes of mcf_1st to mcf_
    mcf_.cf_dict['folds'] = mcf_1st.cf_dict['folds']
    mcf_.cf_dict['est_rounds'] = mcf_1st.cf_dict['est_rounds']
    mcf_.cf_dict['x_name_mcf'] = mcf_1st.cf_dict['x_name_mcf'].copy()


def update_inst_data(mcf_: 'ModifiedCausalForest',
                     mcf_1st: 'ModifiedCausalForest',
                     mcf_redf: 'ModifiedCausalForest',
                     data_df: pd.DataFrame
                     ) -> pd.DataFrame:
    """Update instances and Dataframe."""
    data_df, _ = mcf_data.data_frame_vars_lower(data_df)
    data_df = mcf_init_update.p_update_pred(mcf_, data_df)
    mcf_init_update.p_update_pred(mcf_1st, data_df)
    mcf_init_update.p_update_pred(mcf_redf, data_df)

    # Check treatment data
    if mcf_.p_dict['d_in_pred']:
        data_df = mcf_data.check_recode_treat_variable(mcf_, data_df)

    mcf_init_update.int_update_pred(mcf_, len(data_df))
    mcf_init_update.post_update_pred(mcf_, data_df)
    mcf_init_update.int_update_pred(mcf_1st, len(data_df))
    mcf_init_update.post_update_pred(mcf_1st, data_df)
    mcf_init_update.int_update_pred(mcf_redf, len(data_df))
    mcf_init_update.post_update_pred(mcf_redf, data_df)

    return data_df


def local_center(mcf_1st: 'ModifiedCausalForest',
                 mcf_redf: 'ModifiedCausalForest',
                 data_df: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute centering variables."""
    (_, _, d_pred_x_df, _) = mcf_lc.local_centering(mcf_1st, data_df, None,
                                                    train=False,
                                                    title='(1st stage)'
                                                    )
    (_, _, y_pred_x_df, _) = mcf_lc.local_centering(mcf_redf, data_df, None,
                                                    train=False,
                                                    title='(reduced form)')

    return y_pred_x_df, d_pred_x_df


def get_forests(mcf_: 'ModifiedCausalForest',
                mcf_1st: 'ModifiedCausalForest',
                mcf_redf: 'ModifiedCausalForest',
                only_one_fold_one_round: bool,
                fold: int, round_: bool
                ) -> tuple[dict, dict]:
    """Extract forsts."""
    if only_one_fold_one_round:
        forest_1st_dic = mcf_1st.forest[fold][0]
        forest_redf_dic = mcf_redf.forest[fold][0]
    else:
        forest_1st_dic = deepcopy(
            mcf_1st.forest[fold][0 if round_ == 'regular' else 1])
        forest_redf_dic = deepcopy(
            mcf_redf.forest[fold][0 if round_ == 'regular' else 1])
    if mcf_.int_dict['with_output'] and mcf_.int_dict['verbose']:
        print(f'\n\nWeight maxtrix (all effects) {fold+1} /',
              f'{mcf_.cf_dict["folds"]} forests, {round_}')

    return forest_1st_dic, forest_redf_dic


def new_instances_iv_train(mcf_: 'ModifiedCausalForest'
                           ) -> tuple['ModifiedCausalForest',
                                      'ModifiedCausalForest'
                                      ]:
    """Create instances for 1st stage and reduced form."""
    mcf_1st = deepcopy(mcf_)
    mcf_1st.var_dict['y_name'] = mcf_.var_dict['d_name'].copy()
    mcf_1st.var_dict['y_tree_name'] = mcf_1st.var_dict['y_name']
    mcf_1st.var_dict['d_name'] = mcf_.var_dict['iv_name'].copy()
    mcf_1st.int_dict['verbose'] = False

    mcf_redf = deepcopy(mcf_)
    mcf_redf.var_dict['d_name'] = mcf_.var_dict['iv_name'].copy()
    mcf_redf.int_dict['verbose'] = False

    return mcf_1st, mcf_redf


def prepare_data_pred(mcf_: 'ModifiedCausalForest',
                      data_df: pd.DataFrame,
                      report: dict
                      ) -> tuple[pd.DataFrame, dict]:
    """Prepare data and print some descriptive statistics."""
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

    # Descriptives by treatment
    if (mcf_.p_dict['d_in_pred'] and mcf_.int_dict['descriptive_stats']
            and mcf_.int_dict['with_output']):
        mcf_ps.desc_by_treatment(mcf_, data_df, summary=False, stage=3)

    return data_df, report


def common_support_pred(mcf_: 'ModifiedCausalForest',
                        data_df: pd.DataFrame,
                        report: dict
                        ) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    """Adjust common support based on training information."""
    if mcf_.cs_dict['type']:
        (data_df, _, report['cs_p_share_deleted'],
         report['cs_p_obs_remain'],
         _, _, _, cs_pred_prob) = mcf_cs.common_support(mcf_, data_df, None,
                                                        train=False
                                                        )
    else:
        cs_pred_prob = None

    data_df = data_df.copy().reset_index(drop=True)
    if (mcf_.p_dict['d_in_pred'] and mcf_.int_dict['descriptive_stats']
            and mcf_.int_dict['with_output']):
        mcf_ps.desc_by_treatment(mcf_, data_df, summary=True, stage=3)

    return data_df, report, cs_pred_prob
