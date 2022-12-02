"""Created on Fri Apr  3 11:05:15 2020.

Contains the functions needed for the running all parts of the programme
@author: MLechner
-*- coding: utf-8 -*-
"""
import copy
import sys
import time
import os
from multiprocessing import freeze_support

import pandas as pd
import numpy as np

from mcf import general_purpose as gp
from mcf import general_purpose_system_files as gp_sys
from mcf import mcf_general_purpose as mcf_gp
from mcf import mcf_init_functions as mcf_init
from mcf import mcf_init_add_functions as mcf_init_add
from mcf import mcf_data_functions as mcf_data
from mcf import mcf_loccent_functions as mcf_lc
from mcf import mcf_forest_functions as mcf_forest
from mcf import mcf_vi_functions as mcf_vi
from mcf import mcf_cs_functions as mcf_cs
from mcf import mcf_weight_functions as mcf_w
from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_gate_functions as mcf_gate
from mcf import mcf_gate_tables_functions as mcf_gate_tables
from mcf import mcf_iate_functions as mcf_iate
from mcf import mcf_iate_add_functions as mcf_iate_add
from mcf import mcf_iate_cv_functions as mcf_iate_cv
from mcf import mcf_forest_add_functions as mcf_forest_add


def modified_causal_forest(
        id_name=None, cluster_name=None, w_name=None, d_name=None,
        d_type='discrete', y_tree_name=None, y_name=None, x_name_ord=None,
        x_name_unord=None, x_name_always_in_ord=None, z_name_list=None,
        x_name_always_in_unord=None, z_name_split_ord=None,
        z_name_split_unord=None, z_name_mgate=None, z_name_amgate=None,
        x_name_remain_ord=None, x_name_remain_unord=None,
        x_balance_name_ord=None, x_balance_name_unord=None,
        alpha_reg_min=0.1, alpha_reg_max=0.2, alpha_reg_grid=2,
        atet_flag=False, balancing_test=True,
        boot=1000, ci_level=0.9, check_perfectcorr=True,
        choice_based_sampling=False, choice_based_weights=None,
        clean_data_flag=True, cluster_std=False, cond_var_flag=True,
        ct_grid_nn=10, ct_grid_w=10, ct_grid_dr=100,
        datpfad=None, fs_other_sample_share=0.2,
        fs_other_sample=True, fs_rf_threshold=0, fs_yes=None,
        forest_files=None,
        gates_minus_previous=False, gates_smooth=True,
        gates_smooth_bandwidth=1,
        gates_smooth_no_evaluation_points=50, gatet_flag=False,
        gmate_no_evaluation_points=50, gmate_sample_share=None,
        iate_flag=True, iate_se_flag=True, iate_eff_flag=True,
        iate_cv_flag=False, iate_cv_folds=5,
        indata=None, knn_flag=True, knn_min_k=10,
        knn_const=1, l_centering=True, l_centering_share=0.25,
        l_centering_cv_k=5, l_centering_new_sample=False,
        l_centering_replication=True, l_centering_undo_iate=True,
        m_min_share=-1, m_max_share=-1, m_grid=2, m_random_poisson=True,
        match_nn_prog_score=True, max_cats_z_vars=None, max_weight_share=0.05,
        mce_vart=1, min_dummy_obs=10,  mp_parallel=None,
        n_min_grid=1, n_min_min=-1, n_min_max=-1, n_min_treat=3,
        nn_main_diag_only=False, nw_kern_flag=1, nw_bandw=None,
        outfiletext=None, outpfad=None, output_type=2,
        panel_data=False, panel_in_rf=True,
        preddata=None, p_diff_penalty=None,
        post_bin_corr_threshold=0.1, post_bin_corr_yes=True,
        post_est_stats=True, post_plots=True, post_kmeans_yes=True,
        post_kmeans_max_tries=1000, post_kmeans_no_of_groups=None,
        post_kmeans_replications=10, post_random_forest_vi=True,
        post_relative_to_first_group_only=True,
        predict_mcf=True, reduce_prediction=False, reduce_prediction_share=0.5,
        reduce_split_sample=False, reduce_split_sample_pred_share=0.5,
        reduce_training=False, reduce_training_share=0.5,
        reduce_largest_group_train=False,
        reduce_largest_group_train_share=0.5, save_forest=False,
        screen_covariates=True, se_boot_ate=False, se_boot_gate=False,
        se_boot_iate=False, random_thresholds=None,
        subsample_factor_forest=None, subsample_factor_eval=None,
        support_check=1, support_min_p=None, support_quantil=1,
        support_max_del_train=0.5, support_adjust_limits=None, train_mcf=True,
        variable_importance_oob=False, weighted=False,
        _share_forest_sample=0.5, _weight_as_sparse=True,
        _mp_vim_type=None, _mp_weights_type=1, _mp_weights_tree_batch=None,
        _verbose=True, _fontsize=2, _descriptive_stats=True,
        _no_filled_plot=20, _show_plots=True, _smaller_sample=0,
        _with_output=True, _dpi=500, _max_cats_cont_vars=None,
        _max_save_values=50, _seed_sample_split=67567885, _mp_ray_del=None,
        _mp_ray_shutdown=None, _mp_ray_objstore_multiplier=None,
        _ray_or_dask='ray', _return_iate_sp=False):
    """Compute the honest causal/random forest (based on mcf)."""
    freeze_support()

    # Some temporary operational flags may be changed at some point
    _boot_by_boot = 10    # Build forest x-by-x or in larger groups (not Ray)
    _obs_by_obs = False      # Compute IATE obs by obs or in chuncks
    _max_elements_per_split = 100 * 10e5  # reduce if breakdown (weights)
    _load_old_forest = False  # Useful for testing to save time
    _no_ray_in_forest_building = False

# Collect vars in a dictionary
    variable_dict = mcf_init_add.make_user_variable(
        id_name, cluster_name, w_name, d_name, y_tree_name, y_name, x_name_ord,
        x_name_unord, x_name_always_in_ord, z_name_list,
        x_name_always_in_unord, z_name_split_ord, z_name_split_unord,
        z_name_mgate, z_name_amgate, x_name_remain_ord, x_name_remain_unord,
        x_balance_name_ord, x_balance_name_unord)

# use smaller random sample (usually for testing purposes)
    if isinstance(_smaller_sample, (int, float)) and 0 < _smaller_sample < 1:
        if train_mcf:
            gp.randomsample(datpfad, indata + '.csv', 'smaller_indata.csv',
                            _smaller_sample, True, seed=_seed_sample_split)
        if preddata is None:
            preddata = 'smaller_indata'
        if predict_mcf and preddata != indata:
            gp.randomsample(datpfad, preddata + '.csv', 'smaller_preddata.csv',
                            _smaller_sample, True, seed=_seed_sample_split)
            preddata = 'smaller_preddata'
        else:
            preddata = 'smaller_indata'
        indata = 'smaller_indata'
    iate_cv_flag, iate_cv_folds = mcf_iate_cv.check_if_iate_cv(
        iate_cv_flag, iate_cv_folds, preddata, indata, iate_flag, d_type)

# set values for control variables
    controls_dict = mcf_init_add.controls_into_dic(
        mp_parallel, _mp_vim_type, output_type, outpfad,
        datpfad, indata, preddata, outfiletext, screen_covariates,
        n_min_grid, check_perfectcorr, n_min_min, clean_data_flag,
        min_dummy_obs, mce_vart, p_diff_penalty, boot, n_min_max,
        support_min_p, weighted, support_check, support_quantil,
        subsample_factor_forest, subsample_factor_eval, m_min_share, m_grid,
        m_random_poisson, alpha_reg_min, alpha_reg_max, alpha_reg_grid,
        random_thresholds, knn_min_k, _share_forest_sample,
        _descriptive_stats, m_max_share, max_cats_z_vars,
        variable_importance_oob, balancing_test, choice_based_sampling,
        knn_const, choice_based_weights, nw_kern_flag, post_kmeans_max_tries,
        cond_var_flag, knn_flag, nw_bandw, panel_data, _max_cats_cont_vars,
        cluster_std, fs_yes, fs_other_sample_share, gatet_flag,
        fs_other_sample, post_bin_corr_yes, panel_in_rf, fs_rf_threshold,
        post_plots, post_est_stats, post_relative_to_first_group_only,
        post_kmeans_yes, atet_flag, post_bin_corr_threshold,
        post_kmeans_no_of_groups, post_kmeans_replications, _with_output,
        _max_save_values, nn_main_diag_only, _fontsize, _dpi, ci_level,
        max_weight_share, save_forest, l_centering, l_centering_share,
        l_centering_new_sample, l_centering_cv_k, post_random_forest_vi,
        gmate_no_evaluation_points, gmate_sample_share, _no_filled_plot,
        gates_smooth, gates_smooth_bandwidth,
        gates_smooth_no_evaluation_points, _show_plots, _weight_as_sparse,
        _mp_weights_type, _mp_weights_tree_batch, _boot_by_boot, _obs_by_obs,
        _max_elements_per_split, _mp_ray_objstore_multiplier,
        _verbose, _no_ray_in_forest_building, predict_mcf, train_mcf,
        forest_files, match_nn_prog_score, se_boot_ate, se_boot_gate,
        se_boot_iate, support_max_del_train, _mp_ray_del, _mp_ray_shutdown,
        reduce_split_sample, reduce_split_sample_pred_share, reduce_training,
        reduce_training_share, reduce_prediction, reduce_prediction_share,
        reduce_largest_group_train, reduce_largest_group_train_share,
        iate_flag, iate_se_flag, l_centering_undo_iate, d_type, ct_grid_nn,
        ct_grid_w, ct_grid_dr, support_adjust_limits, l_centering_replication,
        iate_eff_flag, _return_iate_sp, iate_cv_flag, iate_cv_folds,
        n_min_treat, gates_minus_previous, _ray_or_dask)
# Set defaults for many control variables of the MCF & define variables
    (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
     pred_outfile, pot_y_eff, iate_eff, names_pot_iate, preddata_all
     ) = modified_causal_forest_master(controls_dict, variable_dict,
                                       _load_old_forest, _seed_sample_split)
    if iate_cv_flag:
        iate_cv_file, iate_cv_names = mcf_iate_cv.iate_cv_proc(
            controls_dict, variable_dict, seed_sample_split=_seed_sample_split,
            with_output=controls_dict['with_output'],
            file_with_out_path=pred_outfile)
    else:
        iate_cv_file = iate_cv_names = None
    return (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
            pred_outfile, pot_y_eff, iate_eff, names_pot_iate, preddata_all,
            iate_cv_file, iate_cv_names)


def modified_causal_forest_master(controls_dict, variable_dict,
                                  _load_old_forest, _seed_sample_split):
    """Compute the honest causal/random forest (based on mcf)."""
    # Some descriptive stats of input and direction of output file
    time1 = time.time()
    c_dict, v_dict, text_to_print = mcf_init.get_controls(controls_dict,
                                                          variable_dict)
    ofs = c_dict['outfilesummary']
    if c_dict['with_output']:
        if c_dict['print_to_file']:
            orig_stdout = sys.stdout
            if c_dict['print_to_terminal']:
                sys.stdout = gp.OutputTerminalFile(c_dict['outfiletext'])
            else:
                outfiletext = open(c_dict['outfiletext'], 'a',
                                   encoding="utf-8")
                sys.stdout = outfiletext
    if c_dict['reduce_split_sample']:
        outdatei1 = c_dict['indata'][:-4] + 'estredu.csv'
        outdatei2 = c_dict['preddata'][:-4] + 'predredu.csv'
        c_dict['indata'], c_dict['preddata'], _ = gp.sample_split_2_3(
            c_dict['indata'], outdatei1,
            1-c_dict['reduce_split_sample_pred_share'], outdatei2,
            c_dict['reduce_split_sample_pred_share'],
            random_seed=_seed_sample_split, with_output=c_dict['with_output'])
    if c_dict['with_output'] and c_dict['verbose']:
        if c_dict['train_mcf']:
            if text_to_print is not None:
                print(text_to_print, '{c_dict["n_min_min"]:<5} ')
            print('\nParameter for MCF:')
            gp.print_dic(c_dict)
            print('\nVariables used:')
            gp.print_dic(v_dict)
            print('-' * 80)
            print('Estimator used: ', c_dict['estimator_str'])
            print('\nValue of penalty multiplier:',
                  f'{c_dict["mtot_p_diff_penalty"]:8.3f}')
            print('-' * 80)
            gp.print_f(ofs,
                       '=' * 80, '\nEstimation of the Modified Causal Forest',
                       '\n' + '-' * 80, f'\nIndata: {c_dict["indata"]}',
                       f'\nPrediction Data: {c_dict["indata"]}',
                       '\n' + '-' * 80, '\nThis is only the summary file. ',
                       'For the complete results check ',
                       c_dict['outfiletext'], '\n' + '-' * 80)
            if c_dict['desc_stat']:
                gp.print_descriptive_stats_file(c_dict['indata'], 'all',
                                                c_dict['print_to_file'])
            gp.check_all_vars_in_data(c_dict['indata'],
                                      v_dict['names_to_check_train'])
        if c_dict['indata'] != c_dict['preddata'] and c_dict['pred_mcf']:
            if c_dict['desc_stat']:
                gp.print_descriptive_stats_file(
                    c_dict['preddata'], 'all', c_dict['print_to_file'])
            gp.check_all_vars_in_data(c_dict['preddata'],
                                      v_dict['names_to_check_pred'])
    else:
        c_dict['print_to_file'] = False

    if not c_dict['train_mcf']:
        loaded_tuple = gp_sys.save_load(c_dict['save_forest_file_pickle'],
                                        save=False,
                                        output=c_dict['with_output'])
        forest, x_name_mcf = loaded_tuple[0], loaded_tuple[1]
        c_dict['max_cats_z_vars'] = loaded_tuple[2]
        d_in_values, no_val_dict = loaded_tuple[3], loaded_tuple[4]
        q_inv_dict, q_inv_cr_dict = loaded_tuple[5], loaded_tuple[6]
        prime_values_dict, unique_val_dict = loaded_tuple[7], loaded_tuple[8]
        common_support_list = loaded_tuple[9]
        z_new_name_dict, z_new_dic_dict = loaded_tuple[10], loaded_tuple[11]
        c_dict['max_cats_cont_vars'] = loaded_tuple[12]
        lc_forest = loaded_tuple[13]
        old_y_name, new_y_name = loaded_tuple[14], loaded_tuple[15]
        v_dict['y_tree_name'] = loaded_tuple[16]
        c_dict['l_centering'] = loaded_tuple[17]
        if c_dict['l_centering'] is False:
            c_dict['l_centering_uncenter'] = False
    else:
        d_in_values = no_val_dict = q_inv_dict = q_inv_cr_dict = None
        prime_values_dict = unique_val_dict = common_support_list = None
        z_new_name_dict = z_new_dic_dict = lc_forest = None
        old_y_name = new_y_name = None

# Prepare data: Add and recode variables for GATES (Z)
#               Recode categorical variables to prime numbers, cont. vars
    (v_dict, var_x_type, var_x_values, c_dict, indata_with_z, predata_with_z,
     d_in_values, no_val_dict, q_inv_dict, q_inv_cr_dict, prime_values_dict,
     unique_val_dict, z_new_name_dict, z_new_dic_dict
     ) = mcf_data.create_xz_variables(
         v_dict, c_dict, False, d_in_values, no_val_dict, q_inv_dict,
         q_inv_cr_dict, prime_values_dict, unique_val_dict,
         z_new_name_dict, z_new_dic_dict)

    if c_dict['with_output'] and c_dict['verbose']:
        print('\nDictionary for recoding of categorical variables into primes')
        key_list_unique_val_dict = list(unique_val_dict)
        key_list_prime_values_dict = list(prime_values_dict)
        for key_nr, key in enumerate(key_list_unique_val_dict):
            print(key)
            print(unique_val_dict[key])
            print(key_list_prime_values_dict[key_nr])  # order may differ!
            print(prime_values_dict[key_list_prime_values_dict[key_nr]])
    # Remove missing and keep only variables needed for further analysis
    if c_dict['clean_data_flag']:
        if c_dict['train_mcf']:
            namen1_to_inc = gp.add_var_names(
                v_dict['id_name'], v_dict['y_name'], v_dict['cluster_name'],
                v_dict['w_name'], v_dict['d_name'], v_dict['y_tree_name'],
                v_dict['x_balance_name'], v_dict['x_name'])
            if c_dict['d_type'] == 'continuous':
                namen1_to_inc.append(*v_dict['d_grid_nn_name'])
            indata2 = gp.clean_reduce_data(
                indata_with_z, c_dict['indata2_temp'],
                namen1_to_inc, c_dict['with_output'],
                c_dict['desc_stat'], c_dict['print_to_file'])
        if indata_with_z != predata_with_z and c_dict['pred_mcf']:
            namen2_to_inc = gp.add_var_names(
                v_dict['id_name'], v_dict['w_name'],
                v_dict['x_balance_name'], v_dict['x_name'])
            if c_dict['atet_flag']:
                namen2_to_inc.extend(v_dict['d_name'])
            preddata2 = gp.clean_reduce_data(
                predata_with_z, c_dict['preddata2_temp'], namen2_to_inc,
                c_dict['with_output'], c_dict['desc_stat'],
                c_dict['print_to_file'])
        else:
            preddata2 = predata_with_z
    else:
        indata2, preddata2 = indata_with_z, predata_with_z
# get rid of variables that do not have enough independent variation
    if c_dict['train_mcf']:
        if c_dict['screen_covariates']:
            _, x_variables_out = gp.screen_variables(
                indata2, v_dict['x_name'], c_dict['check_perfectcorr'],
                c_dict['min_dummy_obs'], c_dict['with_output'])
            # if (x_variables_out != []):
            if x_variables_out:
                v_dict, var_x_type, var_x_values, c_dict = (
                    mcf_data.adjust_variables(v_dict, var_x_type, var_x_values,
                                              c_dict, x_variables_out))
        time11 = time.time()
        lc_forest = None
        if c_dict['l_centering'] and c_dict['l_centering_new_sample']:
            if c_dict['with_output'] and c_dict['verbose']:
                gp.print_f(ofs, 'Local centering with differnt sample')
            l_cent_sample, nonlc_sample, _ = gp.sample_split_2_3(
                indata2, c_dict['lc_sample'], c_dict['l_centering_share'],
                c_dict['nonlc_sample'], 1-c_dict['l_centering_share'],
                random_seed=_seed_sample_split,
                with_output=c_dict['with_output'])
            (indata2, old_y_name, new_y_name, lc_forest
             ) = mcf_lc.local_centering_new_sample(
                l_cent_sample, nonlc_sample, v_dict, var_x_type, c_dict,
                seed=_seed_sample_split)
            v_dict = mcf_data.adjust_y_names(v_dict, old_y_name, new_y_name,
                                             c_dict['with_output'])
        time12 = time.time()
        # Sample splitting
        share1 = (1 - c_dict['fs_other_sample_share']
                  ) * c_dict['share_forest_sample']
        share2 = 1 - c_dict['fs_other_sample_share'] - share1
        tree_sample, fill_y_sample, fs_sample = gp.sample_split_2_3(
            indata2, c_dict['tree_sample_temp'], share1,
            c_dict['fill_y_sample_temp'], share2, c_dict['fs_sample_temp'],
            c_dict['fs_other_sample_share'],
            _seed_sample_split, c_dict['with_output'])
        if c_dict['l_centering'] and not c_dict['l_centering_new_sample']:
            if c_dict['with_output'] and c_dict['verbose']:
                gp.print_f(ofs, 'Local centering with cross-fitting')
            time11 = time.time()
            if c_dict['fs_yes'] and c_dict['fs_other_sample']:
                files_to_center = (tree_sample, fill_y_sample, fs_sample)
            else:
                files_to_center = (tree_sample, fill_y_sample)
            old_y_name, new_y_name, lc_forest = mcf_lc.local_centering_cv(
                files_to_center, v_dict, var_x_type, c_dict,
                seed=_seed_sample_split)
            time12 = time.time()
            v_dict = mcf_data.adjust_y_names(v_dict, old_y_name, new_y_name,
                                             c_dict['with_output'])
        if (c_dict['fs_yes'] == 1) and (c_dict['fs_other_sample'] == 0):
            fs_sample = tree_sample
        if c_dict['indata'] == c_dict['preddata']:
            preddata2 = fill_y_sample
            if c_dict['with_output'] and c_dict['verbose']:
                print('\nBoth input samples are the same and sample splitting',
                      'is activated. Therefore, the part of the input that is',
                      'used for estimating the forest is not used as',
                      'reference sample to get more reliable inference')
# Descriptive statistics by treatment for outcomes and balancing variables
        if c_dict['with_output'] and c_dict['desc_stat']:
            variables_to_desc = [*v_dict['y_name'], *v_dict['x_balance_name']]
            d_name = (v_dict['d_grid_nn_name']
                      if c_dict['d_type'] == 'continuous'
                      else v_dict['d_name'])
            if c_dict['l_centering'] and not c_dict['l_centering_new_sample']:
                mcf_gp.statistics_by_treatment(
                    tree_sample, d_name, variables_to_desc,
                    c_dict['d_type'] == 'continuous')
            else:
                mcf_gp.statistics_by_treatment(
                    indata2, d_name, variables_to_desc,
                    c_dict['d_type'] == 'continuous')
            mcf_data.variable_features(var_x_type, var_x_values)
# Common support
    if c_dict['common_support'] > 0:
        if not c_dict['train_mcf']:
            tree_sample = fill_y_sample = fs_sample = None
            prob_score = np.load(c_dict['save_forest_file_ps'])
            d_train_tree = np.load(c_dict['save_forest_file_d_train_tree'])
        else:
            prob_score = d_train_tree = None
        (preddata3, common_support_list, prob_score, d_train_tree
         ) = mcf_cs.common_support(
            preddata2, tree_sample, fill_y_sample, fs_sample, var_x_type,
            v_dict, c_dict, common_support_list, prime_values_dict, prob_score,
            d_train_tree)
    else:
        preddata3 = preddata2
        prob_score, d_train_tree = None, None
    if c_dict['reduce_training'] or c_dict['reduce_largest_group_train']:
        err_txt = 'No d-specific sample reduction with continuous treatment.'
        assert ((c_dict['d_type'] == 'discrete') and
                c_dict['reduce_largest_group_train']), err_txt
        infiles = ((tree_sample, fill_y_sample, fs_sample)
                   if c_dict['fs_yes'] else (tree_sample, fill_y_sample))
        if c_dict['with_output']:
            print('=' * 80)
            for file in infiles:
                print(file)
                d_name = (v_dict['d_grid_nn_name']
                          if c_dict['d_type'] == 'continuous'
                          else v_dict['d_name'])
                mcf_gp.statistics_by_treatment(
                    file, d_name, v_dict['y_name'],
                    c_dict['d_type'] == 'continuous')
        if c_dict['reduce_training']:
            if c_dict['with_output']:
                print('Randomly deleting training data.')
            mcf_data.random_obs_reductions(
                infiles, fraction=c_dict['reduce_training_share'],
                seed=_seed_sample_split)
        if c_dict['reduce_largest_group_train']:
            assert c_dict['d_type'] == 'discrete', 'N/A for cont. treatment'
            if c_dict['with_output']:
                print('Randomly deleting training data of largest treatment.')
            mcf_data.random_obs_reductions_treatment(
                infiles, fraction=c_dict['reduce_largest_group_train_share'],
                d_name=v_dict['d_name'], seed=_seed_sample_split)
        if c_dict['with_output']:
            print('=' * 80)
            for file in infiles:
                print(file)
                mcf_gp.statistics_by_treatment(
                    file, d_name, v_dict['y_name'],
                    c_dict['d_type'] == 'continuous')
    if c_dict['reduce_prediction']:
        if c_dict['with_output']:
            print('=' * 80)
            print('Randomly deleting prediction data.')
        infiles = (preddata3,)
        outfile = (preddata3[:-4] + 'red.csv',)
        mcf_data.random_obs_reductions(
            infiles, outfiles=outfile,
            fraction=c_dict['reduce_prediction_share'],
            seed=_seed_sample_split)
        preddata3 = outfile[0]
        if c_dict['with_output']:
            print('-' * 80)
    if c_dict['iate_eff_flag']:
        est_rounds = ('regular', 'additional')
    else:
        est_rounds = ('regular', )
        pot_y_eff = iate_eff = None
    for round_ in est_rounds:
        if round_ == 'regular':
            reg_round, pot_y = True, None
            c_dict_prev = copy.deepcopy(c_dict)
            time3_2 = time.time()   # Initialisation, will be overwritten
        else:
            reg_round, c_dict = False, c_dict_prev
            # Swap samples for training and estimation
            tree_sample, fill_y_sample = fill_y_sample, tree_sample
            if c_dict['with_output'] and c_dict['verbose']:
                print('Second round of estimation to get better IATEs')
        if c_dict['train_mcf']:
            if reg_round:
                time2 = time.time()
            if c_dict['fs_yes'] and reg_round:
                # Pre-analysis feature selection
                fs_in, var_fs = mcf_data.nn_matched_outcomes(
                    fs_sample, v_dict, var_x_type, c_dict)
                # Analyse features
                if c_dict['with_output'] and c_dict['verbose']:
                    gp.statistics_covariates(fs_in, var_x_type)
                    print('\n\nFeature selection')
                    gp.print_f(ofs, 'Feature selection active.')
                cfs_dict = copy.deepcopy(c_dict)
                if isinstance(cfs_dict['grid_n_min'], (list, tuple)):
                    cfs_dict['grid_n_min'] = cfs_dict['grid_n_min'][0]
                cfs_dict['m_grid'] = 1
                if isinstance(cfs_dict['grid_alpha_reg'], (list, tuple)):
                    cfs_dict['grid_alpha_reg'] = cfs_dict['grid_alpha_reg'][0]
                fs_f, x_name_mcf = mcf_forest.build_forest(
                    fs_in, var_fs, var_x_type, var_x_values, cfs_dict)
                vi_i, vi_g, vi_ag, name = mcf_vi.variable_importance(
                    fs_in, fs_f, var_fs, var_x_type, var_x_values, cfs_dict,
                    x_name_mcf)
                (v_dict, var_x_type, var_x_values
                 ) = mcf_forest_add.fs_adjust_vars(
                    vi_i, vi_g, vi_ag, v_dict, var_x_type, var_x_values, name,
                    cfs_dict)
                del fs_f, fs_in, cfs_dict, vi_i, vi_g, vi_ag, name, var_fs
            if reg_round:
                v_dict_prev = copy.deepcopy(v_dict)
                var_x_type_prev = copy.deepcopy(var_x_type)
                var_x_values_prev = copy.deepcopy(var_x_values)
            else:
                v_dict = v_dict_prev
                var_x_type = var_x_type_prev
                var_x_values = var_x_values_prev
            # Estimate forests
            if reg_round:
                time3 = time.time()
            else:
                time3_2 = time.time()
            assert not _load_old_forest, 'Currently not active.Dics not saved.'
            if c_dict['with_output'] and c_dict['verbose']:
                print('\nMatching outcomes')
            # Match neighbours from other treatments
            indatei_tree, v_dict = mcf_data.nn_matched_outcomes(
                tree_sample, v_dict, var_x_type, c_dict)
            if c_dict['with_output'] and c_dict['desc_stat'] and reg_round:
                print('\nStatistics on matched neighbours of variable ',
                      ' used for tree building')
                d_name = (v_dict['d_grid_nn_name']
                          if c_dict['d_type'] == 'continuous'
                          else v_dict['d_name'])
                mcf_gp.statistics_by_treatment(
                    indatei_tree, d_name, v_dict['y_match_name'],
                    c_dict['d_type'] == 'continuous')
            # Estimate forest structure
            forest, x_name_mcf = mcf_forest.build_forest(
                indatei_tree, v_dict, var_x_type, var_x_values, c_dict)
            if reg_round:
                time4 = time.time()
            if c_dict['with_output'] and c_dict['verbose'] and reg_round:
                gp.print_timing(['Forst Building: '], [time4 - time3])
            # Variable importance
            if (c_dict['var_import_oob'] and c_dict['with_output']
                    and reg_round):
                mcf_vi.variable_importance(
                    indatei_tree, forest, v_dict, var_x_type, var_x_values,
                    c_dict, x_name_mcf)
            forest = mcf_forest_add.remove_oob_from_leaf0(forest)
            if reg_round:
                time5 = time.time()
            # Filling of trees with indices of outcomes:
            forest, _, _ = mcf_forest_add.fill_trees_with_y_indices_mp(
                forest, fill_y_sample, v_dict, var_x_type, var_x_values,
                c_dict, x_name_mcf)
            if reg_round:
                time6 = time.time()
            if c_dict['with_output'] and c_dict['verbose'] and reg_round:
                print()
                print('-' * 80)
                print('Size of forest: ', round(
                    gp_sys.total_size(forest) / (1024 * 1024), 2), ' MB',
                    flush=True)
                print('-' * 80)
        else:
            time11 = time12 = time2 = time3 = time4 = time5 = time.time()
            time6 = time.time()
        if c_dict['save_forest'] and c_dict['train_mcf'] and reg_round:
            save_train_data_for_pred(fill_y_sample, v_dict, c_dict, prob_score,
                                     d_train_tree)
            gp_sys.save_load(
                c_dict['save_forest_file_pickle'],
                (forest, x_name_mcf, c_dict['max_cats_z_vars'], d_in_values,
                 no_val_dict, q_inv_dict, q_inv_cr_dict, prime_values_dict,
                 unique_val_dict, common_support_list, z_new_name_dict,
                 z_new_dic_dict, c_dict['max_cats_cont_vars'], lc_forest,
                 old_y_name, new_y_name, v_dict['y_tree_name'],
                 c_dict['l_centering']), save=True,
                output=c_dict['with_output'])
        if reg_round:
            del prob_score, d_train_tree, common_support_list, unique_val_dict
            del z_new_name_dict, d_in_values, no_val_dict, q_inv_dict
            del q_inv_cr_dict, prime_values_dict, z_new_dic_dict
        if c_dict['pred_mcf']:
            if not c_dict['train_mcf'] and reg_round:
                fill_y_sample = c_dict['save_forest_file_csv']
                if c_dict['l_centering']:
                    v_dict = mcf_data.adjust_y_names(
                        v_dict, old_y_name, new_y_name, c_dict['with_output'])
            # compute weights
            if reg_round:
                time7 = time.time()
            (weights, y_train, x_bala_train, cl_train, w_train
             ) = mcf_w.get_weights_mp(
                forest, preddata3, fill_y_sample, v_dict, c_dict, x_name_mcf)
            if c_dict['with_output'] and c_dict['verbose'] and reg_round:
                print()
                print('-' * 80)
                no_of_treat = (len(c_dict['ct_grid_w_val'])
                               if c_dict['d_type'] == 'continuous'
                               else c_dict['no_of_treat'])
                mcf_gp.print_size_weight_matrix(
                    weights, c_dict['weight_as_sparse'], no_of_treat)
                print('-' * 80)
            if not (c_dict['marg_plots'] and c_dict['with_output']
                    ) or not reg_round:
                del forest

            # Estimation and inference given weights
            if reg_round:
                time8 = time.time()
            if reg_round:
                w_ate, _, _, ate, ate_se, effect_list = mcf_ate.ate_est(
                    weights, preddata3, y_train, cl_train, w_train, v_dict,
                    c_dict)
                time9_ate = time.time()
                cont = c_dict['d_type'] == 'continuous'
                if c_dict['marg_plots'] and c_dict['with_output']:
                    (mgate, mgate_se, mgate_diff, mgate_se_diff, amgate,
                     amgate_se, amgate_diff, amgate_se_diff
                     ) = mcf_gate.marg_gates_est(
                        forest, fill_y_sample, preddata3, v_dict, c_dict,
                        x_name_mcf, var_x_type, var_x_values, w_ate)
                    mcf_gate_tables.gate_tables_nice(c_dict, gate=False)
                    del forest
                time9_marg = time.time()
                if c_dict['gate_yes']:
                    gate, gate_se, gate_diff, gate_se_diff = mcf_gate.gate_est(
                        weights, preddata3, y_train, cl_train, w_train, v_dict,
                        c_dict, var_x_type, var_x_values, w_ate, ate, ate_se)
                    if c_dict['with_output']:
                        mcf_gate_tables.gate_tables_nice(c_dict, gate=True)
                else:
                    gate = gate_se = None
                time9_gate = time.time()
            if c_dict['iate_flag']:
                if not reg_round:
                    w_ate = None
                (pred_outfile, pot_y_, pot_y_var_, iate_, iate_se_,
                 names_pot_iate_, preddata_all) = mcf_iate.iate_est_mp(
                    weights, preddata3, y_train, cl_train, w_train, v_dict,
                    c_dict, w_ate, pot_y_prev=pot_y, lc_forest=lc_forest,
                    var_x_type=var_x_type)
                if reg_round:
                    pot_y, pot_y_var = pot_y_, pot_y_var_
                    iate, iate_se = iate_, iate_se_
                    names_pot_iate = names_pot_iate_
                else:
                    pot_y_eff, iate_eff = pot_y_, iate_
            else:
                pot_y = pot_y_var = iate = iate_se = None
                iate_eff = pot_y_eff = None
                pred_outfile = names_pot_iate = preddata_all = None
            del _, w_ate
            if reg_round:
                time9_iate = time.time()
            else:
                time9_iate_2 = time.time()
            if (c_dict['with_output'] and c_dict['balancing_test_w']
                    and not cont and reg_round):
                mcf_ate.ate_est(weights, preddata3, x_bala_train, cl_train,
                                w_train, v_dict, c_dict, True, None)
            if reg_round:
                del weights, y_train, x_bala_train, cl_train, w_train
                time9_bal = time.time()

            if (c_dict['with_output'] and c_dict['post_est_stats']
                    and reg_round):
                mcf_iate_add.post_estimation_iate(
                    pred_outfile, names_pot_iate, ate, ate_se, effect_list,
                    v_dict, c_dict, var_x_type)
            if reg_round:
                time10 = time.time()
        else:
            time7 = time8 = time9_ate = time9_marg = time9_gate = time.time()
            time9_iate = time9_bal = time10 = time3_2 = time.time()
            ate = ate_se = gate = gate_se = iate = iate_se = pot_y = None
            pot_y_var = pred_outfile = iate_eff = pot_y_eff = None
            preddata_all = names_pot_iate = None
    if not c_dict['iate_eff_flag']:
        time9_iate_2 = time.time()
    timetot = time9_iate_2 if c_dict['iate_eff_flag'] else time10
    # Print timing information
    time_string = ['Data preparation and stats I:    ',
                   'Local centering (recoding of Y): ',
                   'Data preparation and stats II:   ',
                   'Feature preselection:            ',
                   'Estimate forest structure:       ',
                   'Variable importance              ',
                   'Fill tree with outcomes:         ',
                   'Common support:                  ',
                   'Weight computation:              ',
                   'Inference for ATEs:              ',
                   'Inference for MGATE & AMGATE:    ',
                   'Inference for GATEs:             ',
                   'Inference for IATEs:             ',
                   'Balancing test:                  ',
                   'Post estimation analysis:        ',
                   'Second round IATE predictions:   ',
                   'Total time:                      ']
    time_difference = [time11 - time1, time12 - time11, time2 - time12,
                       time3 - time2, time4 - time3,
                       time5 - time4, time6 - time5, time7 - time6,
                       time8 - time7, time9_ate - time8,
                       time9_marg - time9_ate, time9_gate - time9_marg,
                       time9_iate - time9_gate, time9_bal - time9_iate,
                       time10 - time9_bal, time9_iate_2 - time3_2,
                       timetot - time1]
    temppfad = c_dict['outpfad'] + '/_tempmcf_'
    if os.path.isdir(temppfad):
        for temp_file in os.listdir(temppfad):
            os.remove(os.path.join(temppfad, temp_file))
        try:
            os.rmdir(temppfad)
        except OSError:
            if c_dict['with_output'] and c_dict['verbose']:
                print(f"Removal of the temorary directory {temppfad:s} failed")
        else:
            if c_dict['with_output'] and c_dict['verbose']:
                print(f"Successfully removed the directory {temppfad:s}")
    else:
        if c_dict['with_output'] and c_dict['verbose']:
            print('Temporary directory does not exist.')
    if c_dict['with_output'] and c_dict['verbose']:
        print('\n')
        if (c_dict['no_parallel'] == 1) or (c_dict['no_parallel'] == 0):
            print('No parallel processing')
        else:
            print('Multiprocessing')
            print('Number of cores used: ', c_dict['no_parallel'])
            if c_dict['_ray_or_dask'] == 'ray':
                print('Ray used for MP.')
            elif c_dict['_ray_or_dask'] == 'dask':
                print('Dask used for MP.')
            else:
                print('Concurrent futures used for MP.')
                if c_dict['mp_type_vim'] == 1:
                    print('MP for variable importance was variable based')
                else:
                    print('MP for variable importance was bootstrap based')
                if c_dict['mp_type_weights'] == 1:
                    print('MP for weights was based on groups of observations')
                else:
                    print('MP for weights was based on bootstrap for each',
                          'observation')
    if c_dict['with_output']:
        print_str = gp.print_timing(time_string, time_difference)
        print(' ', flush=True)   # clear print buffer
        gp.print_f(ofs, print_str)
    if c_dict['print_to_file']:
        if c_dict['print_to_terminal']:
            sys.stdout.output.close()
        else:
            outfiletext.close()
        sys.stdout = orig_stdout
    gate_all = (gate, mgate, mgate_diff, amgate, amgate_diff)
    gate_se_all = (gate_se, mgate_se,  mgate_se_diff, amgate_se,
                   amgate_se_diff)
    return (ate, ate_se, gate_all, gate_se_all, iate, iate_se, pot_y,
            pot_y_var, pred_outfile, pot_y_eff, iate_eff, names_pot_iate,
            preddata_all)


def save_train_data_for_pred(data_file, v_dict, c_dict, prob_score,
                             d_train_tree, regrf=None):
    """Load training data needed also for prediction parts."""
    data_train = pd.read_csv(data_file)
    names_to_save = copy.copy(v_dict['y_name'])
    if not regrf:
        names_to_save.append(*v_dict['d_name'])
    if c_dict['cluster_std']:
        names_to_save.append(*v_dict['cluster_name'])
    if c_dict['w_yes']:
        names_to_save.append(*v_dict['w_name'])
    if not regrf:
        if c_dict['balancing_test_w']:
            names_to_save += v_dict['x_balance_name']
    gp.delete_file_if_exists(c_dict['save_forest_file_csv'])
    gp.delete_file_if_exists(c_dict['save_forest_file_ps'])
    gp.delete_file_if_exists(c_dict['save_forest_file_d_train_tree'])
    data_train[names_to_save].to_csv(c_dict['save_forest_file_csv'],
                                     index=False)
    np.save(c_dict['save_forest_file_ps'], prob_score)
    np.save(c_dict['save_forest_file_d_train_tree'], d_train_tree)


class ModifiedCausalForest:
    """Estimate mcf in OOP style."""

    def __init__(
            self, id_name=None, cluster_name=None, w_name=None, d_name=None,
            d_type='discrete', y_tree_name=None, y_name=None, x_name_ord=None,
            x_name_unord=None, x_name_always_in_ord=None, z_name_list=None,
            x_name_always_in_unord=None, z_name_split_ord=None,
            z_name_split_unord=None, z_name_mgate=None, z_name_amgate=None,
            x_name_remain_ord=None, x_name_remain_unord=None,
            x_balance_name_ord=None, x_balance_name_unord=None,
            alpha_reg_min=0.1, alpha_reg_max=0.2, alpha_reg_grid=2,
            atet_flag=False, balancing_test=True,
            boot=1000, ci_level=0.9, check_perfectcorr=True,
            choice_based_sampling=False, choice_based_weights=None,
            clean_data_flag=True, cluster_std=False, cond_var_flag=True,
            ct_grid_nn=10, ct_grid_w=10, ct_grid_dr=100,
            datpfad=None, fs_other_sample_share=0.2,
            fs_other_sample=True, fs_rf_threshold=0, fs_yes=None,
            forest_files=None,
            gates_minus_previous=False,
            gates_smooth=True, gates_smooth_bandwidth=1,
            gates_smooth_no_evaluation_points=50, gatet_flag=False,
            gmate_no_evaluation_points=50, gmate_sample_share=None,
            iate_flag=True, iate_se_flag=True, iate_eff_flag=True,
            iate_cv_flag=False, iate_cv_folds=5,
            indata=None, knn_flag=True, knn_min_k=10, knn_const=1,
            l_centering=True, l_centering_share=0.25, l_centering_cv_k=5,
            l_centering_new_sample=False, l_centering_replication=True,
            l_centering_undo_iate=True, m_min_share=-1, m_max_share=-1,
            m_grid=2, m_random_poisson=True, match_nn_prog_score=True,
            max_cats_z_vars=None, max_weight_share=0.05,
            mce_vart=1, min_dummy_obs=10,  mp_parallel=None,
            n_min_grid=1, n_min_min=-1, n_min_max=-1, n_min_treat=3,
            nn_main_diag_only=False, nw_kern_flag=1, nw_bandw=None,
            outfiletext=None, outpfad=None, output_type=2, panel_data=False,
            panel_in_rf=True, preddata=None, p_diff_penalty=None,
            post_bin_corr_threshold=0.1, post_bin_corr_yes=True,
            post_est_stats=True,
            post_plots=True, post_kmeans_yes=True, post_kmeans_max_tries=1000,
            post_kmeans_no_of_groups=None, post_kmeans_replications=10,
            post_random_forest_vi=True, post_relative_to_first_group_only=True,
            predict_mcf=True,
            reduce_prediction=False, reduce_prediction_share=0.5,
            reduce_split_sample=False, reduce_split_sample_pred_share=0.5,
            reduce_training=False, reduce_training_share=0.5,
            reduce_largest_group_train=False,
            reduce_largest_group_train_share=0.5, save_forest=False,
            screen_covariates=True, se_boot_ate=False, se_boot_gate=False,
            se_boot_iate=False, random_thresholds=None,
            subsample_factor_forest=None, subsample_factor_eval=False,
            support_check=1, support_min_p=None, support_quantil=1,
            support_max_del_train=0.5, support_adjust_limits=None,
            train_mcf=True, variable_importance_oob=False, weighted=False,
            _share_forest_sample=0.5, _weight_as_sparse=True,
            _mp_vim_type=None, _mp_weights_type=1, _mp_weights_tree_batch=None,
            _verbose=True, _fontsize=2, _descriptive_stats=True,
            _no_filled_plot=20, _show_plots=True,
            _smaller_sample=0, _with_output=True, _dpi=500,
            _max_cats_cont_vars=None, _max_save_values=50,
            _seed_sample_split=67567885, _mp_ray_del=None,
            _mp_ray_shutdown=None, _mp_ray_objstore_multiplier=None,
            _ray_or_dask='ray', _return_iate_sp=False):
        self.train_mcf = train_mcf
        self.predict_mcf = predict_mcf
        self.params = {
            'id_name': id_name,
            'cluster_name': cluster_name,
            'w_name': w_name,
            'd_name': d_name,
            'd_type': d_type,
            'y_tree_name': y_tree_name,
            'y_name': y_name,
            'x_name_ord': x_name_ord,
            'x_name_unord': x_name_unord,
            'x_name_always_in_ord': x_name_always_in_ord,
            'z_name_list': z_name_list,
            'x_name_always_in_unord': x_name_always_in_unord,
            'z_name_split_ord': z_name_split_ord,
            'z_name_split_unord': z_name_split_unord,
            'z_name_mgate': z_name_mgate,
            'z_name_amgate': z_name_amgate,
            'x_name_remain_ord': x_name_remain_ord,
            'x_name_remain_unord': x_name_remain_unord,
            'x_balance_name_ord': x_balance_name_ord,
            'x_balance_name_unord': x_balance_name_unord,
            'alpha_reg_min': alpha_reg_min,
            'alpha_reg_max': alpha_reg_max,
            'alpha_reg_grid': alpha_reg_grid,
            'atet_flag': atet_flag,
            'balancing_test': balancing_test,
            'post_bin_corr_threshold': post_bin_corr_threshold,
            'post_bin_corr_yes': post_bin_corr_yes,
            'boot': boot,
            'ci_level': ci_level,
            'check_perfectcorr': check_perfectcorr,
            'choice_based_sampling': choice_based_sampling,
            'choice_based_weights': choice_based_weights,
            'clean_data_flag': clean_data_flag,
            'cluster_std': cluster_std,
            'cond_var_flag': cond_var_flag,
            'ct_grid_nn': ct_grid_nn,
            'ct_grid_w': ct_grid_w,
            'ct_grid_dr': ct_grid_dr,
            'datpfad': datpfad,
            'fs_other_sample_share': fs_other_sample_share,
            'fs_other_sample': fs_other_sample,
            'fs_rf_threshold': fs_rf_threshold,
            'fs_yes': fs_yes,
            'forest_files': forest_files,
            'gates_minus_previous': gates_minus_previous,
            'gatet_flag': gatet_flag,
            'gmate_no_evaluation_points': gmate_no_evaluation_points,
            'gmate_sample_share': gmate_sample_share,
            'iate_flag': iate_flag,
            'iate_se_flag': iate_se_flag,
            'iate_eff_flag': iate_eff_flag,
            'iate_cv_flag': iate_cv_flag,
            'iate_cv_folds': iate_cv_folds,
            'indata': indata,
            'knn_flag': knn_flag,
            'knn_min_k': knn_min_k,
            'knn_const': knn_const,
            'l_centering': l_centering,
            'l_centering_share': l_centering_share,
            'l_centering_cv_k': l_centering_cv_k,
            'l_centering_new_sample': l_centering_new_sample,
            'l_centering_replication': l_centering_replication,
            'l_centering_undo_iate': l_centering_undo_iate,
            'm_min_share': m_min_share,
            'm_max_share': m_max_share,
            'm_grid': m_grid,
            'm_random_poisson': m_random_poisson,
            'match_nn_prog_score': match_nn_prog_score,
            'max_cats_z_vars': max_cats_z_vars,
            'max_weight_share': max_weight_share,
            'mce_vart': mce_vart,
            'min_dummy_obs': min_dummy_obs,
            'mp_parallel': mp_parallel,
            'n_min_grid': n_min_grid,
            'n_min_min': n_min_min,
            'n_min_max': n_min_max,
            'n_min_treat': n_min_treat,
            'nn_main_diag_only': nn_main_diag_only,
            'nw_kern_flag': nw_kern_flag,
            'nw_bandw': nw_bandw,
            'outfiletext': outfiletext,
            'outpfad': outpfad,
            'output_type': output_type,
            'panel_data': panel_data,
            'panel_in_rf': panel_in_rf,
            'preddata': preddata,
            'p_diff_penalty': p_diff_penalty,
            'post_est_stats': post_est_stats,
            'post_plots': post_plots,
            'post_kmeans_yes': post_kmeans_yes,
            'post_kmeans_max_tries': post_kmeans_max_tries,
            'post_kmeans_no_of_groups': post_kmeans_no_of_groups,
            'post_kmeans_replications': post_kmeans_replications,
            'post_random_forest_vi': post_random_forest_vi,
            'predict_mcf': predict_mcf,
            'reduce_prediction': reduce_prediction,
            'reduce_prediction_share': reduce_prediction_share,
            'reduce_split_sample': reduce_split_sample,
            'reduce_split_sample_pred_share': reduce_split_sample_pred_share,
            'reduce_training': reduce_training,
            'reduce_training_share': reduce_training_share,
            'post_relative_to_first_group_only':
                post_relative_to_first_group_only,
            'reduce_largest_group_train': reduce_largest_group_train,
            'reduce_largest_group_train_share':
                reduce_largest_group_train_share,
            'save_forest': save_forest,
            'screen_covariates': screen_covariates,
            'se_boot_ate': se_boot_ate,
            'se_boot_gate': se_boot_gate,
            'se_boot_iate': se_boot_iate,
            'gates_smooth': gates_smooth,
            'gates_smooth_bandwidth': gates_smooth_bandwidth,
            'gates_smooth_no_evaluation_points':
                gates_smooth_no_evaluation_points,
            'random_thresholds': random_thresholds,
            'subsample_factor_forest': subsample_factor_forest,
            'subsample_factor_eval': subsample_factor_eval,
            'support_check': support_check,
            'support_min_p': support_min_p,
            'support_quantil': support_quantil,
            'support_max_del_train': support_max_del_train,
            'support_adjust_limits': support_adjust_limits,
            'train_mcf': train_mcf,
            'variable_importance_oob': variable_importance_oob,
            'weighted': weighted,
            '_share_forest_sample': _share_forest_sample,
            '_weight_as_sparse': _weight_as_sparse,
            '_mp_vim_type': _mp_vim_type,
            '_mp_weights_type': _mp_weights_type,
            '_mp_weights_tree_batch': _mp_weights_tree_batch,
            '_verbose': _verbose,
            '_fontsize': _fontsize,
            '_descriptive_stats': _descriptive_stats,
            '_no_filled_plot': _no_filled_plot,
            '_show_plots': _show_plots,
            '_smaller_sample': _smaller_sample,
            '_with_output': _with_output,
            '_dpi': _dpi,
            '_max_cats_cont_vars': _max_cats_cont_vars,
            '_max_save_values': _max_save_values,
            '_seed_sample_split': _seed_sample_split,
            '_mp_ray_del': _mp_ray_del,
            '_mp_ray_shutdown': _mp_ray_shutdown,
            '_mp_ray_objstore_multiplier': _mp_ray_objstore_multiplier,
            '_ray_or_dask': _ray_or_dask}

    def train(self):
        """Train the mcf without prediction."""
        self.params['train_mcf'] = True
        self.params['predict_mcf'] = False
        modified_causal_forest(**self.params)

    def predict(self):
        """Predict from an already trained mcf."""
        self.params['train_mcf'] = False
        self.params['predict_mcf'] = True
        (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
         pred_outfile, pot_y_eff, iate_eff, names_pot_iate, preddata_all,
         iate_cv_file, iate_cv_names) = modified_causal_forest(**self.params)
        return (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
                pred_outfile, pot_y_eff, iate_eff, names_pot_iate,
                preddata_all, iate_cv_file, iate_cv_names)

    def train_predict(self):
        """Train and predict mcf in one go (recommended)."""
        self.params['train_mcf'] = True
        self.params['predict_mcf'] = True
        (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
         pred_outfile, pot_y_eff, iate_eff, names_pot_iate, preddata_all,
         iate_cv_file, iate_cv_names) = modified_causal_forest(**self.params)
        return (ate, ate_se, gate, gate_se, iate, iate_se, pot_y, pot_y_var,
                pred_outfile, pot_y_eff, iate_eff, names_pot_iate,
                preddata_all, iate_cv_file, iate_cv_names)
