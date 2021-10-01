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
from mcf import general_purpose_mcf as gp_mcf
from mcf import mcf_init_functions as mcf_init
from mcf import mcf_data_functions as mcf_data
from mcf import mcf_loccent_functions as mcf_lc
from mcf import mcf_forest_functions as mcf_forest
from mcf import mcf_vi_functions as mcf_vi
from mcf import mcf_cs_functions as mcf_cs
from mcf import mcf_weight_functions as mcf_w
from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_gate_functions as mcf_gate
from mcf import mcf_iate_functions as mcf_iate


def ModifiedCausalForest(
        id_name=None, cluster_name=None, w_name=None, d_name=None,
        y_tree_name=None, y_name=None, x_name_ord=None, x_name_unord=None,
        x_name_always_in_ord=None, z_name_list=None,
        x_name_always_in_unord=None, z_name_split_ord=None,
        z_name_split_unord=None, z_name_mgate=None, z_name_amgate=None,
        x_name_remain_ord=None, x_name_remain_unord=None,
        x_balance_name_ord=None, x_balance_name_unord=None,
        alpha_reg_min=0.1, alpha_reg_max=0.2, alpha_reg_grid=2,
        atet_flag=False, balancing_test=True, bin_corr_threshold=0.1,
        bin_corr_yes=True, boot=1000, ci_level=0.9, check_perfectcorr=True,
        choice_based_sampling=False, choice_based_weights=None,
        clean_data_flag=True, cluster_std=False, cond_var_flag=True,
        datpfad=None, descriptive_stats=True, dpi=500,
        fs_other_sample_share=0.2, fs_other_sample=True, fs_rf_threshold=0,
        fs_yes=None, fontsize=2, forest_files=None, gatet_flag=False,
        gmate_no_evaluation_points=50, gmate_sample_share=None, indata=None,
        knn_flag=False, knn_min_k=10, knn_const=1, l_centering=False,
        l_centering_share=0.25, l_centering_cv_k=5,
        l_centering_new_sample=False, m_min_share=-1, m_max_share=-1, m_grid=2,
        m_random_poisson=True, match_nn_prog_score=True, max_cats_z_vars=None,
        max_weight_share=0.05, mce_vart=1, min_dummy_obs=10,  mp_parallel=None,
        mp_ray_objstore_multiplier=None, mp_with_ray=True, mp_vim_type=None,
        mp_weights_tree_batch=None,  mp_weights_type=1, n_min_grid=1,
        n_min_min=-1, n_min_max=-1, nn_main_diag_only=False, no_filled_plot=20,
        nw_kern_flag=1, nw_bandw=None, outfiletext=None, outpfad=None,
        output_type=2, panel_data=False, panel_in_rf=True, preddata=None,
        p_diff_penalty=None, post_est_stats=True, post_plots=True,
        post_kmeans_yes=True, post_kmeans_max_tries=1000,
        post_kmeans_no_of_groups=None, post_kmeans_replications=10,
        post_random_forest_vi=True, predict_mcf=True,
        random_thresholds=None, relative_to_first_group_only=True,
        save_forest=False, screen_covariates=True, share_forest_sample=0.5,
        show_plots=True, smooth_gates=True, smooth_gates_bandwidth=1,
        smooth_gates_no_evaluation_points=50, stop_empty=25,
        subsample_factor_forest=None, subsample_factor_eval=None,
        support_check=1, support_min_p=None,
        support_quantil=1, support_max_del_train=0.5,
        train_mcf=True, variable_importance_oob=False,
        verbose=True, weight_as_sparse=True, weighted=False,
        se_boot_ate=False, se_boot_gate=False, se_boot_iate=False,
        _smaller_sample=0, _with_output=True, _max_cats_cont_vars=None,
        _max_save_values=50, _seed_sample_split=67567885):
    """Compute the honest causal/random forest (based on mcf)."""
    freeze_support()
    time1 = time.time()

    # Some temporary operational flags may be changed at some point
    _boot_by_boot = 10    # Build forest x-by-x or in larger groups (not Ray)
    _obs_by_obs = False      # Compute IATE obs by obs or in chuncks
    _max_elements_per_split = 100 * 10e5  # reduce if breakdown (weights)
    _load_old_forest = False  # Useful for testing to save time
    _no_ray_in_forest_building = False
    # Problem in Ray 1.2.0 when large forests are build; check if still there
    # Ray 2.0.0 once available - concurrent futures even more problematic!

# Collect vars in a dictionary
    variable_dict = mcf_init.make_user_variable(
        id_name, cluster_name, w_name, d_name, y_tree_name, y_name, x_name_ord,
        x_name_unord, x_name_always_in_ord, z_name_list,
        x_name_always_in_unord, z_name_split_ord, z_name_split_unord,
        z_name_mgate, z_name_amgate, x_name_remain_ord, x_name_remain_unord,
        x_balance_name_ord, x_balance_name_unord)

# use smaller random sample (usually for testing purposes)
    if 0 < _smaller_sample < 1:
        if train_mcf:
            gp.randomsample(datpfad, indata + '.csv', 'smaller_indata.csv',
                            _smaller_sample, True)
        if preddata is None:
            preddata = 'smaller_indata'
        if predict_mcf and preddata != indata:
            gp.randomsample(datpfad, preddata + '.csv', 'smaller_preddata.csv',
                            _smaller_sample, True)
            preddata = 'smaller_preddata'
        else:
            preddata = 'smaller_indata'
        indata = 'smaller_indata'
# set values for control variables
    controls_dict = mcf_init.controls_into_dic(
        mp_parallel, mp_vim_type, output_type, outpfad,
        datpfad, indata, preddata, outfiletext, screen_covariates,
        n_min_grid, check_perfectcorr, n_min_min, clean_data_flag,
        min_dummy_obs, mce_vart, p_diff_penalty, boot, n_min_max,
        support_min_p, weighted, support_check, support_quantil,
        subsample_factor_forest, subsample_factor_eval, m_min_share, m_grid,
        stop_empty, m_random_poisson, alpha_reg_min, alpha_reg_max,
        alpha_reg_grid, random_thresholds, knn_min_k, share_forest_sample,
        descriptive_stats, m_max_share, max_cats_z_vars,
        variable_importance_oob,
        balancing_test, choice_based_sampling, knn_const, choice_based_weights,
        nw_kern_flag, post_kmeans_max_tries, cond_var_flag, knn_flag, nw_bandw,
        panel_data, _max_cats_cont_vars, cluster_std, fs_yes,
        fs_other_sample_share, gatet_flag, fs_other_sample, bin_corr_yes,
        panel_in_rf, fs_rf_threshold, post_plots, post_est_stats,
        relative_to_first_group_only, post_kmeans_yes, atet_flag,
        bin_corr_threshold, post_kmeans_no_of_groups, post_kmeans_replications,
        _with_output, _max_save_values, nn_main_diag_only, fontsize, dpi,
        ci_level, max_weight_share, save_forest, l_centering,
        l_centering_share, l_centering_new_sample, l_centering_cv_k,
        post_random_forest_vi, gmate_no_evaluation_points,
        gmate_sample_share, no_filled_plot, smooth_gates,
        smooth_gates_bandwidth, smooth_gates_no_evaluation_points, show_plots,
        weight_as_sparse, mp_weights_type, mp_weights_tree_batch,
        _boot_by_boot, _obs_by_obs, _max_elements_per_split, mp_with_ray,
        mp_ray_objstore_multiplier, verbose, _no_ray_in_forest_building,
        predict_mcf, train_mcf, forest_files, match_nn_prog_score,
        se_boot_ate, se_boot_gate, se_boot_iate, support_max_del_train)
# Set defaults for many control variables of the MCF & define variables

    c_dict, v_dict, text_to_print = mcf_init.get_controls(controls_dict,
                                                          variable_dict)
# Some descriptive stats of input and direction of output file
    if c_dict['with_output']:
        if c_dict['print_to_file']:
            orig_stdout = sys.stdout
            gp.delete_file_if_exists(c_dict['outfiletext'])
            if c_dict['print_to_terminal']:
                sys.stdout = gp.OutputTerminalFile(c_dict['outfiletext'])
            else:
                outfiletext = open(c_dict['outfiletext'], 'w')
                sys.stdout = outfiletext
    if c_dict['with_output'] and c_dict['verbose']:
        if c_dict['train_mcf']:
            if text_to_print is not None:
                print(text_to_print, '{:<5} '.format(c_dict['n_min_min']))
            print('\nParameter for MCF:')
            gp.print_dic(c_dict)
            print('\nVariables used:')
            gp.print_dic(v_dict)
            print('\nValue of penalty multiplier:',
                  '{:8.3f}'.format(c_dict['mtot_p_diff_penalty']))
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
        forest = loaded_tuple[0]
        x_name_mcf = loaded_tuple[1]
        c_dict['max_cats_z_vars'] = loaded_tuple[2]
        d_in_values = loaded_tuple[3]
        no_val_dict = loaded_tuple[4]
        q_inv_dict = loaded_tuple[5]
        q_inv_cr_dict = loaded_tuple[6]
        prime_values_dict = loaded_tuple[7]
        unique_val_dict = loaded_tuple[8]
        common_support_list = loaded_tuple[9]
        z_new_name_dict = loaded_tuple[10]
        z_new_dic_dict = loaded_tuple[11]
        c_dict['max_cats_cont_vars'] = loaded_tuple[12]
    else:
        d_in_values = None
        no_val_dict = None
        q_inv_dict = None
        q_inv_cr_dict = None
        prime_values_dict = None
        unique_val_dict = None
        common_support_list = None
        z_new_name_dict = None
        z_new_dic_dict = None
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
        print()
        print('Dictionary for recoding of categorical variables into primes')
        # key_list_unique_val_dict = [key for key in unique_val_dict]
        # key_list_prime_values_dict = [key for key in prime_values_dict]
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
            indata2 = gp.clean_reduce_data(
                indata_with_z, c_dict['indata2_temp'],
                namen1_to_inc, c_dict['with_output'],
                c_dict['desc_stat'], c_dict['print_to_file'])
        if indata_with_z != predata_with_z and c_dict['pred_mcf']:
            # namen2_to_inc = gp.add_var_names(
            #     v_dict['id_name'], v_dict['cluster_name'], v_dict['w_name'],
            #     v_dict['x_balance_name'], v_dict['x_name'])
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
        indata2 = indata_with_z
        preddata2 = predata_with_z
# get rid of variables that do not have enough independent variation
    if c_dict['train_mcf']:
        if c_dict['screen_covariates']:
            _, x_variables_out = gp.screen_variables(
                indata2, v_dict['x_name'], c_dict['check_perfectcorr'],
                c_dict['min_dummy_obs'], c_dict['with_output'])
            if x_variables_out != []:
                v_dict, var_x_type, var_x_values, c_dict = (
                    mcf_data.adjust_variables(v_dict, var_x_type, var_x_values,
                                              c_dict, x_variables_out))
        time11 = time.time()
        if c_dict['l_centering'] and c_dict['l_centering_new_sample']:
            l_cent_sample, nonlc_sample, _ = gp.sample_split_2_3(
                indata2, c_dict['lc_sample'], c_dict['l_centering_share'],
                c_dict['nonlc_sample'], 1-c_dict['l_centering_share'],
                random_seed=_seed_sample_split,
                with_output=c_dict['with_output'])
            (indata2, old_y_name, new_y_name
             ) = mcf_lc.local_centering_new_sample(
                l_cent_sample, nonlc_sample, v_dict, var_x_type, c_dict)
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
            time11 = time.time()
            if c_dict['fs_yes']:
                files_to_center = (tree_sample, fill_y_sample, fs_sample)
            else:
                files_to_center = (tree_sample, fill_y_sample)
            old_y_name, new_y_name = mcf_lc.local_centering_cv(
                files_to_center, v_dict, var_x_type, c_dict)
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
            if c_dict['l_centering'] and not c_dict['l_centering_new_sample']:
                gp_mcf.statistics_by_treatment(tree_sample, v_dict['d_name'],
                                               variables_to_desc)
            else:
                gp_mcf.statistics_by_treatment(indata2, v_dict['d_name'],
                                               variables_to_desc)
            mcf_data.variable_features(var_x_type, var_x_values)
# Common support
    if c_dict['common_support'] > 0:
        if not c_dict['train_mcf']:
            tree_sample, fill_y_sample, fs_sample = None, None, None
            prob_score = np.load(c_dict['save_forest_file_ps'])
            d_train_tree = np.load(c_dict['save_forest_file_d_train_tree'])
        else:
            prob_score, d_train_tree = None, None
        (preddata3, common_support_list, prob_score, d_train_tree
         ) = mcf_cs.common_support(
            preddata2, tree_sample, fill_y_sample, fs_sample, var_x_type,
            v_dict, c_dict, common_support_list, prime_values_dict, prob_score,
            d_train_tree)
    else:
        preddata3 = preddata2
        prob_score, d_train_tree = None, None

# Pre-analysis feature selection
    if c_dict['train_mcf']:
        time2 = time.time()
        if c_dict['fs_yes']:
            fs_in, var_fs = mcf_data.nn_matched_outcomes(fs_sample, v_dict,
                                                         var_x_type, c_dict)
            # Analyse features
            if c_dict['with_output'] and c_dict['verbose']:
                gp.statistics_covariates(fs_in, var_x_type)
                print('\n\nFeature selection')
            cfs_dict = copy.deepcopy(c_dict)
            if not isinstance(cfs_dict['grid_n_min'], int):
                cfs_dict['grid_n_min'] = cfs_dict['grid_n_min'][0]
            cfs_dict['m_grid'] = 1
            # cfs_dict['alpha_reg_grid'] = 1
            if not isinstance(cfs_dict['grid_alpha_reg'], int):
                cfs_dict['grid_alpha_reg'] = cfs_dict['grid_alpha_reg'][0]
            fs_f, x_name_mcf = mcf_forest.build_forest(
                fs_in, var_fs, var_x_type, var_x_values, cfs_dict)
            vi_i, vi_g, vi_ag, name = mcf_vi.variable_importance(
                fs_in, fs_f, var_fs, var_x_type, var_x_values, cfs_dict,
                x_name_mcf)
            v_dict, var_x_type, var_x_values = mcf_forest.fs_adjust_vars(
                vi_i, vi_g, vi_ag, v_dict, var_x_type, var_x_values, name,
                cfs_dict)
            del fs_f, fs_in, cfs_dict, vi_i, vi_g, vi_ag, name, var_fs

    # Estimate forests
        time3 = time.time()
        if not _load_old_forest:
            if c_dict['with_output'] and c_dict['verbose']:
                print('\nMatching outcomes')
            # Match neighbours from other treatments
            indatei_tree, v_dict = mcf_data.nn_matched_outcomes(
                tree_sample, v_dict, var_x_type, c_dict)
            if c_dict['with_output'] and c_dict['desc_stat']:
                print('\nStatistics on matched neighbours of variable used',
                      '  for tree building')
                gp_mcf.statistics_by_treatment(indatei_tree, v_dict['d_name'],
                                               v_dict['y_match_name'])
            if c_dict['with_output'] and c_dict['desc_stat']:
                mcf_forest.structure_of_node_tabl()
    # Estimate forest structure
            forest, x_name_mcf = mcf_forest.build_forest(
                indatei_tree, v_dict, var_x_type, var_x_values, c_dict)
            time4 = time.time()
            if c_dict['with_output'] and c_dict['verbose']:
                gp.print_timing(['Forst Building: '], [time4 - time3])
    # Variable importance
            if c_dict['var_import_oob'] and c_dict['with_output']:
                mcf_vi.variable_importance(
                    indatei_tree, forest, v_dict, var_x_type, var_x_values,
                    c_dict, x_name_mcf)
            forest = mcf_forest.remove_oob_from_leaf0(forest)  # Forest is list
            time5 = time.time()

    # Filling of trees with indices of outcomes:
            forest, _, _ = mcf_forest.fill_trees_with_y_indices_mp(
                forest, fill_y_sample, v_dict, var_x_type, var_x_values,
                c_dict, x_name_mcf)
        else:
            raise Exception('Currently deactivated. Dics not saved.')

        time6 = time.time()    # Forest is tuple
        if c_dict['with_output'] and c_dict['verbose']:
            print()
            print('-' * 80)
            print('Size of forest: ', round(
                gp_sys.total_size(forest) / (1024 * 1024), 2), ' MB',
                flush=True)
            print('-' * 80)
    else:
        time11 = time.time()
        time12 = time.time()
        time2 = time.time()
        time3 = time.time()
        time4 = time.time()
        time5 = time.time()
        time6 = time.time()

    if c_dict['save_forest'] and c_dict['train_mcf']:
        save_train_data_for_pred(fill_y_sample, v_dict, c_dict, prob_score,
                                 d_train_tree)
        gp_sys.save_load(
            c_dict['save_forest_file_pickle'],
            (forest, x_name_mcf, c_dict['max_cats_z_vars'], d_in_values,
             no_val_dict, q_inv_dict, q_inv_cr_dict, prime_values_dict,
             unique_val_dict, common_support_list, z_new_name_dict,
             z_new_dic_dict, c_dict['max_cats_cont_vars']),
            save=True, output=c_dict['with_output'])
    del prob_score, d_train_tree, common_support_list, unique_val_dict
    del z_new_name_dict, d_in_values, no_val_dict, q_inv_dict, q_inv_cr_dict
    del prime_values_dict, z_new_dic_dict
    if c_dict['pred_mcf']:
        if not c_dict['train_mcf']:
            fill_y_sample = c_dict['save_forest_file_csv']
        # compute weights
        time7 = time.time()
        (weights, y_train, x_bala_train, cl_train, w_train
         ) = mcf_w.get_weights_mp(
            forest, preddata3, fill_y_sample, v_dict, c_dict, x_name_mcf)
        if c_dict['with_output'] and c_dict['verbose']:
            print()
            print('-' * 80)
            gp_mcf.print_size_weight_matrix(
                weights, c_dict['weight_as_sparse'], c_dict['no_of_treat'])
            print('-' * 80)
        if not (c_dict['marg_plots'] and c_dict['with_output']):
            del forest
        # Estimation and inference given weights
        time8 = time.time()

        w_ate, _, _, ate, ate_se, effect_list = mcf_ate.ate_est(
            weights, preddata3, y_train, cl_train, w_train, v_dict, c_dict)
        time9_ate = time.time()
        if c_dict['marg_plots'] and c_dict['with_output']:
            mcf_gate.marg_gates_est(
                forest, fill_y_sample, preddata3, v_dict, c_dict, x_name_mcf,
                var_x_type, var_x_values, w_ate)
            del forest
        time9_marg = time.time()
        # if c_dict['with_output']:
        if c_dict['gate_yes']:
            gate, gate_se = mcf_gate.gate_est(
                weights, preddata3, y_train, cl_train, w_train, v_dict, c_dict,
                var_x_type, var_x_values, w_ate, ate, ate_se)
        else:
            gate = gate_se = None
        time9_gate = time.time()
        (pred_outfile, _, _, iate, iate_se, names_pot_iate
         ) = mcf_iate.iate_est_mp(
            weights, preddata3, y_train, cl_train, w_train, v_dict, c_dict,
            w_ate)
        del _, w_ate
        time9_iate = time.time()

        if c_dict['with_output'] and c_dict['balancing_test_w']:
            mcf_ate.ate_est(weights, preddata3, x_bala_train, cl_train,
                            w_train, v_dict, c_dict, True)
        del weights, y_train, x_bala_train, cl_train, w_train
        time9_bal = time.time()

        if c_dict['with_output'] and c_dict['post_est_stats']:
            mcf_iate.post_estimation_iate(
                pred_outfile, names_pot_iate, ate, ate_se, effect_list, v_dict,
                c_dict, var_x_type)
        time10 = time.time()
    else:
        time7 = time.time()
        time8 = time.time()
        time9_ate = time.time()
        time9_marg = time.time()
        time9_gate = time.time()
        time9_iate = time.time()
        time9_bal = time.time()
        time10 = time.time()
# Finally, save everything: Con, Var (auch wegen recoding), forests etc.

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
                   'Total time:                      ']
    time_difference = [time11 - time1, time12 - time11, time2 - time12,
                       time3 - time2, time4 - time3,
                       time5 - time4, time6 - time5, time7 - time6,
                       time8 - time7, time9_ate - time8,
                       time9_marg - time9_ate, time9_gate - time9_marg,
                       time9_iate - time9_gate, time9_bal - time9_iate,
                       time10 - time9_bal, time10 - time1]
    temppfad = c_dict['outpfad'] + '/_tempmcf_'
    if os.path.isdir(temppfad):
        for temp_file in os.listdir(temppfad):
            os.remove(os.path.join(temppfad, temp_file))
        try:
            os.rmdir(temppfad)
        except OSError:
            if c_dict['with_output'] and c_dict['verbose']:
                print("Removal of the temorary directory %s failed" % temppfad)
        else:
            if c_dict['with_output'] and c_dict['verbose']:
                print("Successfully removed the directory %s" % temppfad)
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
            if c_dict['mp_with_ray']:
                print('Ray used for MP.')
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
        gp.print_timing(time_string, time_difference)  # print to file
    if c_dict['print_to_file']:
        if c_dict['print_to_terminal']:
            sys.stdout.output.close()
        else:
            outfiletext.close()
        sys.stdout = orig_stdout
    return ate, ate_se, gate, gate_se, iate, iate_se, pred_outfile


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
