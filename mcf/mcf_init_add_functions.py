"""Created on Thu Mar 17 08:17:06 2022.

Contains the functions needed for the running all parts of the programme part 2
@author: MLechner
-*- coding: utf-8 -*-
"""
import os

import numpy as np


def interpol_weights(ct_grid_dr, ct_grid_w, ct_grid_w_val, precision_of_treat):
    """Generate interpolation measures for continuous treatments."""
    interpol_points = round(ct_grid_dr / ct_grid_w) + 1
    int_w01 = np.linspace(0, 1, interpol_points, endpoint=False)
    int_w10 = 1 - int_w01
    treat_val_list, j_all = [], 0
    index_full = np.zeros((ct_grid_w, len(int_w01)))
    for i, (val, val1) in enumerate(zip(ct_grid_w_val[:-1],
                                        ct_grid_w_val[1:])):
        for j in range(interpol_points):
            value = int_w10[j] * val + int_w01[j] * val1
            treat_val_list.append(round(value, precision_of_treat))
            index_full[i, j] = j_all
            j_all = j_all + 1    # do not use +=
    treat_val_list.append(ct_grid_w_val[-1])
    treat_val_np = np.around(np.array(treat_val_list), precision_of_treat)
    astr = 'Continuous treatment needs higher precision'
    assert len(treat_val_np) == len(np.unique(treat_val_np)), astr
    index_full[ct_grid_w-1, 0] = j_all
    index_full = np.int32(index_full)
    return int_w01, int_w10, index_full, treat_val_list, treat_val_np


def to_list_if_needed(string_or_list):
    """Help for initialisation."""
    if isinstance(string_or_list, (tuple, set)):
        return list(string_or_list)
    if isinstance(string_or_list, str):
        return [string_or_list]
    return string_or_list


def sub_size(n_train, subsample_share_mult, max_share):
    """Help for initialisation."""
    if subsample_share_mult is None:
        subsample_share_mult = -1
    if subsample_share_mult <= 0:
        subsample_share_mult = 1
    subsam_share = 4 * ((n_train / 2)**0.85) / n_train
    subsam_share = min(subsam_share, 0.67)
    subsam_share = subsam_share * subsample_share_mult
    subsam_share = min(subsam_share, max_share)
    subsam_share = max(subsam_share, 1e-4)
    return subsam_share


def ct_grid(user_grid, defaultgrid):
    """Help for initialisation."""
    if isinstance(user_grid, int):
        grid = defaultgrid if user_grid < 1 else user_grid
    else:
        grid = defaultgrid
    return grid


def grid_val(grid, d_dat):
    """Help for initialisation."""
    quantile = np.linspace(1/(grid)/2, 1-1/grid/2, num=grid)
    d_dat_min = d_dat.min()
    d_dat_r = d_dat - d_dat_min if d_dat_min != 0 else d_dat
    gridvalues = np.around(
        np.quantile(d_dat_r[d_dat_r > 1e-15], quantile), decimals=6)
    gridvalues = np.insert(gridvalues, 0, 0)
    return gridvalues


def grid_name(d_name, add_name):
    """Help for initialisation."""
    grid_name_tmp = d_name[0] + add_name
    grid_name_l = [grid_name_tmp.upper()]
    return grid_name_l


def controls_into_dic(
        mp_parallel, mp_type_vim, output_type, outpfad, datpfad, indata,
        preddata, outfiletext, screen_covariates, n_min_grid,
        check_perfectcorr, n_min_min, clean_data_flag,
        min_dummy_obs, mce_vart, p_diff_penalty, boot, n_min_max,
        support_min_p, weighted, support_check, support_quantil,
        subsample_factor_forest, subsample_factor_eval,
        m_min_share, m_grid, m_random_poisson, alpha_reg_min,
        alpha_reg_max, alpha_reg_grid,
        random_thresholds, knn_min_k, share_forest_sample, descriptive_stats,
        m_max_share, max_cats_z_vars, variable_importance_oob,
        balancing_test, choice_based_sampling, knn_const, choice_based_weights,
        nw_kern_flag, post_kmeans_max_tries, cond_var_flag, knn_flag, nw_bandw,
        panel_data, max_cats_cont_vars, cluster_std, fs_yes,
        fs_other_sample_share, gatet_flag, fs_other_sample, bin_corr_yes,
        panel_in_rf, fs_rf_threshold, post_plots, post_est_stats,
        relative_to_first_group_only, post_kmeans_yes, atet_flag,
        bin_corr_threshold, post_kmeans_no_of_groups, post_kmeans_replications,
        with_output, max_save_values, nn_main_diag_only, fontsize, dpi,
        ci_level, max_weight_share, save_forest, l_centering,
        l_centering_share, l_centering_new_sample, l_centering_cv_k,
        post_random_forest_vi, gmate_no_evaluation_points,
        gmate_sample_share, no_filled_plot, smooth_gates,
        smooth_gates_bandwidth, smooth_gates_no_evaluation_points, show_plots,
        weight_as_sparse, mp_type_weights, mp_weights_tree_batch,
        boot_by_boot, obs_by_obs, max_elements_per_split,
        mp_ray_objstore_multiplier, verbose, no_ray_in_forest_building,
        predict_mcf, train_mcf, forest_files, match_nn_prog_score,
        se_boot_ate, se_boot_gate, se_boot_iate, support_max_del_train,
        _mp_ray_del, _mp_ray_shutdown,
        reduce_split_sample, reduce_split_sample_pred_share, reduce_training,
        reduce_training_share, reduce_prediction, reduce_prediction_share,
        reduce_largest_group_train, reduce_largest_group_train_share,
        iate_flag, iate_se_flag, l_centering_uncenter, d_type, ct_grid_nn,
        ct_grid_w, ct_grid_dr, support_adjust_limits, l_centering_replication,
        iate_eff_flag, _return_iate_sp, iate_cv_flag, iate_cv_folds,
        n_min_treat, gates_minus_previous, _ray_or_dask):
    """Build dictionary with parameters.

    Parameters
    ----------
    ... : All user defined control parameters.

    Returns
    -------
    controls_dict: Dictionary with a collection of these parameters.
    """
    controls_dict = {
        'output_type': output_type, 'outpfad': outpfad, 'datpfad': datpfad,
        'indata': indata, 'preddata': preddata, 'outfiletext': outfiletext,
        'screen_covariates': screen_covariates, 'n_min_grid': n_min_grid,
        'check_perfectcorr': check_perfectcorr, 'n_min_min': n_min_min,
        'clean_data_flag': clean_data_flag, 'min_dummy_obs': min_dummy_obs,
        'mce_vart': mce_vart, 'mtot_p_diff_penalty': p_diff_penalty,
        'boot': boot, 'support_min_p': support_min_p,
        'common_support': support_check, 'support_quantil': support_quantil,
        'support_adjust_limits': support_adjust_limits,
        'support_max_del_train': support_max_del_train,
        'w_yes': weighted, 'knn_min_k': knn_min_k, 'n_min_max': n_min_max,
        'nw_bandw': nw_bandw, 'nw_kern': nw_kern_flag,
        'subsample_factor_forest': subsample_factor_forest,
        'subsample_factor_eval': subsample_factor_eval,
        'm_min_share': m_min_share, 'm_grid': m_grid,
        'm_random_poisson': m_random_poisson, 'alpha_reg_min': alpha_reg_min,
        'alpha_reg_max': alpha_reg_max, 'alpha_reg_grid': alpha_reg_grid,
        'random_thresholds': random_thresholds, 'knn_const': knn_const,
        'share_forest_sample': share_forest_sample, 'knn': knn_flag,
        'desc_stat': descriptive_stats, 'm_max_share': m_max_share,
        'var_import_oob': variable_importance_oob, 'cond_var': cond_var_flag,
        'balancing_test_w': balancing_test, 'max_cats_z_vars': max_cats_z_vars,
        'choice_based_yes': choice_based_sampling, 'panel_data': panel_data,
        'choice_based_probs': choice_based_weights, 'post_plots': post_plots,
        'post_kmeans_max_tries': post_kmeans_max_tries, 'atet_flag': atet_flag,
        'max_cats_cont_vars': max_cats_cont_vars, 'cluster_std': cluster_std,
        'fs_yes': fs_yes, 'fs_other_sample_share': fs_other_sample_share,
        'fs_other_sample': fs_other_sample, 'bin_corr_yes': bin_corr_yes,
        'panel_in_rf': panel_in_rf, 'fs_rf_threshold': fs_rf_threshold,
        'gatet_flag': gatet_flag, 'post_est_stats': post_est_stats,
        'relative_to_first_group_only': relative_to_first_group_only,
        'post_km': post_kmeans_yes, 'bin_corr_thresh': bin_corr_threshold,
        'post_km_no_of_groups': post_kmeans_no_of_groups,
        'post_km_replications': post_kmeans_replications,
        'with_output': with_output, 'no_parallel': mp_parallel,
        'mp_type_vim': mp_type_vim, 'max_save_values': max_save_values,
        'match_nn_prog_score': match_nn_prog_score,
        'nn_main_diag_only': nn_main_diag_only, 'fig_fontsize': fontsize,
        'fig_dpi': dpi, 'fig_ci_level': ci_level,
        'max_weight_share': max_weight_share, 'save_forest': save_forest,
        'l_centering': l_centering, 'l_centering_share': l_centering_share,
        'l_centering_new_sample': l_centering_new_sample,
        'l_centering_cv_k': l_centering_cv_k,
        'l_centering_uncenter': l_centering_uncenter,
        'l_centering_replication': l_centering_replication,
        'post_random_forest_vi': post_random_forest_vi,
        'gmate_no_evaluation_points': gmate_no_evaluation_points,
        'gmate_sample_share': gmate_sample_share,
        'no_filled_plot': no_filled_plot, 'smooth_gates': smooth_gates,
        'sgates_bandwidth': smooth_gates_bandwidth,
        'sgates_no_evaluation_points': smooth_gates_no_evaluation_points,
        'show_plots': show_plots, 'weight_as_sparse': weight_as_sparse,
        'mp_type_weights': mp_type_weights, 'mp_weights_tree_batch':
        mp_weights_tree_batch, 'boot_by_boot': boot_by_boot, 'obs_by_obs':
        obs_by_obs, 'max_elements_per_split': max_elements_per_split,
        'mp_ray_objstore_multiplier': mp_ray_objstore_multiplier,
        'verbose': verbose,
        'no_ray_in_forest_building': no_ray_in_forest_building,
        'pred_mcf': predict_mcf, 'train_mcf': train_mcf,
        'save_forest_files': forest_files,
        'se_boot_ate': se_boot_ate, 'se_boot_gate': se_boot_gate,
        'se_boot_iate': se_boot_iate,
        '_mp_ray_del': _mp_ray_del, '_mp_ray_shutdown': _mp_ray_shutdown,
        'reduce_split_sample': reduce_split_sample,
        'reduce_split_sample_pred_share': reduce_split_sample_pred_share,
        'reduce_training': reduce_training,
        'reduce_training_share': reduce_training_share,
        'reduce_prediction': reduce_prediction,
        'reduce_prediction_share': reduce_prediction_share,
        'reduce_largest_group_train': reduce_largest_group_train,
        'reduce_largest_group_train_share': reduce_largest_group_train_share,
        'iate_flag': iate_flag, 'iate_se_flag': iate_se_flag,
        'iate_eff_flag': iate_eff_flag,
        'd_type': d_type, 'ct_grid_nn': ct_grid_nn, 'ct_grid_w': ct_grid_w,
        'ct_grid_dr': ct_grid_dr,
        '_return_iate_sp': _return_iate_sp,
        'iate_cv_flag': iate_cv_flag,  'iate_cv_folds': iate_cv_folds,
        'n_min_treat': n_min_treat,
        'gates_minus_previous': gates_minus_previous,
        '_ray_or_dask': _ray_or_dask
            }
    return controls_dict


def make_user_variable(
    id_name, cluster_name, w_name, d_name, y_tree_name, y_name, x_name_ord,
    x_name_unord, x_name_always_in_ord, z_name_list, x_name_always_in_unord,
    z_name_split_ord, z_name_split_unord, z_name_mgate, z_name_amgate,
    x_name_remain_ord, x_name_remain_unord, x_balance_name_ord,
        x_balance_name_unord):
    """Put variable names in dictionary."""
    def check_none(name):
        if name is None:
            return []
        return name

    variable_dict = {
        'id_name': check_none(id_name),
        'cluster_name': check_none(cluster_name), 'w_name': check_none(w_name),
        'd_name': check_none(d_name), 'y_tree_name': check_none(y_tree_name),
        'y_name': check_none(y_name), 'x_name_ord': check_none(x_name_ord),
        'x_name_unord': check_none(x_name_unord),
        'x_name_always_in_ord': check_none(x_name_always_in_ord),
        'z_name_list': check_none(z_name_list),
        'x_name_always_in_unord': check_none(x_name_always_in_unord),
        'z_name_ord': check_none(z_name_split_ord),
        'z_name_unord': check_none(z_name_split_unord),
        'z_name_mgate': check_none(z_name_mgate),
        'z_name_amgate': check_none(z_name_amgate),
        'x_name_remain_ord': check_none(x_name_remain_ord),
        'x_name_remain_unord': check_none(x_name_remain_unord),
        'x_balance_name_ord': check_none(x_balance_name_ord),
        'x_balance_name_unord': check_none(x_balance_name_unord),
                     }
    return variable_dict


def get_fig_path(c_dict, add_name, create_dir, no_csv=False):
    """Define and create directories to store figures."""
    fig_pfad = c_dict['outpfad'] + '/' + add_name
    fig_pfad_jpeg = fig_pfad + '/jpeg'
    fig_pfad_csv = fig_pfad + '/csv'
    fig_pfad_pdf = fig_pfad + '/pdf'
    if create_dir:
        if not os.path.isdir(fig_pfad):
            os.mkdir(fig_pfad)
        if not os.path.isdir(fig_pfad_jpeg):
            os.mkdir(fig_pfad_jpeg)
        if not os.path.isdir(fig_pfad_csv) and not no_csv:
            os.mkdir(fig_pfad_csv)
        if not os.path.isdir(fig_pfad_pdf):
            os.mkdir(fig_pfad_pdf)
    c_dict[add_name + '_fig_pfad_jpeg'] = fig_pfad_jpeg
    c_dict[add_name + '_fig_pfad_csv'] = fig_pfad_csv
    c_dict[add_name + '_fig_pfad_pdf'] = fig_pfad_pdf
    return c_dict
