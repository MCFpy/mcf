"""
Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for initialising the parameters of the programme.
@author: MLechner
-*- coding: utf-8 -*-
"""
from psutil import virtual_memory
import numpy as np

from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats_functions as mcf_ps


def int_update_train(mcf_):
    """Update internal parameters before training."""
    dic, cf_dic = mcf_.int_dict, mcf_.cf_dict

    dic['mp_ray_shutdown'] = ray_shut_down(dic['mp_ray_shutdown'],
                                           cf_dic['n_train_eff']
                                           )
    if dic['max_cats_cont_vars'] is None or dic['max_cats_cont_vars'] < 1:
        dic['max_cats_cont_vars'] = cf_dic['n_train_eff'] + 100000000
    else:
        dic['max_cats_cont_vars'] = round(dic['max_cats_cont_vars'])
    if dic['mp_vim_type'] != 1 and dic['mp_vim_type'] != 2:
        dic['mp_vim_type'] = 1 if cf_dic['n_train_eff'] < 20000 else 2
    # 1: MP over var's, fast, lots RAM; 2: MP over bootstraps.
    if cf_dic['n_train_eff'] < 20000:
        dic['mem_object_store_1'] = dic['mem_object_store_2'] = None
    else:
        memory = virtual_memory()
        # Obs. & number of trees as determinants of obj.store when forest build
        min_obj_str_n_1 = (
            (cf_dic['n_train_eff'] / 60000) * (cf_dic['boot'] / 1000)
            * (cf_dic['m_grid'] * cf_dic['n_min_grid']
               * cf_dic['alpha_reg_grid'] / 12)
            * (120 * 1024 * 1024 * 1024) * 5)
        min_obj_str_n_2 = ((cf_dic['n_train_eff'] / 60000)
                           * (cf_dic['boot'] / 1000)
                           * (120 * 1024 * 1024 * 1024))
        dic['mem_object_store_1'] = min(memory.available*0.5, min_obj_str_n_1)
        dic['mem_object_store_2'] = min(memory.available*0.5, min_obj_str_n_2)
        if dic['mp_ray_objstore_multiplier'] > 0:
            dic['mem_object_store_1'] *= dic['mp_ray_objstore_multiplier']
            dic['mem_object_store_2'] *= dic['mp_ray_objstore_multiplier']
        if dic['mem_object_store_1'] > 0.7 * memory.available:
            dic['mem_object_store_1'] = 0.7 * memory.available
        if dic['mem_object_store_2'] > 0.5 * memory.available:
            dic['mem_object_store_2'] = 0.5 * memory.available

    dic['bigdata_train'] = cf_dic['n_train'] > dic['obs_bigdata']

    mcf_.int_dict = dic


def int_update_pred(mcf_, n_pred):
    """Update internal parameters before prediction."""
    int_dic = mcf_.int_dict
    # Adjusted for effective training sample size
    n_pred_adj = n_pred * (mcf_.cf_dict['n_train_eff'] / n_pred)**0.5
    int_dic['mp_ray_shutdown'] = ray_shut_down(int_dic['mp_ray_shutdown'],
                                               n_pred_adj)
    memory = virtual_memory()
    if n_pred_adj < 20000:
        int_dic['mem_object_store_3'] = None
    else:
        min_obj_str_n_3 = (n_pred_adj / 60000) * (120 * 1024 * 1024 * 1024)
        int_dic['mem_object_store_3'] = min(memory.available*0.5,
                                            min_obj_str_n_3)
        if int_dic['mp_ray_objstore_multiplier'] > 0:
            int_dic['mem_object_store_3'] *= int_dic[
                'mp_ray_objstore_multiplier']
        if int_dic['mem_object_store_3'] > 0.5 * memory.available:
            int_dic['mem_object_store_3'] = 0.5 * memory.available
    mcf_.int_dict = int_dic


def gen_update_train(mcf_, data_df):
    """Add and update some dictionary entry based on data."""
    gen_dic, var_dic = mcf_.gen_dict, mcf_.var_dict
    d_dat = data_df[var_dic['d_name']].to_numpy()
    gen_dic['d_values'] = np.unique(np.int32(np.round(d_dat))).tolist()
    gen_dic['no_of_treat'] = len(gen_dic['d_values'])
    if (len(data_df) > mcf_.int_dict['obs_bigdata']
            and gen_dic['mp_parallel'] > 1):
        gen_dic['mp_parallel'] = int(gen_dic['mp_parallel'] * 0.75)

    mcf_.gen_dict = gen_dic


def var_update_train(mcf_, data_df):
    """Update var_dict with training data if needed."""
    var_dic = mcf_.var_dict
    var_dic['d_name'][0] = mcf_gp.adjust_var_name(var_dic['d_name'][0],
                                                  data_df.columns.tolist())
    mcf_.var_dict = var_dic


def ct_update_train(mcf_, data_df):
    """Initialise dictionary with parameters of continuous treatment."""
    gen_dic, ct_dic, var_dic = mcf_.gen_dict, mcf_.ct_dict, mcf_.var_dict
    d_dat = data_df[var_dic['d_name']].to_numpy()
    if gen_dic['d_type'] == 'continuous':
        grid_nn, grid_w, grid_dr = 10, 10, 100
        ct_dic['grid_nn'] = ct_grid(grid_nn, grid_nn)
        ct_dic['grid_w'] = ct_grid(grid_w, grid_w)
        ct_dic['grid_dr'] = ct_grid(grid_dr, grid_dr)
        if ct_dic['grid_dr'] < ct_dic['grid_w']:
            ct_dic['grid_dr'] = ct_dic['grid_w']
        ct_dic['grid_nn_val'] = grid_val(ct_dic['grid_nn'], d_dat)
        ct_dic['grid_w_val'] = grid_val(ct_dic['grid_w'], d_dat)
        ct_dic['grid_dr_val'] = grid_val(ct_dic['grid_dr'], d_dat)
        ct_dic['grid_nn'] = len(ct_dic['grid_nn_val'])
        ct_dic['grid_w'] = len(ct_dic['grid_w_val'])
        ct_dic['grid_dr'] = len(ct_dic['grid_dr_val'])
        gen_dic['no_of_treat'] = len(ct_dic['grid_nn_val'])
        gen_dic['d_values'] = None
        precision_of_cont_treat = 4
        (ct_dic['w_to_dr_int_w01'], ct_dic['w_to_dr_int_w10'],
         ct_dic['w_to_dr_index_full'], ct_dic['d_values_dr_list'],
         ct_dic['d_values_dr_np']) = interpol_weights(
            ct_dic['grid_dr'], ct_dic['grid_w'], ct_dic['grid_w_val'],
            precision_of_cont_treat)
        var_dic['grid_nn_name'] = grid_name(var_dic['d_name'],
                                            ct_dic['grid_nn'])
        var_dic['grid_w_name'] = grid_name(var_dic['d_name'],
                                           ct_dic['grid_w'])
        var_dic['grid_dr_name'] = grid_name(var_dic['d_name'],
                                            ct_dic['grid_dr'])
    else:
        ct_dic['grid_nn'] = ct_dic['grid_nn_val'] = ct_dic['grid_w'] = None
        ct_dic['grid_dr'] = ct_dic['grid_dr_val'] = ct_dic['grid_w_val'] = None
    mcf_.gen_dict = gen_dic
    mcf_.ct_dict = ct_dic


def lc_update_train(mcf_, data_df):
    """Adjust lc for number of training observations."""
    obs = len(data_df)
    if mcf_.lc_dict['cs_cv_k'] is None:
        if obs < 100000:
            mcf_.lc_dict['cs_cv_k'] = 5
        elif obs < 250000:
            mcf_.lc_dict['cs_cv_k'] = 4
        elif obs < 500000:
            mcf_.lc_dict['cs_cv_k'] = 3
        else:
            mcf_.lc_dict['cs_cv_k'] = 2


def cs_update_train(mcf_):
    """Adjust cs for number of treatments."""
    cs_dic = mcf_.cs_dict
    if cs_dic['adjust_limits'] is None:
        cs_dic['adjust_limits'] = (mcf_.gen_dict['no_of_treat'] - 2) * 0.05
    if cs_dic['adjust_limits'] < 0:
        raise ValueError('Negative common support adjustment factor is not'
                         ' possible.')
    mcf_.cs_dict = cs_dic


def cf_update_train(mcf_, data_df):
    """Update cf parameters that need information from training data."""
    cf_dic, gen_dic, lc_dic = mcf_.cf_dict, mcf_.gen_dict, mcf_.lc_dict
    fs_dic, var_dic, int_dic = mcf_.fs_dict, mcf_.var_dict, mcf_.int_dict
    n_train = len(data_df)
    # Number of obs in treatments before any selection
    vcount = data_df.groupby(var_dic['d_name']).size()  # pylint:disable=E1101
    obs_by_treat = vcount.to_numpy()
    if abs(n_train - obs_by_treat.sum()) > len(obs_by_treat):
        raise RuntimeError(
            f'Counting treatments does not work. n_d_sum:'
            f' {obs_by_treat.sum()}, n_train: {n_train}. Difference'
            ' could be due to missing values in treatment.')

    # Adjust for smaller effective training samples due to feature selection
    # and possibly local centering and common support
    cf_dic['n_train'] = reduce_effective_n_train(mcf_, n_train)
    if fs_dic['yes'] and fs_dic['other_sample']:
        obs_by_treat = obs_by_treat * (1 - fs_dic['other_sample_share'])
    if (not isinstance(cf_dic['chunks_maxsize'], (int, float))
            or cf_dic['chunks_maxsize'] < 100):
        cf_dic['baseline'] = 100000
        cf_dic['chunks_maxsize'] = get_chunks_maxsize_forest(
            cf_dic['baseline'], cf_dic['n_train'],
            mcf_.gen_dict['no_of_treat'])
    else:
        cf_dic['chunks_maxsize'] = round(cf_dic['chunks_maxsize'])

    # Effective sample sizes per chuck
    no_of_chuncks = int(np.ceil(cf_dic['n_train'] / cf_dic['chunks_maxsize']))
    # Actual number of chuncks could be smaller if lot's of data is deleted in
    # common support adjustment
    # This will be updated in the train method, adjusting for common support
    cf_dic['n_train_eff'] = np.int32(cf_dic['n_train'] / no_of_chuncks)
    obs_by_treat_eff = np.int32(obs_by_treat / no_of_chuncks)

    # size of subsampling samples         n/2: size of forest sample
    cf_dic['subsample_share_forest'] = sub_size(
        cf_dic['n_train_eff'], cf_dic['subsample_factor_forest'], 0.67)
    if cf_dic['subsample_factor_eval'] is None:
        cf_dic['subsample_factor_eval'] = 2      # Default
    elif cf_dic['subsample_factor_eval'] is False:
        cf_dic['subsample_factor_eval'] = 1000000000
    elif cf_dic['subsample_factor_eval'] is True:
        cf_dic['subsample_factor_eval'] = 2
    elif cf_dic['subsample_factor_eval'] < 0.01:
        cf_dic['subsample_factor_eval'] = 1000000000
    cf_dic['subsample_share_eval'] = min(
        cf_dic['subsample_share_forest'] * cf_dic['subsample_factor_eval'], 1)
    n_d_subsam = (obs_by_treat_eff.min()
                  * int_dic['share_forest_sample']
                  * cf_dic['subsample_share_forest'])

    # Further adjustments when data is used for other purposes
    if mcf_.cs_dict['type'] > 0 and not lc_dic['cs_cv']:
        n_d_subsam *= (1 - lc_dic['cs_share'])
    if lc_dic['yes'] and not lc_dic['cs_cv']:
        n_d_subsam *= (1 - lc_dic['cs_share'])

    # Check only random thresholds to save computation time when building CF
    if cf_dic['random_thresholds'] is None or cf_dic['random_thresholds'] < 0:
        cf_dic['random_thresholds'] = round(4 + cf_dic['n_train_eff']**0.2)
    # Penalty multiplier in CF building
    if (cf_dic['p_diff_penalty'] is None
            or cf_dic['p_diff_penalty'] < 0):  # Default
        if cf_dic['mtot'] == 4:
            cf_dic['p_diff_penalty'] = 0.5
        else:                                   # Approx 1 for N = 1000
            cf_dic['p_diff_penalty'] = (
                2 * ((cf_dic['n_train_eff']
                      * cf_dic['subsample_share_forest'])**0.9)
                / (cf_dic['n_train_eff'] * cf_dic['subsample_share_forest']))
            if cf_dic['mtot'] == 2:
                cf_dic['p_diff_penalty'] = 100 * cf_dic['p_diff_penalty']
            if gen_dic['d_type'] == 'discrete':
                cf_dic['p_diff_penalty'] *= np.sqrt(
                    gen_dic['no_of_treat'] * (gen_dic['no_of_treat'] - 1) / 2)
    elif cf_dic['p_diff_penalty'] == 0:
        if cf_dic['mtot'] == 4:
            cf_dic['mtot'] = 1  # No random mixing  prob of MSE+MCE rule== 1
    else:
        if cf_dic['mtot'] == 4:
            if cf_dic['p_diff_penalty'] > 1:  # if accidently scaled %
                cf_dic['p_diff_penalty'] = cf_dic['p_diff_penalty'] / 100
            if not 0 <= cf_dic['p_diff_penalty'] <= 1:
                raise ValueError('Probability of using p-score > 1. Programm'
                                 ' stopped.')
    if cf_dic['p_diff_penalty']:
        if cf_dic["penalty_type"] == 'mse_d':
            cf_dic['estimator_str'] += ' Penalty "MSE of treatment variable"'
        else:
            cf_dic['estimator_str'] += f' Penalty {cf_dic["penalty_type"]}'
    # Minimum leaf size
    if cf_dic['n_min_min'] is None or cf_dic['n_min_min'] < 1:
        cf_dic['n_min_min'] = round(max((n_d_subsam**0.4) / 10, 1.5)
                                    * gen_dic['no_of_treat'])
    else:
        cf_dic['n_min_min'] = round(cf_dic['n_min_min'])
    if cf_dic['n_min_max'] is None or cf_dic['n_min_max'] < 1:
        cf_dic['n_min_max'] = round(max(n_d_subsam**0.5 / 10, 2)
                                    * gen_dic['no_of_treat'])
    else:
        cf_dic['n_min_max'] = round(cf_dic['n_min_max'])
    cf_dic['n_min_max'] = max(cf_dic['n_min_min'], cf_dic['n_min_max'])
    if gen_dic['d_type'] == 'discrete':
        if cf_dic['n_min_treat'] is None or cf_dic['n_min_treat'] < 1:
            cf_dic['n_min_treat'] = round(
                max((cf_dic['n_min_min'] + cf_dic['n_min_max']) / 2
                    / gen_dic['no_of_treat'] / 10, 1))
        else:
            cf_dic['n_min_treat'] = round(cf_dic['n_min_treat'])
        min_leaf_size = cf_dic['n_min_treat'] * gen_dic['no_of_treat']
        if cf_dic['n_min_min'] < min_leaf_size:
            cf_dic['n_min_min'] = min_leaf_size
            mcf_ps.print_mcf(gen_dic, 'Minimum leaf size adjusted. Smallest ',
                             f' leafsize set to: {cf_dic["n_min_min"]}',
                             summary=True)
    else:
        cf_dic['n_min_treat'] = 0
    if cf_dic['n_min_grid'] is None or cf_dic['n_min_grid'] < 1:
        cf_dic['n_min_grid'] = 1
    else:
        cf_dic['n_min_grid'] = round(cf_dic['n_min_grid'])

    if cf_dic['n_min_min'] == cf_dic['n_min_max']:
        cf_dic['n_min_grid'] = 1
    if cf_dic['n_min_grid'] == 1:
        cf_dic['n_min_min'] = cf_dic['n_min_max'] = round(
            (cf_dic['n_min_min'] + cf_dic['n_min_max']) / 2)
        cf_dic['n_min_values'] = cf_dic['n_min_min']
    else:
        if cf_dic['n_min_grid'] == 2:
            n_min = np.hstack((cf_dic['n_min_min'], cf_dic['n_min_max']))
        else:
            n_min = np.linspace(cf_dic['n_min_min'], cf_dic['n_min_max'],
                                cf_dic['n_min_grid'])
        n_min = list(np.unique(np.round(n_min)))
        cf_dic['n_min_min'], cf_dic['n_min_max'] = n_min[0], n_min[-1]
        cf_dic['n_min_grid'] = len(n_min)
        cf_dic['n_min_values'] = n_min
    cf_dic['forests'] = None              # To be filled at the end of training
    mcf_.cf_dict = cf_dic


def p_update_train(mcf_):
    """Update parameters with sample size information."""
    p_dic = mcf_.p_dict
    # Categorise continuous gate variables
    if p_dic['max_cats_z_vars'] is None or p_dic['max_cats_z_vars'] < 1:
        p_dic['max_cats_z_vars'] = round(mcf_.cf_dict['n_train_eff'] ** 0.3)
    else:
        p_dic['max_cats_z_vars'] = round(p_dic['max_cats_z_vars'])
    mcf_.p_dict = p_dic


def p_update_pred(mcf_, data_df):
    """Update parameters of p_dict with data_df related information."""
    gen_dic, p_dic, var_dic = mcf_.gen_dict, mcf_.p_dict, mcf_.var_dict
    n_pred = len(data_df)
    if p_dic['bgate_sample_share'] is None or p_dic['bgate_sample_share'] <= 0:
        p_dic['bgate_sample_share'] = (
            1 if n_pred < 1000 else (1000 + ((n_pred-1000) ** 0.75)) / n_pred)
    d_name = (var_dic['d_name'][0]
              if isinstance(var_dic['d_name'], (list, tuple))
              else var_dic['d_name'])
    # Capitalise all variable names
    data_df = data_df.rename(columns=lambda x: x.casefold())
    # Check if treatment is included
    if gen_dic['d_type'] == 'continuous':
        p_dic['d_in_pred'] = False
    else:
        p_dic['d_in_pred'] = d_name in data_df.columns
    if not p_dic['d_in_pred']:
        if p_dic['atet'] or p_dic['gatet']:
            mcf_ps.print_mcf(gen_dic,
                             'Treatment variable not in prediction data. ATET '
                             'and GATET cannot be computed.',
                             summary=True)

        p_dic['atet'] = p_dic['gatet'] = False
        if p_dic['choice_based_sampling']:
            raise ValueError('Choice based sampling relates only to prediction'
                             ' data. It requires treatment information in'
                             ' prediction data, WHICH IS MISSING!')
    if p_dic['choice_based_sampling'] is True:
        if len(p_dic['choice_based_probs']) != gen_dic['no_of_treat']:
            raise ValueError('Choice based sampling. Rows in choice'
                             ' probabilities do not correspond to number of'
                             ' treatments.')
        if any(v <= 0 for v in p_dic['choice_based_probs']):
            raise ValueError('Choice based sampling active. Not possible to'
                             ' have zero or negative choice probability.')
        # Normalize
        pcb = np.array(p_dic['choice_based_probs'])
        pcb = pcb / np.sum(pcb) * gen_dic['no_of_treat']
        p_dic['choice_based_probs'] = pcb.tolist()
    else:
        p_dic['choice_based_sampling'], p_dic['choice_based_probs'] = False, 1
    mcf_.p_dict = p_dic
    return data_df


def post_update_pred(mcf_, data_df):
    """Update entries in post_dic that need info from prediction data."""
    n_pred = len(data_df)
    post_dic = mcf_.post_dict
    if isinstance(post_dic['kmeans_no_of_groups'], (int, float)):
        post_dic['kmeans_no_of_groups'] = [(post_dic['kmeans_no_of_groups'])]
    if (post_dic['kmeans_no_of_groups'] is None
        or len(post_dic['kmeans_no_of_groups']) == 1
            or post_dic['kmeans_no_of_groups'][0] < 2):
        if n_pred < 10000:
            middle = 5
        elif n_pred > 100000:
            middle = 10
        else:
            middle = 5 + round(n_pred/20000)
        if middle < 7:
            post_dic['kmeans_no_of_groups'] = [
                middle-2, middle-1, middle, middle+1, middle+2]
        else:
            post_dic['kmeans_no_of_groups'] = [
                middle-4, middle-2, middle, middle+2, middle+4]
    else:
        if not isinstance(post_dic['kmeans_no_of_groups'], list):
            post_dic['kmeans_no_of_groups'] = list(
                post_dic['kmeans_no_of_groups'])
            post_dic['kmeans_no_of_groups'] = [
                round(a) for a in post_dic['kmeans_no_of_groups']]
    mcf_.post_dict = post_dic


def name_unique(all_names):
    """Remove any duplicates."""
    seen = set()
    name_unique_ = [
        item for item in all_names if item not in seen and not seen.add(item)]
    return name_unique_


def get_ray_del_defaults(_mp_ray_del_user):
    """Get values for :mp_ray_del."""
    if _mp_ray_del_user is None:
        _mp_ray_del = ('refs',)
    else:
        possible_vals = ('refs', 'rest', 'none')
        if isinstance(_mp_ray_del_user, str):
            _mp_ray_del = (_mp_ray_del_user,)
        elif isinstance(_mp_ray_del_user, list):
            _mp_ray_del = tuple(_mp_ray_del_user)
        else:
            _mp_ray_del = _mp_ray_del_user
        if len(_mp_ray_del) > 2:
            raise ValueError(
                f'Too many parameters for _mp_ray_del{_mp_ray_del}')
        if not isinstance(_mp_ray_del, tuple):
            raise ValueError(f'mp_ray_del is no Tuple {_mp_ray_del}')
        if not all(i in possible_vals for i in _mp_ray_del):
            raise ValueError(f'Wrong values for _mp_ray_del {_mp_ray_del}')
    return _mp_ray_del


def ray_shut_down(ray_shutdown, len_data):
    """Define mimimum sample size for ray_shut_down."""
    if ray_shutdown is None:
        ray_shutdown = not len_data < 150000
    if not isinstance(ray_shutdown, bool):
        raise ValueError('mp_ray_shutdown must be either None or Boolean')
    return ray_shutdown


def get_alpha(alpha_reg_grid, alpha_reg_max, alpha_reg_min):
    """Get the alphas for the CF."""
    if alpha_reg_min is None or not 0 <= alpha_reg_min < 0.4:
        alpha_reg_min = 0.05
    if alpha_reg_max is None or not 0 <= alpha_reg_max < 0.4:
        alpha_reg_max = 0.15
    alpha_reg_grid = (1 if alpha_reg_grid is None or alpha_reg_grid < 1
                      else round(alpha_reg_grid))
    if alpha_reg_min >= alpha_reg_max:
        alpha_reg_grid, alpha_reg_max = 1, alpha_reg_min
    if alpha_reg_grid == 1:
        alpha_reg = (alpha_reg_max + alpha_reg_min) / 2
    elif alpha_reg_grid == 2:
        alpha_reg = np.hstack((alpha_reg_min, alpha_reg_max))
    else:
        alpha_reg = np.linspace(alpha_reg_min, alpha_reg_max, alpha_reg_grid)
        alpha_reg = list(np.unique(alpha_reg))
        alpha_reg_grid = len(alpha_reg)
    return alpha_reg_grid, alpha_reg_max, alpha_reg_min, alpha_reg


def grid_val(grid, d_dat):
    """Help for initialisation."""
    quantile = np.linspace(1/(grid)/2, 1-1/grid/2, num=grid)
    d_dat_min = d_dat.min()
    d_dat_r = d_dat - d_dat_min if d_dat_min != 0 else d_dat
    gridvalues = np.around(
        np.quantile(d_dat_r[d_dat_r > 1e-15], quantile), decimals=6)
    gridvalues = np.insert(gridvalues, 0, 0)
    return gridvalues


def ct_grid(user_grid, defaultgrid):
    """Help for initialisation."""
    if isinstance(user_grid, int):
        grid = defaultgrid if user_grid < 1 else user_grid
    else:
        grid = defaultgrid
    return grid


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
    if len(treat_val_np) != len(np.unique(treat_val_np)):
        raise ValueError('Continuous treatment needs higher precision')
    index_full[ct_grid_w-1, 0] = j_all
    index_full = np.int32(index_full)

    return int_w01, int_w10, index_full, treat_val_list, treat_val_np


def grid_name(d_name, add_name):
    """Help for initialisation."""
    grid_name_tmp = d_name[0] + str(add_name)
    grid_name_l = [grid_name_tmp.casefold()]
    return grid_name_l


def sub_size(n_train, share_mult, max_share):
    """Help for initialisation."""
    if share_mult is None or share_mult <= 0:
        share_mult = 1
    subsam_share = min(4 * ((n_train / 2)**0.85) / n_train, 0.67) * share_mult
    subsam_share = max(min(subsam_share, max_share),
                       (2 * (n_train / 2)**0.5) / n_train)
    return subsam_share


def bootstrap(se_boot, cut_off, bnr, cluster_std):
    """Check and correct bootstrap level."""
    if se_boot is None:
        se_boot = bnr if cluster_std else False
    if 0 < se_boot < cut_off:
        return bnr
    if se_boot >= cut_off:
        return round(se_boot)
    return False


def reduce_effective_n_train(mcf_, n_train):
    """Compute effective training sample size."""
    if mcf_.fs_dict['yes'] and mcf_.fs_dict['other_sample']:
        n_train *= 1 - mcf_.fs_dict['other_sample_share']
    if mcf_.cs_dict['type'] > 0 and not mcf_.lc_dict['cs_cv']:
        n_train *= (1 - mcf_.lc_dict['cs_share'])
    if mcf_.lc_dict['yes'] and not mcf_.lc_dict['cs_cv']:
        n_train *= (1 - mcf_.lc_dict['cs_share'])
    return int(n_train)


def get_chunks_maxsize_forest(base_level, obs, no_of_treat):
    """Compute optimal chunksize for forest splitting."""
    return round(base_level
                 + (max(obs - base_level, 0) ** 0.8) / (no_of_treat - 1))
