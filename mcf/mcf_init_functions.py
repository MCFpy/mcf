"""Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for initialising the parameters of the programme.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy

import psutil
import numpy as np

from mcf import mcf_general as gp
from mcf import mcf_print_stats_functions as ps
from mcf import mcf_general_sys as mcf_sys


def int_init(descriptive_stats=None, dpi=None, fontsize=None,
             no_filled_plot=None, max_save_values=None,
             max_cats_cont_vars=None, mp_ray_del=None,
             mp_ray_objstore_multiplier=None, mp_ray_shutdown=None,
             mp_vim_type=None, mp_weights_tree_batch=None,
             mp_weights_type=None,  ray_or_dask=None, return_iate_sp=None,
             seed_sample_split=None, share_forest_sample=None,
             show_plots=None, verbose=None, smaller_sample=None,
             weight_as_sparse=None, weight_as_sparse_splits=None,
             with_output=None):
    """Initialise dictionary of parameters of internal variables."""
    dic = {}
    dic['descriptive_stats'] = descriptive_stats is not False
    dic['dpi'] = 500 if (dpi is None or dpi < 10) else round(dpi)
    if fontsize is not None and 0.5 < fontsize < 7.5:
        dic['fontsize'] = round(fontsize)
    else:
        dic['fontsize'] = 2
    dic['all_fonts'] = ('xx-small', 'x-small', 'small', 'medium', 'large',
                        'x-large', 'xx-large')
    for i, i_lab in enumerate(dic['all_fonts']):
        if dic['fontsize'] == i + 1:
            dic['fontsize'] = i_lab
    dic['legend_loc'] = 'best'
    if no_filled_plot is None or no_filled_plot < 5:
        dic['no_filled_plot'] = 20
    else:
        dic['no_filled_plot'] = round(no_filled_plot)
    dic['mp_ray_del'] = get_ray_del_defaults(mp_ray_del)
    if mp_ray_objstore_multiplier is None or mp_ray_objstore_multiplier < 0:
        dic['mp_ray_objstore_multiplier'] = 1
    else:
        dic['mp_ray_objstore_multiplier'] = mp_ray_objstore_multiplier
    dic['ray_or_dask'] = 'ray' if ray_or_dask != 'dask' else 'dask'
    dic['no_ray_in_forest_building'] = False
    if mp_weights_tree_batch is not None and 0 < mp_weights_tree_batch < 1:
        dic['mp_weights_tree_batch'] = round(mp_weights_tree_batch)
    else:
        dic['mp_weights_tree_batch'] = 0
    if share_forest_sample is None or not 0.01 < share_forest_sample < 0.99:
        dic['share_forest_sample'] = 0.5
    else:
        dic['share_forest_sample'] = share_forest_sample
    dic['weight_as_sparse'] = weight_as_sparse is not False

    dic['show_plots'] = show_plots is not False
    dic['with_output'] = with_output is not False
    dic['verbose'] = verbose is not False
    if not dic['with_output']:
        dic['verbose'] = False
    dic['max_save_values'] = (50 if max_save_values is None
                              else max_save_values)
    if mp_weights_type is None or mp_weights_type != 2:
        dic['mp_weights_type'] = 1
    else:
        dic['mp_weights_type'] = 2
    dic['return_iate_sp'] = return_iate_sp is True
    if dic['with_output']:
        dic['return_iate_sp'] = True
    dic['seed_sample_split'] = (67567885 if seed_sample_split is None
                                else seed_sample_split)
    if isinstance(smaller_sample, float) and 0 < smaller_sample < 1:
        dic['smaller_sample'] = smaller_sample
    else:
        dic['smaller_sample'] = None
    if not isinstance(weight_as_sparse_splits, int):  # To be updated later
        dic['weight_as_sparse_splits'] = None
    else:
        dic['weight_as_sparse_splits'] = weight_as_sparse_splits
    # Variables varying with data. To be initialised when training.
    dic['mp_ray_shutdown'] = mp_ray_shutdown
    dic['mp_vim_type'] = mp_vim_type
    dic['max_cats_cont_vars'] = max_cats_cont_vars
    return dic


def int_update_train(mcf_, n_train):
    """Update internal parameters before training."""
    dic, cf_dic = mcf_.int_dict, mcf_.cf_dict
    dic['mp_ray_shutdown'] = ray_shut_down(dic['mp_ray_shutdown'], n_train)
    if dic['max_cats_cont_vars'] is None or dic['max_cats_cont_vars'] < 1:
        dic['max_cats_cont_vars'] = n_train + 100000000
    else:
        dic['max_cats_cont_vars'] = round(dic['max_cats_cont_vars'])
    if dic['mp_vim_type'] != 1 and dic['mp_vim_type'] != 2:
        dic['mp_vim_type'] = 1 if n_train < 20000 else 2
    # 1: MP over var's, fast, lots RAM; 2: MP over bootstraps.
    if n_train < 20000:
        dic['mem_object_store_1'] = dic['mem_object_store_2'] = None
    else:
        memory = psutil.virtual_memory()
        # Obs. & number of trees as determinants of obj.store when forest build
        min_obj_str_n_1 = (
            (n_train / 60000) * (cf_dic['boot'] / 1000)
            * (cf_dic['m_grid'] * cf_dic['n_min_grid']
               * cf_dic['alpha_reg_grid'] / 12)
            * (120 * 1024 * 1024 * 1024) * 5)
        min_obj_str_n_2 = ((n_train / 60000) * (cf_dic['boot'] / 1000)
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
    mcf_.int_dict = dic


def int_update_pred(mcf_, n_pred):
    """Update internal parameters before prediction."""
    dic = mcf_.int_dict
    dic['mp_ray_shutdown'] = ray_shut_down(dic['mp_ray_shutdown'], n_pred)
    memory = psutil.virtual_memory()
    if n_pred < 20000:
        dic['mem_object_store_3'] = None
    else:
        min_obj_str_n_3 = (n_pred / 60000) * (120 * 1024 * 1024 * 1024)
        dic['mem_object_store_3'] = min(memory.available*0.5, min_obj_str_n_3)
        if dic['mp_ray_objstore_multiplier'] > 0:
            dic['mem_object_store_3'] *= dic['mp_ray_objstore_multiplier']
        if dic['mem_object_store_3'] > 0.5 * memory.available:
            dic['mem_object_store_3'] = 0.5 * memory.available
    mcf_.int_dict = dic


def gen_init(int_dic, d_type=None, iate_eff=None, mp_parallel=None,
             outfiletext=None, outpath=None, output_type=None,
             replication=None, weighted=None, panel_data=None,
             panel_in_rf=None):
    """Initialise dictionary with general parameters."""
    dic = {}
    # Discrete or continuous treatments
    if d_type is None or d_type == 'discrete':
        dic['d_type'] = 'discrete'
    elif d_type == 'continuous':
        dic['d_type'] = 'continuous'
    else:
        raise ValueError(f'{d_type} is wrong treatment type.')
    #  More precise IATEs
    dic['iate_eff'] = iate_eff is True
    # Number of cores for multiprocessiong
    if mp_parallel is None or not isinstance(mp_parallel, (float, int)):
        dic['mp_parallel'] = round(psutil.cpu_count(logical=True)*0.8)
        dic['mp_automatic'] = True
    elif mp_parallel <= 1.5:
        dic['mp_parallel'] = 1
        dic['mp_automatic'] = False
    else:
        dic['mp_parallel'] = round(mp_parallel)
        dic['mp_automatic'] = False
    dic['replication'] = replication is True
    dic['sys_share'] = 0.7 * getattr(psutil.virtual_memory(), 'percent') / 100
    # Define or create directory for output and avoid overwritting
    if int_dic['with_output']:
        dic['outpath'] = mcf_sys.define_outpath(outpath)
        # Files to write output to
        dic['outfiletext'] = ('txtFileWithOutput'
                              if outfiletext is None else outfiletext)
        dic['outfiletext'] = dic['outpath'] + '/' + dic['outfiletext'] + '.txt'
        dic['outfilesummary'] = dic['outfiletext'][:-4] + '_Summary.txt'
        mcf_sys.delete_file_if_exists(dic['outfiletext'])
        mcf_sys.delete_file_if_exists(dic['outfilesummary'])
    else:
        dic['outpath'] = dic['outfiletext'] = dic['outfilesummary'] = None
    # Output
    dic['output_type'] = 2 if output_type is None else output_type
    if dic['output_type'] == 0:
        dic['print_to_file'], dic['print_to_terminal'] = False, True
    elif dic['output_type'] == 1:
        dic['print_to_file'], dic['print_to_terminal'] = True, False
    else:
        dic['print_to_file'] = dic['print_to_terminal'] = True
    if not int_dic['with_output']:
        dic['print_to_file'] = dic['print_to_terminal'] = False
    dic['with_output'] = int_dic['with_output']
    dic['verbose'] = int_dic['verbose']
    # Weighting
    dic['weighted'] = weighted is True
    # Panel data
    if panel_data is True:
        dic['panel_in_rf'] = panel_in_rf is not False
        dic['panel_data'] = True
    else:
        dic['panel_data'] = dic['panel_in_rf'] = False
    return dic


def gen_update_train(mcf_, data_df):
    """Add and update some dictionary entry based on data."""
    gen_dic, var_dic = mcf_.gen_dict, mcf_.var_dict
    d_dat = data_df[var_dic['d_name']].to_numpy()
    gen_dic['d_values'] = np.int16(np.round(np.unique(d_dat))).tolist()
    gen_dic['no_of_treat'] = len(gen_dic['d_values'])
    mcf_.gen_dict = gen_dic


def var_init(gen_dic, fs_dic, p_dic, bgate_name=None, cluster_name=None,
             d_name=None, id_name=None, w_name=None, x_balance_name_ord=None,
             x_balance_name_unord=None, x_name_always_in_ord=None,
             x_name_always_in_unord=None, x_name_remain_ord=None,
             x_name_remain_unord=None, x_name_ord=None,
             x_name_unord=None, y_name=None, y_tree_name=None,
             z_name_list=None, z_name_ord=None, z_name_unord=None):
    """Initialise dictionary with parameters of variance computation."""
    # Check for missing information
    if y_name is None or y_name == []:
        raise ValueError('y_name must be specified.')
    if d_name is None or d_name == []:
        raise ValueError('d_name must be specified.')
    if p_dic['cluster_std'] or gen_dic['panel_data']:
        if cluster_name is None or cluster_name == []:
            raise ValueError('cluster_name must be specified.')
    # Consistency check and assignments
    if y_tree_name is None or y_tree_name == []:
        if isinstance(y_name, (list, tuple)):
            y_tree_name = [y_name[0]]
        else:
            y_tree_name = [y_name]
    if (x_name_ord is None or x_name_ord == []) and (
            x_name_unord is None or x_name_unord == []):
        raise ValueError('x_name_ord or x_name_unord must be specified.')
    if p_dic['cluster_std'] or gen_dic['panel_data']:
        cluster_name = gp.cleaned_var_names(cluster_name)
    else:
        cluster_name = []
    # Clean variable names ... set all to capital letters
    d_name = gp.cleaned_var_names(gp.to_list_if_needed(d_name))
    y_tree_name = gp.cleaned_var_names(gp.to_list_if_needed(y_tree_name))
    y_name = gp.to_list_if_needed(y_name)
    y_name.extend(y_tree_name)
    y_name = gp.cleaned_var_names(y_name)
    x_name_always_in_ord = gp.cleaned_var_names(x_name_always_in_ord)
    x_name_always_in_unord = gp.cleaned_var_names(x_name_always_in_unord)
    x_name_remain_ord = gp.cleaned_var_names(x_name_remain_ord)
    x_name_remain_unord = gp.cleaned_var_names(x_name_remain_unord)
    x_balance_name_ord = gp.cleaned_var_names(x_balance_name_ord)
    x_balance_name_unord = gp.cleaned_var_names(x_balance_name_unord)
    z_name_list = gp.cleaned_var_names(z_name_list)
    z_name_ord = gp.cleaned_var_names(z_name_ord)
    z_name_unord = gp.cleaned_var_names(z_name_unord)
    bgate_name = gp.cleaned_var_names(bgate_name) if p_dic['bgate'] else []
    x_name_ord = gp.cleaned_var_names(x_name_ord)
    x_name_unord = gp.cleaned_var_names(x_name_unord)
    if gen_dic['weighted']:      # Former w_yes
        if w_name is None or w_name == []:
            raise ValueError('No name for sample weights specified.')
        w_name = gp.cleaned_var_names(w_name)
    else:
        w_name = []
    id_name = gp.cleaned_var_names(id_name)
    x_name = deepcopy(x_name_ord + x_name_unord)
    x_name = gp.cleaned_var_names(x_name)
    x_name_in_tree = deepcopy(x_name_always_in_ord + x_name_always_in_unord)
    x_name_in_tree = gp.cleaned_var_names(x_name_in_tree)
    x_balance_name = gp.cleaned_var_names(deepcopy(x_balance_name_ord
                                                   + x_balance_name_unord))
    if not x_balance_name:
        p_dic['bt_yes'] = False
    x_name_remain = gp.cleaned_var_names(
        deepcopy(x_name_remain_ord + x_name_remain_unord + x_name_in_tree
                 + x_balance_name))
    x_name_always_in = gp.cleaned_var_names(deepcopy(x_name_always_in_ord
                                                     + x_name_always_in_unord))
    name_ordered = gp.cleaned_var_names(
        deepcopy(x_name_ord + x_name_always_in_ord + x_name_remain_ord))
    name_unordered = gp.cleaned_var_names(
        deepcopy(x_name_unord + x_name_always_in_unord + x_name_remain_unord))
    if fs_dic['yes']:
        if p_dic['bt_yes']:
            x_name_remain = gp.cleaned_var_names(deepcopy(x_balance_name
                                                          + x_name_remain))
    if x_name_in_tree:
        x_name_remain = gp.cleaned_var_names(deepcopy(x_name_in_tree
                                                      + x_name_remain))
        x_name = gp.cleaned_var_names(deepcopy(x_name_in_tree + x_name))
    if not ((not name_ordered) or (not name_unordered)):
        if any(value for value in name_ordered if value in name_unordered):
            raise ValueError('Remove overlap in ordered + unordered variables')
    # Define variables for consistency check in data sets
    names_to_check_train = d_name + y_name + x_name
    names_to_check_pred = x_name[:]
    if (not z_name_list) and (not z_name_ord) and (not z_name_unord):
        gen_dic['agg_yes'] = p_dic['gate'] = False
        z_name = []
    else:
        gen_dic['agg_yes'] = p_dic['gate'] = True
        if z_name_list:
            names_to_check_train.extend(z_name_list)
            names_to_check_pred.extend(z_name_list)
        if z_name_ord:
            names_to_check_train.extend(z_name_ord)
            names_to_check_pred.extend(z_name_ord)
        if z_name_unord:
            names_to_check_train.extend(z_name_unord)
            names_to_check_pred.extend(z_name_unord)
        z_name = z_name_list + z_name_ord + z_name_unord
    txt = '\n'
    if p_dic['bgate'] and not p_dic['gate']:
        txt += 'BGATEs can only be computed if GATEs are computed.'
        p_dic['bgate'] = False
    if p_dic['amgate'] and not p_dic['gate']:
        txt += 'AMGATEs can only be computed if GATEs are computed.'
        p_dic['amgate'] = False
    if p_dic['gatet'] and not p_dic['gate']:
        txt += 'GATETs can only be computed if GATEs are computed.'
        p_dic['gatet'] = False
    ps.print_mcf(gen_dic, txt, summary=True)
    if p_dic['bgate'] and bgate_name == z_name and len(z_name) == 1:
        p_dic['bgate'] = False
    if p_dic['bgate']:
        if (bgate_name is None or bgate_name == []):
            if len(z_name) > 1:
                bgate_name = z_name[:]
            else:
                p_dic['bgate'], bgate_name = False, []
        else:
            names_to_check_train.extend(bgate_name)
            names_to_check_pred.extend(bgate_name)
    else:
        bgate_name = []
    if bgate_name == [] and len(z_name) == 1:
        p_dic['bgate'] = False
    dic = {
        'bgate_name': bgate_name, 'cluster_name': cluster_name,
        'd_name': d_name, 'id_name': id_name, 'name_unordered': name_unordered,
        'names_to_check_train': names_to_check_train,
        'names_to_check_pred': names_to_check_pred,
        'w_name': w_name, 'x_balance_name_ord': x_balance_name_ord,
        'x_balance_name_unord': x_balance_name_unord,
        'x_name_always_in_ord': x_name_always_in_ord,
        'x_name_always_in_unord': x_name_always_in_unord,
        'x_name_remain_ord': x_name_remain_ord,
        'x_name_remain_unord': x_name_remain_unord,
        'x_name_ord': x_name_ord, 'x_name_unord': x_name_unord,
        'y_name': y_name, 'y_tree_name': y_tree_name,
        'x_balance_name': x_balance_name, 'x_name_always_in': x_name_always_in,
        'name_ordered': name_ordered, 'x_name_remain': x_name_remain,
        'x_name': x_name, 'x_name_in_tree': x_name_in_tree, 'z_name': z_name,
        'z_name_list': z_name_list, 'z_name_ord': z_name_ord,
        'z_name_unord': z_name_unord}
    return dic, gen_dic, p_dic


def var_update_train(mcf_, data_df):
    """Update var_dict with training data if needed."""
    var_dic = mcf_.var_dict
    var_dic['d_name'][0] = gp.adjust_var_name(var_dic['d_name'][0],
                                              data_df.columns.tolist())
    mcf_.var_dict = var_dic


def dc_init(check_perfectcorr=None, clean_data=None, min_dummy_obs=None,
            screen_covariates=None):
    """Initialise dictionary with parameters of data cleaning."""
    dic = {}
    dic['screen_covariates'] = screen_covariates is not False
    dic['check_perfectcorr'] = check_perfectcorr is not False
    dic['clean_data'] = clean_data is not False
    dic['min_dummy_obs'] = (10 if min_dummy_obs is None or min_dummy_obs < 1
                            else round(min_dummy_obs))
    return dic


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


# def ct_update_pred(mcf_):
#     """Initialise dictionary with parameters of continuous treatment."""
#     ct_dic, var_dic = mcf_.ct_dict, mcf_.var_dict
#     gen_dic = mcf_.gen_dict
#     if gen_dic['d_type'] == 'continuous':
#         var_dic['grid_nn_name'] = grid_name(var_dic['d_name'],
#                                             ct_dic['grid_nn'])
#         var_dic['grid_w_name'] = grid_name(var_dic['d_name'],
#                                            ct_dic['grid_w'])
#         var_dic['grid_dr_name'] = grid_name(var_dic['d_name'],
#                                             ct_dic['grid_dr'])
#     else:
#         var_dic['grid_nn_name'] = var_dic['grid_w_name'] = None
#         var_dic['grid_dr_name'] = None
#     return var_dic


def fs_init(rf_threshold=None, other_sample=None, other_sample_share=None,
            yes=None):
    """Initialise dictionary with parameters of feature selection."""
    dic = {}
    dic['yes'] = yes is True
    if rf_threshold is None or rf_threshold <= 0 or rf_threshold > 100:
        dic['rf_threshold'] = 1
    else:
        dic['rf_threshold'] = rf_threshold
    dic['rf_threshold'] /= 100
    dic['other_sample'] = other_sample is not False
    if other_sample_share is None or not (
            0 <= other_sample_share <= 0.5):
        dic['other_sample_share'] = 0.33
    else:
        dic['other_sample_share'] = other_sample_share    
    if dic['other_sample'] is False or (dic['yes'] is False):
        dic['other_sample_share'] = 0
    return dic


def cs_init(gen_dic, max_del_train=None, min_p=None, quantil=None, type_=None,
            adjust_limits=None):
    """Initialise dictionary with parameters of common support."""
    dic = {}
    # type_ corresponds to former common_support and support_check
    if gen_dic['d_type'] == 'continuous':  # No common support check
        type_ = 0
    dic['type'] = 1 if type_ not in (0, 1, 2) else type_
    dic['quantil'] = (1 if quantil is None or not (0 <= quantil <= 1)
                      else quantil)
    dic['min_p'] = 0.01 if min_p is None or not (0 <= min_p <= 0.5) else min_p
    dic['max_del_train'] = (
        0.5 if max_del_train is None or not (0 < max_del_train <= 1)
        else max_del_train)
    # Data dependent, to be adjusted later
    dic['adjust_limits'] = adjust_limits
    if gen_dic['outpath'] is not None:
        dic = mcf_sys.get_fig_path(dic, gen_dic['outpath'], 'common_support',
                                   gen_dic['with_output'], no_csv=True)
    dic['cut_offs'], dic['forests'] = None, None
    return dic


def cs_update_train(mcf_):
    """Adjust cs for number of treatments."""
    cs_dic = mcf_.cs_dict
    if cs_dic['adjust_limits'] is None:
        cs_dic['adjust_limits'] = (mcf_.gen_dict['no_of_treat'] - 2) * 0.05
    if cs_dic['adjust_limits'] < 0:
        raise ValueError('Negative common support adjustment factor is not'
                         ' possible.')
    mcf_.cs_dict = cs_dic


def lc_init(cs_cv=None, cs_cv_k=None, cs_share=None, undo_iate=None, yes=None):
    """Initialise dictionary with parameters of local centering, CS and CV."""
    # General parameters for crossvalidation or using a new sample
    dic = {}
    dic['cs_cv'] = cs_cv is not False
    dic['cs_share'] = (0.25 if cs_share is None
                       or not (0.0999 < cs_share < 0.9001) else cs_share)
    dic['cs_cv_k'] = (5 if cs_cv_k is None or cs_cv_k < 1 else round(cs_cv_k))
    # local centering
    # dic['cs_cv'] corresponds to cnew_dict['l_centering_new_sample']
    dic['yes'] = yes is not False
    dic['uncenter_po'] = undo_iate is not False
    if not dic['yes']:
        dic['uncenter_po'] = False
    # if fs_dic['yes']:   # This is most likely from the old version where fs
    #     dic['uncenter_po'] = False  # was conducted after local centering
    return dic


def cf_init(alpha_reg_grid=None, alpha_reg_max=None, alpha_reg_min=None,
            boot=None, chunks_maxsize=None, nn_main_diag_only=None,
            m_grid=None, m_share_max=None, m_share_min=None,
            m_random_poisson=None, match_nn_prog_score=None, mce_vart=None,
            vi_oob_yes=None, n_min_grid=None, n_min_max=None, n_min_min=None,
            n_min_treat=None, p_diff_penalty=None, subsample_factor_eval=None,
            subsample_factor_forest=None, random_thresholds=None):
    """Initialise dictionary with parameters of causal forest building."""
    dic = {}
    (dic['alpha_reg_grid'], dic['alpha_reg_max'], dic['alpha_reg_min'],
     dic['alpha_reg_values']) = get_alpha(alpha_reg_grid, alpha_reg_max,
                                          alpha_reg_min)
    dic['boot'] = 1000 if boot is None or boot < 1 else round(boot)
    dic['match_nn_prog_score'] = match_nn_prog_score is not False
    dic['nn_main_diag_only'] = nn_main_diag_only is True
    # Select grid for number of parameters
    if m_share_min is None or not 0 < m_share_min <= 1:
        dic['m_share_min'] = 0.1
    if m_share_max is None or not 0 < m_share_max <= 1:
        dic['m_share_max'] = 0.6
    if m_random_poisson is False:
        dic['m_random_poisson'] = False
        dic['m_random_poisson_min'] = 1000000
    else:
        dic['m_random_poisson'] = True
        dic['m_random_poisson_min'] = 10
    dic['m_grid'] = 1 if m_grid is None or m_grid < 1 else round(m_grid)
    if mce_vart is None or mce_vart == 1:
        mtot, mtot_no_mce, estimator_str = 1, 0, 'MSE & MCE'    # MSE + MCE
    elif mce_vart == 2:                  # -Var(treatment effect)
        mtot, mtot_no_mce, estimator_str = 2, 1, '-Var(effect)'
    elif mce_vart == 0:                             # MSE rule
        mtot, mtot_no_mce, estimator_str = 3, 1, 'MSE'
    elif mce_vart == 3:  # MSE+MCE rule or penalty function rule
        mtot, mtot_no_mce = 4, 0                      # (randomly decided)
        estimator_str = 'MSE,MCE or penalty (random)'
    else:
        raise ValueError('Inconsistent MTOT definition of  MCE_VarT.')
    dic['mtot'], dic['mtot_no_mce'] = mtot, mtot_no_mce
    dic['estimator_str'] = estimator_str
    dic['chunks_maxsize'] = chunks_maxsize
    dic['vi_oob_yes'] = vi_oob_yes is True
    dic['n_min_grid'], dic['n_min_max'] = n_min_grid, n_min_max
    dic['n_min_min'], dic['n_min_treat'] = n_min_min, n_min_treat
    dic['p_diff_penalty'] = p_diff_penalty
    dic['subsample_factor_eval'] = subsample_factor_eval
    dic['subsample_factor_forest'] = subsample_factor_forest
    dic['random_thresholds'] = random_thresholds
    return dic


def cf_update_train(mcf_, data_df):
    """Update cf parameters that need information from training data."""
    cf_dic, gen_dic, lc_dic = mcf_.cf_dict, mcf_.gen_dict, mcf_.lc_dict
    fs_dic, var_dic, int_dic = mcf_.fs_dict, mcf_.var_dict, mcf_.int_dict
    n_train = len(data_df)
    # Number of obs in treatments before any selection
    vcount = data_df.groupby(var_dic['d_name']).size()  # pylint:disable=E1101
    gen_dic['obs_by_treat'] = vcount.to_numpy()
    if n_train != gen_dic['obs_by_treat'].sum():
        raise RuntimeError(
            f'Counting treatments does not work. n_d_sum:'
            f' {gen_dic["obs_by_treat"].sum()}, n_train: {n_train}. Difference'
            ' could be due to missing values in treatment.')
    # size of subsampling samples         n/2: size of forest sample
    cf_dic['subsample_share_forest'] = sub_size(
        n_train, cf_dic['subsample_factor_forest'], 0.67)
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
    n_d_subsam = (gen_dic['obs_by_treat'].min()
                  * int_dic['share_forest_sample']
                  * cf_dic['subsample_share_forest'])
    if fs_dic['yes'] and lc_dic['cs_cv']:
        n_d_subsam *= (1 - lc_dic['cs_share'])
    if lc_dic['yes'] and lc_dic['cs_cv']:
        n_d_subsam *= (1 - lc_dic['cs_share'])
    # Check only random thresholds to save computation time when building CF
    if cf_dic['random_thresholds'] is None or cf_dic['random_thresholds'] < 0:
        cf_dic['random_thresholds'] = round(4 + n_train**0.2)
    # Penalty multiplier in CF building
    if (cf_dic['p_diff_penalty'] is None
            or cf_dic['p_diff_penalty'] < 0):  # Default
        if cf_dic['mtot'] == 4:
            cf_dic['p_diff_penalty'] = 0.5
        else:                                   # Approx 1 for N = 1000
            cf_dic['p_diff_penalty'] = (
                2 * ((n_train * cf_dic['subsample_share_forest'])**0.9)
                / (n_train * cf_dic['subsample_share_forest']))
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
        cf_dic['estimator_str'] += ' penalty fct'
    # Minimum leaf size
    if cf_dic['n_min_min'] is None or cf_dic['n_min_min'] < 1:
        cf_dic['n_min_min'] = (max(round((n_d_subsam**0.4) / 6), 2)
                               * gen_dic['no_of_treat'])
    else:
        cf_dic['n_min_min'] = round(cf_dic['n_min_min'])
    if cf_dic['n_min_max'] is None or cf_dic['n_min_max'] < 1:
        cf_dic['n_min_max'] = (max(round(n_d_subsam**0.5 / 6), 3)
                               * gen_dic['no_of_treat'])
    else:
        cf_dic['n_min_max'] = round(cf_dic['n_min_max'])
    cf_dic['n_min_max'] = max(cf_dic['n_min_min'], cf_dic['n_min_max'])
    if gen_dic['d_type'] == 'discrete':
        if cf_dic['n_min_treat'] is None or cf_dic['n_min_treat'] < 1:
            cf_dic['n_min_treat'] = max(
                round((cf_dic['n_min_min'] + cf_dic['n_min_max']) / 2
                      / gen_dic['no_of_treat'] / 4), 2)
        else:
            cf_dic['n_min_treat'] = round(cf_dic['n_min_treat'])
        min_leaf_size = cf_dic['n_min_treat'] * gen_dic['no_of_treat']
        if cf_dic['n_min_min'] < min_leaf_size:
            cf_dic['n_min_min'] = min_leaf_size
            ps.print_mcf(gen_dic, 'Minimum leaf size adjusted. Smallest ',
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
    if not isinstance(cf_dic['chunks_maxsize'], int):
        base_level = 60000
        cf_dic['chunks_maxsize'] = round(max(
            base_level + np.sqrt(max(n_train - base_level, 0)), base_level))
    cf_dic['forests'] = None              # To be filled at the end of training
    mcf_.cf_dict = cf_dic
    mcf_.gen_dict = gen_dic


def p_init(gen_dic, amgate=None, atet=None, bgate=None,
           bt_yes=None, choice_based_sampling=None, choice_based_probs=None,
           ci_level=None, cluster_std=None, cond_var=None,
           gates_minus_previous=None, gates_smooth=None,
           gates_smooth_bandwidth=None, gates_smooth_no_evalu_points=None,
           gatet=None, gmate_no_evalu_points=None,  gmate_sample_share=None,
           iate=None, iate_se=None, iate_m_ate=None, knn=None, knn_const=None,
           knn_min_k=None, nw_bandw=None, nw_kern=None, max_cats_z_vars=None,
           max_weight_share=None, se_boot_ate=None, se_boot_gate=None,
           se_boot_iate=None):
    """Initialise dictionary with parameters of parameter prediction."""
    atet, gatet = atet is True, gatet is True
    if gatet:
        atet = True
    amgate, bgate, bt_yes = amgate is True,  bgate is True, bt_yes is True
    if choice_based_sampling is True:
        if gen_dic['d_type'] != 'discrete':
            raise NotImplementedError('No choice based sample with continuous'
                                      ' treatments.')
    else:
        choice_based_sampling, choice_based_probs = False, 1
    if ci_level is None or not 0.5 < ci_level < 0.99999999:
        ci_level = 0.90
    cluster_std = (cluster_std is True) or gen_dic['panel_data']
    cond_var = cond_var is not False
    gates_minus_previous = gates_minus_previous is True
    # gates_smooth: smooth_gates, gates_smooth_bandwidth: sgates_bandwidth
    # gates_smooth_no_evalu_points: sgates_no_evaluation_points
    gates_smooth = gates_smooth is not False
    if gates_smooth_bandwidth is None or gates_smooth_bandwidth <= 0:
        gates_smooth_bandwidth = 1
    if (gates_smooth_no_evalu_points is None
            or gates_smooth_no_evalu_points < 2):
        gates_smooth_no_evalu_points = 50
    else:
        gates_smooth_no_evalu_points = round(gates_smooth_no_evalu_points)
    if gmate_no_evalu_points is None or gmate_no_evalu_points < 2:
        gmate_no_evalu_points = 50
    else:
        gmate_no_evalu_points = round(gmate_no_evalu_points)
    iate = iate is not False
    iate_se = iate_se is True
    iate_m_ate = iate_m_ate is True
    if iate is False:
        iate_se = iate_m_ate = False
    knn = knn is not False
    if knn_min_k is None or knn_min_k < 0:
        knn_min_k = 10                   # minimum number of neighbours in k-NN
    if knn_const is None or knn_const < 0:
        knn_const = 1                     # k: const. in # of neighbour estimat
    if nw_bandw is None or nw_bandw < 0:  # multiplier
        nw_bandw = 1                      # times Silverman's optimal bandwidth
    if nw_kern is None or nw_kern != 2:   # kernel for NW:
        nw_kern = 1                       # 1: Epanechikov 2: Normal
    se_boot_ate = bootstrap(se_boot_ate, 49, 199, cluster_std)
    se_boot_gate = bootstrap(se_boot_gate, 49, 199, cluster_std)
    se_boot_iate = bootstrap(se_boot_iate, 49, 199, cluster_std)
    if max_weight_share is None or max_weight_share <= 0:
        max_weight_share = 0.05
    q_w = [0.5, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    # Assign variables to dictionary
    dic = {
        'amgate': amgate, 'atet': atet, 'bgate': bgate, 'bt_yes': bt_yes,
        'ci_level': ci_level, 'choice_based_sampling': choice_based_sampling,
        'choice_based_probs': choice_based_probs, 'cluster_std': cluster_std,
        'cond_var': cond_var, 'gmate_sample_share': gmate_sample_share,
        'gates_minus_previous': gates_minus_previous,
        'gates_smooth': gates_smooth,
        'gates_smooth_bandwidth': gates_smooth_bandwidth,
        'gates_smooth_no_evalu_points': gates_smooth_no_evalu_points,
        'gatet': gatet, 'gmate_no_evalu_points': gmate_no_evalu_points,
        'iate':  iate, 'iate_se': iate_se, 'iate_m_ate': iate_m_ate,
        'knn': knn, 'knn_const': knn_const,
        'knn_min_k': knn_min_k, 'nw_bandw': nw_bandw, 'nw_kern': nw_kern,
        'max_cats_z_vars': max_cats_z_vars,
        'max_weight_share': max_weight_share, 'se_boot_ate': se_boot_ate,
        'se_boot_gate': se_boot_gate, 'se_boot_iate': se_boot_iate, 'q_w': q_w}
    # Define paths to save figures for plots of effects
    if gen_dic['with_output']:
        dic = mcf_sys.get_fig_path(dic, gen_dic['outpath'], 'ate_iate',
                                   gen_dic['with_output'])
        dic = mcf_sys.get_fig_path(dic, gen_dic['outpath'], 'gate',
                                   gen_dic['with_output'])
        if amgate:
            dic = mcf_sys.get_fig_path(dic, gen_dic['outpath'], 'amgate',
                                       gen_dic['with_output'])
        if bgate:
            dic = mcf_sys.get_fig_path(dic, gen_dic['outpath'], 'bgate',
                                       gen_dic['with_output'])
    return dic


def p_update_train(mcf_, data_df):
    """Update parameters with data."""
    p_dic = mcf_.p_dict
    n_train = len(data_df)
    # Categorise continuous gate variables
    if p_dic['max_cats_z_vars'] is None or p_dic['max_cats_z_vars'] < 1:
        p_dic['max_cats_z_vars'] = round(n_train ** 0.3)
    else:
        p_dic['max_cats_z_vars'] = round(p_dic['max_cats_z_vars'])
    mcf_.p_dict = p_dic


def p_update_pred(mcf_, data_df):
    """Update parameters of p_dict with data_df related information."""
    gen_dic, p_dic, var_dic = mcf_.gen_dict, mcf_.p_dict, mcf_.var_dict
    n_pred = len(data_df)
    if p_dic['gmate_sample_share'] is None or p_dic['gmate_sample_share'] <= 0:
        p_dic['gmate_sample_share'] = (
            1 if n_pred < 1000 else (1000 + ((n_pred-1000) ** 0.75)) / n_pred)
    d_name = (var_dic['d_name'][0]
              if isinstance(var_dic['d_name'], (list, tuple))
              else var_dic['d_name'])
    # Capitalise all variable names
    data_df = data_df.rename(columns=lambda x: x.upper())
    # Check if treatment is included
    if gen_dic['d_type'] == 'continuous':
        p_dic['d_in_pred'] = False
    else:
        p_dic['d_in_pred'] = d_name in data_df.columns
    if not p_dic['d_in_pred']:
        if p_dic['atet'] or p_dic['gatet']:
            ps.print_mcf(gen_dic, 'Treatment variable not in prediction data.'
                         ' ATET and GATET cannot be computed.', summary=True)

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


def post_init(p_dic, bin_corr_threshold=None, bin_corr_yes=None,
              est_stats=None, kmeans_no_of_groups=None, kmeans_max_tries=None,
              kmeans_replications=None, kmeans_yes=None, random_forest_vi=None,
              relative_to_first_group_only=None, plots=None):
    """Initialise dictionary with parameters of post estimation analysis."""
    est_stats = est_stats is not False
    if not p_dic['iate']:
        est_stats = False
    bin_corr_yes = bin_corr_yes is not False
    if bin_corr_threshold is None or not 0 <= bin_corr_threshold <= 1:
        bin_corr_threshold = 0.1  # Minimum threshhold of abs.
    plots = plots is not False
    relative_to_first_group_only = relative_to_first_group_only is not False
    kmeans_yes = kmeans_yes is not False
    if kmeans_replications is None or kmeans_replications < 0:
        kmeans_replications = 10
    else:
        kmeans_replications = round(kmeans_replications)
    if kmeans_max_tries is None:
        kmeans_max_tries = 1000
    kmeans_max_tries = max(kmeans_max_tries, 10)
    add_pred_to_data_file = est_stats
    random_forest_vi = random_forest_vi is not False
    # Put everything in the dictionary
    dic = {
        'bin_corr_threshold': bin_corr_threshold, 'bin_corr_yes': bin_corr_yes,
        'est_stats': est_stats, 'kmeans_no_of_groups': kmeans_no_of_groups,
        'kmeans_max_tries': kmeans_max_tries,
        'kmeans_replications': kmeans_replications, 'kmeans_yes': kmeans_yes,
        'random_forest_vi': random_forest_vi, 'plots': plots,
        'relative_to_first_group_only': relative_to_first_group_only,
        'add_pred_to_data_file': add_pred_to_data_file}
    return dic


def post_update_pred(mcf_, data_df):
    """Update entries in post_dic that need info from prediction data."""
    n_pred = len(data_df)
    post_dic = mcf_.post_dict
    if isinstance(post_dic['kmeans_no_of_groups'], (int, float)):
        post_dic['kmeans_no_of_groups'] = [post_dic['kmeans_no_of_groups']]
    if (post_dic['kmeans_no_of_groups'] is None
            or len(post_dic['kmeans_no_of_groups']) == 1):
        if (post_dic['kmeans_no_of_groups'] is None is None
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
            post_dic['kmeans_no_of_groups'] = [round(
                post_dic['kmeans_no_of_groups'])]
    else:
        if not isinstance(post_dic['kmeans_no_of_groups'], list):
            post_dic['kmeans_no_of_groups'] = list(
                post_dic['kmeans_no_of_groups'])
            post_dic['kmeans_no_of_groups'] = [
                round(a) for a in post_dic['kmeans_no_of_groups']]
    mcf_.post_dict = post_dic


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


def ray_shut_down(_mp_ray_shutdown, len_data):
    """Define mimimum sample size for ray_shut_down."""
    if _mp_ray_shutdown is None:
        _mp_ray_shutdown = not len_data < 100000
    if not isinstance(_mp_ray_shutdown, bool):
        raise ValueError('mp_ray_shutdown must be either None or Boolean')
    return _mp_ray_shutdown


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
    grid_name_l = [grid_name_tmp.upper()]
    return grid_name_l


def sub_size(n_train, share_mult, max_share):
    """Help for initialisation."""
    if share_mult is None or share_mult <= 0:
        share_mult = 1
    subsam_share = min(4 * ((n_train / 2)**0.85) / n_train, 0.67) * share_mult
    subsam_share = max(min(subsam_share, max_share), 1e-4)
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
