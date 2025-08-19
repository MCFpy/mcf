"""
Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for initialising the parameters of the programme.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy

from psutil import virtual_memory, cpu_count
import numpy as np
from torch.cuda import is_available

from mcf import mcf_general as mcf_gp
from mcf import mcf_init_update_helper_functions as mcf_init_update
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_general_sys as mcf_sys


def blind_init(var_x_protected_name=None, var_x_policy_name=None,
               var_x_unrestricted_name=None, weights_of_blind=None,
               obs_ref_data=None, seed=None):
    """Initialise dictionary with parameters for blind_iate methods."""
    dic = {}
    dic['var_x_protected_name'] = var_x_protected_name
    dic['var_x_policy_name'] = var_x_policy_name
    dic['var_x_unrestricted_name'] = var_x_unrestricted_name

    if (obs_ref_data is None or not isinstance(obs_ref_data, (float, int))
            or obs_ref_data < 1):
        dic['obs_ref_data'] = 100
    else:
        dic['obs_ref_data'] = int(obs_ref_data)

    if weights_of_blind is None or not isinstance(weights_of_blind,
                                                  (tuple, list, float, int)):
        dic['weights_of_blind'] = (0, 1,)
    else:
        if isinstance(weights_of_blind, (float, int)):
            if 0 > weights_of_blind > 1:
                dic['weights_of_blind'] = (0, weights_of_blind, 1)
            else:
                dic['weights_of_blind'] = (weights_of_blind,)
        else:
            if isinstance(weights_of_blind, tuple):
                weights_of_blind = list(weights_of_blind)
            dic['weights_of_blind'] = weights_of_blind
            if min(dic['weights_of_blind']) > 0:
                dic['weights_of_blind'] = [0] + dic['weights_of_blind']
            if max(dic['weights_of_blind']) < 1:
                dic['weights_of_blind'] = dic['weights_of_blind'] + [1]
        if not all(0 <= elem <= 1 for elem in dic['weights_of_blind']):
            raise ValueError(f'{dic["weights_of_blind"]} must be nonnegative'
                             ' and not larger than 1')
    if seed is None or not isinstance(seed, (float, int)) or seed < 1:
        dic['seed'] = 123456
    else:
        dic['seed'] = int(seed)
    return dic


def int_init(cuda=None, cython=None,
             del_forest=None, descriptive_stats=None, dpi=None,
             fontsize=None, keep_w0=None, no_filled_plot=None,
             max_save_values=None, max_cats_cont_vars=None, mp_ray_del=None,
             mp_ray_objstore_multiplier=None, mp_ray_shutdown=None,
             mp_vim_type=None, mp_weights_tree_batch=None, mp_weights_type=None,
             obs_bigdata=None,
             output_no_new_dir=None,
             return_iate_sp=None, replication=None, report=None,
             seed_sample_split=None, share_forest_sample=None, show_plots=None,
             verbose=None,
             weight_as_sparse=None, weight_as_sparse_splits=None,
             with_output=None, iate_chunk_size=None,
             max_obs_training=None, max_obs_prediction=None,
             max_obs_kmeans=None, max_obs_post_rel_graphs=None,
             p_ate_no_se_only=None
             ):
    """Initialise dictionary of parameters of internal variables."""
    dic = {}

    dic['iv'] = False  # Will be changed by the train_iv method (if used)

    dic['cuda'] = False if cuda is not True else is_available()
    dic['cython'] = cython is not False

    dic['del_forest'] = del_forest is True
    dic['keep_w0'] = keep_w0 is True
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

    dic['mp_ray_del'] = mcf_init_update.get_ray_del_defaults(mp_ray_del)
    if mp_ray_objstore_multiplier is None or mp_ray_objstore_multiplier < 0:
        dic['mp_ray_objstore_multiplier'] = 1
    else:
        dic['mp_ray_objstore_multiplier'] = mp_ray_objstore_multiplier
    dic['no_ray_in_forest_building'] = False

    if mp_weights_tree_batch is False:
        mp_weights_tree_batch = 1
    if mp_weights_tree_batch is not None and mp_weights_tree_batch > 0.5:
        dic['mp_weights_tree_batch'] = round(mp_weights_tree_batch)
    else:
        dic['mp_weights_tree_batch'] = 0

    if share_forest_sample is None or not 0.01 < share_forest_sample < 0.99:
        dic['share_forest_sample'] = 0.5
    else:
        dic['share_forest_sample'] = share_forest_sample

    dic['weight_as_sparse'] = weight_as_sparse is not False

    dic['show_plots'] = show_plots is not False
    dic['report'] = report is not False
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
    dic['smaller_sample'] = None
    if not isinstance(weight_as_sparse_splits, int):  # Updated in weights_mp
        dic['weight_as_sparse_splits'] = None
    else:
        dic['weight_as_sparse_splits'] = weight_as_sparse_splits
    # Variables varying with data. To be initialised when training.
    dic['mp_ray_shutdown'] = mp_ray_shutdown
    dic['mp_vim_type'] = mp_vim_type
    dic['max_cats_cont_vars'] = max_cats_cont_vars
    if p_ate_no_se_only is not None and p_ate_no_se_only:
        dic['return_iate_sp'] = False
    dic['output_no_new_dir'] = output_no_new_dir is True
    dic['replication'] = replication is True
    if isinstance(iate_chunk_size, (int, float)) and iate_chunk_size > 0:
        dic['iate_chunk_size'] = iate_chunk_size
    else:
        dic['iate_chunk_size'] = None

    if isinstance(obs_bigdata, (int, float)) and obs_bigdata > 10:
        dic['obs_bigdata'] = obs_bigdata
    else:
        dic['obs_bigdata'] = 1000000

    if isinstance(max_obs_training, (int, float)) and max_obs_training > 100:
        dic['max_obs_training'] = max_obs_training
    else:
        dic['max_obs_training'] = float('inf')
    if (isinstance(max_obs_prediction, (int, float))
            and max_obs_prediction > 100):
        dic['max_obs_prediction'] = max_obs_prediction
    else:
        dic['max_obs_prediction'] = 250000
    if isinstance(max_obs_kmeans, (int, float)) and max_obs_kmeans > 100:
        dic['max_obs_post_kmeans'] = max_obs_kmeans
    else:
        dic['max_obs_post_kmeans'] = 200000
    if (isinstance(max_obs_post_rel_graphs, (int, float))
            and (max_obs_post_rel_graphs > 100)):
        dic['max_obs_post_rel_graphs'] = max_obs_post_rel_graphs
    else:
        dic['max_obs_post_rel_graphs'] = 50000

    return dic


def gen_init(int_dic, d_type=None, iate_eff=None, mp_parallel=None,
             outfiletext=None, outpath=None, output_type=None,
             weighted=None, panel_data=None, panel_in_rf=None):
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
        dic['mp_parallel'] = round(cpu_count(logical=True)*0.8)
        dic['mp_automatic'] = True
    elif mp_parallel <= 1.5:
        dic['mp_parallel'] = 1
        dic['mp_automatic'] = False
    else:
        dic['mp_parallel'] = round(mp_parallel)
        dic['mp_automatic'] = False
    dic['sys_share'] = 0.7 * getattr(virtual_memory(), 'percent') / 100
    # Define or create directory for output and avoid overwritting
    if int_dic['with_output']:
        dic['outpath'] = mcf_sys.define_outpath(
            outpath, not int_dic['output_no_new_dir'])
        # Files to write output to
        dic['outfiletext'] = ('txtFileWithOutput'
                              if outfiletext is None else outfiletext)
        dic['outfiletext'] = dic['outpath'] / (dic['outfiletext'] + '.txt')
        dic['outfilesummary'] = dic['outfiletext'].with_name(
            f'{dic["outfiletext"].stem}_Summary.txt')
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


def var_init(gen_dic, fs_dic, p_dic, x_name_balance_bgate=None,
             cluster_name=None, d_name=None, id_name=None, w_name=None,
             iv_name=None, x_name_balance_test_ord=None,
             x_name_balance_test_unord=None, x_name_always_in_ord=None,
             x_name_always_in_unord=None, x_name_remain_ord=None,
             x_name_remain_unord=None, x_name_ord=None,
             x_name_unord=None, y_name=None, y_tree_name=None,
             z_name_cont=None, z_name_ord=None, z_name_unord=None):
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
        cluster_name = mcf_gp.cleaned_var_names(cluster_name)
    else:
        cluster_name = []
    if p_dic['ate_no_se_only'] is True:
        x_name_balance_test_ord = x_name_balance_test_unord = None
        z_name_cont = z_name_ord = z_name_unord = x_name_balance_bgate = None
    # Clean variable names ... set all to capital letters
    d_name = mcf_gp.cleaned_var_names(mcf_gp.to_list_if_needed(d_name))
    y_tree_name = mcf_gp.cleaned_var_names(
        mcf_gp.to_list_if_needed(y_tree_name))
    y_name = mcf_gp.to_list_if_needed(y_name)
    y_name.extend(y_tree_name)
    y_name = mcf_gp.cleaned_var_names(y_name)
    iv_name = mcf_gp.cleaned_var_names(iv_name)
    x_name_always_in_ord = mcf_gp.cleaned_var_names(x_name_always_in_ord)
    x_name_always_in_unord = mcf_gp.cleaned_var_names(x_name_always_in_unord)
    x_name_remain_ord = mcf_gp.cleaned_var_names(x_name_remain_ord)
    x_name_remain_unord = mcf_gp.cleaned_var_names(x_name_remain_unord)
    x_name_balance_test_ord = mcf_gp.cleaned_var_names(x_name_balance_test_ord)
    x_name_balance_test_unord = mcf_gp.cleaned_var_names(
        x_name_balance_test_unord)
    z_name_cont = mcf_gp.cleaned_var_names(z_name_cont)
    z_name_ord = mcf_gp.cleaned_var_names(z_name_ord)
    z_name_unord = mcf_gp.cleaned_var_names(z_name_unord)
    x_name_balance_bgate = (mcf_gp.cleaned_var_names(x_name_balance_bgate)
                            if p_dic['bgate'] else [])
    x_name_ord = mcf_gp.cleaned_var_names(x_name_ord)
    if z_name_cont or z_name_ord:
        x_name_ord += z_name_cont + z_name_ord
    x_name_unord = mcf_gp.cleaned_var_names(x_name_unord)
    if z_name_cont or z_name_ord:
        x_name_unord += z_name_unord
    if gen_dic['weighted']:      # Former w_yes
        if w_name is None or w_name == []:
            raise ValueError('No name for sample weights specified.')
        w_name = mcf_gp.cleaned_var_names(w_name)
    else:
        w_name = []
    if not isinstance(id_name, (list, tuple)):
        id_name = [id_name]
    if not isinstance(id_name[0], str):
        id_name[0] = 'ID'

    if not isinstance(iv_name, (list, tuple)):
        iv_name = [iv_name]
    id_name = mcf_gp.cleaned_var_names(id_name)
    x_name = deepcopy(x_name_ord + x_name_unord)
    x_name = mcf_gp.cleaned_var_names(x_name)
    x_name_in_tree = deepcopy(x_name_always_in_ord + x_name_always_in_unord)
    x_name_in_tree = mcf_gp.cleaned_var_names(x_name_in_tree)
    x_name_balance_test = mcf_gp.cleaned_var_names(
        deepcopy(x_name_balance_test_ord + x_name_balance_test_unord))
    if not x_name_balance_test:
        p_dic['bt_yes'] = False
    x_name_remain = mcf_gp.cleaned_var_names(
        deepcopy(x_name_remain_ord + x_name_remain_unord + x_name_in_tree
                 + x_name_balance_test))
    x_name_always_in = mcf_gp.cleaned_var_names(
        deepcopy(x_name_always_in_ord + x_name_always_in_unord))
    name_ordered = mcf_gp.cleaned_var_names(
        deepcopy(x_name_ord + x_name_always_in_ord + x_name_remain_ord))
    name_unordered = mcf_gp.cleaned_var_names(
        deepcopy(x_name_unord + x_name_always_in_unord + x_name_remain_unord))
    if fs_dic['yes']:
        if p_dic['bt_yes']:
            x_name_remain = mcf_gp.cleaned_var_names(
                deepcopy(x_name_balance_test + x_name_remain))
    if x_name_in_tree:
        x_name_remain = mcf_gp.cleaned_var_names(
            deepcopy(x_name_in_tree + x_name_remain))
        x_name = mcf_gp.cleaned_var_names(deepcopy(x_name_in_tree + x_name))
    if not ((not name_ordered) or (not name_unordered)):
        if any(value for value in name_ordered if value in name_unordered):
            raise ValueError('Remove overlap in ordered + unordered variables')
    # Make x_name is unique
    x_name = mcf_init_update.name_unique(x_name)
    names_to_check_train = d_name + y_name + x_name
    names_to_check_pred = x_name[:]
    if (not z_name_cont) and (not z_name_ord) and (not z_name_unord):
        gen_dic['agg_yes'] = p_dic['gate'] = False
        z_name = []
    else:
        gen_dic['agg_yes'] = p_dic['gate'] = True
        if z_name_cont:
            names_to_check_train.extend(z_name_cont)
            names_to_check_pred.extend(z_name_cont)
        if z_name_ord:
            names_to_check_train.extend(z_name_ord)
            names_to_check_pred.extend(z_name_ord)
        if z_name_unord:
            names_to_check_train.extend(z_name_unord)
            names_to_check_pred.extend(z_name_unord)
        z_name = z_name_cont + z_name_ord + z_name_unord
    txt = '\n'
    if p_dic['bgate'] and not p_dic['gate']:
        txt += 'BGATEs can only be computed if GATEs are computed.'
        p_dic['bgate'] = False
    if p_dic['cbgate'] and not p_dic['gate']:
        txt += 'CBGATEs can only be computed if GATEs are computed.'
        p_dic['cbgate'] = False
    if p_dic['gatet'] and not p_dic['gate']:
        txt += 'GATETs can only be computed if GATEs are computed.'
        p_dic['gatet'] = False
    mcf_ps.print_mcf(gen_dic, txt, summary=True)
    if p_dic['bgate'] and x_name_balance_bgate == z_name and len(z_name) == 1:
        p_dic['bgate'] = False
    if p_dic['bgate']:
        if (x_name_balance_bgate is None or x_name_balance_bgate == []):
            if len(z_name) > 1:
                x_name_balance_bgate = z_name[:]
            else:
                p_dic['bgate'], x_name_balance_bgate = False, []
        else:
            names_to_check_train.extend(x_name_balance_bgate)
            names_to_check_pred.extend(x_name_balance_bgate)
    else:
        x_name_balance_bgate = []
    if x_name_balance_bgate == [] and len(z_name) == 1:
        p_dic['bgate'] = False
    names_to_check_train = mcf_init_update.name_unique(names_to_check_train[:])
    names_to_check_pred = mcf_init_update.name_unique(names_to_check_pred[:])
    dic = {
        'x_name_balance_bgate': x_name_balance_bgate,
        'cluster_name': cluster_name,
        'd_name': d_name, 'id_name': id_name, 'name_unordered': name_unordered,
        'names_to_check_train': names_to_check_train,
        'names_to_check_pred': names_to_check_pred,
        'w_name': w_name, 'x_name_balance_test_ord': x_name_balance_test_ord,
        'x_name_balance_test_unord': x_name_balance_test_unord,
        'x_name_always_in_ord': x_name_always_in_ord,
        'x_name_always_in_unord': x_name_always_in_unord,
        'x_name_remain_ord': x_name_remain_ord,
        'x_name_remain_unord': x_name_remain_unord,
        'x_name_ord': x_name_ord, 'x_name_unord': x_name_unord,
        'y_name': y_name, 'y_tree_name': y_tree_name,
        'x_name_balance_test': x_name_balance_test,
        'x_name_always_in': x_name_always_in,
        'name_ordered': name_ordered, 'x_name_remain': x_name_remain,
        'x_name': x_name, 'x_name_in_tree': x_name_in_tree, 'z_name': z_name,
        'z_name_cont': z_name_cont, 'z_name_ord': z_name_ord,
        'z_name_unord': z_name_unord,
        'iv_name': iv_name,
        }
    return dic, gen_dic, p_dic


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
            adjust_limits=None, detect_const_vars_stop=None):
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
                                   gen_dic['with_output'], no_csv=False)
    dic['cut_offs'], dic['forests'] = None, None

    if isinstance(detect_const_vars_stop, bool):
        dic['detect_const_vars_stop'] = detect_const_vars_stop
    else:
        dic['detect_const_vars_stop'] = True

    return dic


def lc_init(cs_cv=None, cs_cv_k=None, cs_share=None, undo_iate=None, yes=None,
            estimator=None):
    """Initialise dictionary with parameters of local centering, CS and CV."""
    # General parameters for crossvalidation or using a new sample
    dic = {}
    dic['cs_cv'] = cs_cv is not False
    dic['cs_share'] = (0.25 if cs_share is None
                       or not (0.0999 < cs_share < 0.9001) else cs_share)
    if not isinstance(cs_cv_k, (int, float)) or cs_cv_k < 1:
        dic['cs_cv_k'] = None    # To be set when size of training data is known
    else:
        dic['cs_cv_k'] = round(cs_cv_k)
    # local centering
    # dic['cs_cv'] corresponds to cnew_dict['l_centering_new_sample']
    dic['yes'] = yes is not False
    ok_estimators = ('RandomForest', 'RandomForestNminl5',
                     'RandomForestNminls5',
                     'SupportVectorMachine', 'SupportVectorMachineC2',
                     'SupportVectorMachineC4',
                     'AdaBoost', 'AdaBoost100', 'AdaBoost200',
                     'GradBoost', 'GradBoostDepth6',  'GradBoostDepth12',
                     'LASSO',
                     'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger',
                     'Mean', 'automatic')
    if estimator is None:
        dic['estimator'] = 'RandomForest'
    else:
        ok_estimators_cfold = [name.casefold() for name in ok_estimators]
        try:
            position = ok_estimators_cfold.index(estimator.casefold())
            dic['estimator'] = ok_estimators[position]
        except ValueError:
            print('Estimator specified for local centering is not '
                  f'valid.\nSpecified estimator {estimator}.'
                  f'\nAllowed estimators: {" ".join(ok_estimators)}')
    dic['uncenter_po'] = undo_iate is not False
    if not dic['yes']:
        dic['uncenter_po'] = False
    # if fs_dic['yes']:   # This is most likely from the old version where fs
    #     dic['uncenter_po'] = False  # was conducted after local centering
    return dic


def cf_init(alpha_reg_grid=None, alpha_reg_max=None, alpha_reg_min=None,
            boot=None, chunks_maxsize=None, compare_only_to_zero=None,
            nn_main_diag_only=None, m_grid=None, m_share_max=None,
            m_share_min=None, m_random_poisson=None, match_nn_prog_score=None,
            mce_vart=None, vi_oob_yes=None, n_min_grid=None, n_min_max=None,
            n_min_min=None, n_min_treat=None, p_diff_penalty=None,
            penalty_type=None, subsample_factor_eval=None,
            subsample_factor_forest=None, tune_all=None,
            random_thresholds=None):
    """Initialise dictionary with parameters of causal forest building."""
    dic = {}

    dic['boot'] = 1000 if boot is None or boot < 1 else round(boot)
    dic['match_nn_prog_score'] = match_nn_prog_score is not False
    dic['nn_main_diag_only'] = nn_main_diag_only is True
    dic['compare_only_to_zero'] = compare_only_to_zero is True

    dic['m_grid'] = 1 if m_grid is None or m_grid < 1 else round(m_grid)
    dic['n_min_grid'] = 1 if (n_min_grid is None or n_min_grid < 1
                              ) else round(n_min_grid)
    alpha_reg_grid_check = 1 if (alpha_reg_grid is None or alpha_reg_grid < 1
                                 ) else round(alpha_reg_grid)

    dic['tune_all'] = tune_all is True
    if dic['tune_all']:
        no_of_values = 3
        dic['m_grid'] = max(dic['m_grid'], no_of_values)
        dic['n_min_grid'] = max(dic['n_min_grid'], no_of_values)
        alpha_reg_grid_check = max(alpha_reg_grid_check, no_of_values)

    (dic['alpha_reg_grid'], dic['alpha_reg_max'], dic['alpha_reg_min'],
     dic['alpha_reg_values']) = mcf_init_update.get_alpha(alpha_reg_grid_check,
                                                          alpha_reg_max,
                                                          alpha_reg_min
                                                          )
    # Select grid for number of parameters
    if m_share_min is None or not 0 < m_share_min <= 1:
        dic['m_share_min'] = 0.1
    else:
        dic['m_share_min'] = m_share_min
    if m_share_max is None or not 0 < m_share_max <= 1:
        dic['m_share_max'] = 0.6
    else:
        dic['m_share_max'] = m_share_max
    if m_random_poisson is False:
        dic['m_random_poisson'] = False
        dic['m_random_poisson_min'] = 1000000
    else:
        dic['m_random_poisson'] = True
        dic['m_random_poisson_min'] = 10

    if mce_vart is None or mce_vart == 1:
        mtot, mtot_no_mce, estimator_str = 1, 0, 'MSE & MCE'    # MSE + MCE
    elif mce_vart == 2:                  # -Var(treatment effect)
        mtot, mtot_no_mce, estimator_str = 2, 1, '-Var(effect)'
    elif mce_vart == 0:                             # MSE rule
        mtot, mtot_no_mce, estimator_str = 3, 1, 'MSE'
    elif mce_vart == 3:  # MSE+MCE rule or penalty function rule
        mtot, mtot_no_mce = 4, 0                      # (randomly decided)
        estimator_str = 'MSE, MCE or penalty (random)'
    else:
        raise ValueError('Inconsistent MTOT definition of  MCE_VarT.')

    if penalty_type is None or penalty_type != 'diff_d':
        dic['penalty_type'] = 'mse_d'
    else:
        dic['penalty_type'] = penalty_type

    # These values will be updated later
    dic['mtot'], dic['mtot_no_mce'] = mtot, mtot_no_mce
    dic['estimator_str'] = estimator_str
    dic['chunks_maxsize'] = chunks_maxsize
    dic['vi_oob_yes'] = vi_oob_yes is True
    dic['n_min_max'] = n_min_max
    dic['n_min_min'], dic['n_min_treat'] = n_min_min, n_min_treat
    dic['p_diff_penalty'] = p_diff_penalty
    dic['subsample_factor_eval'] = subsample_factor_eval
    dic['subsample_factor_forest'] = subsample_factor_forest
    dic['random_thresholds'] = random_thresholds
    return dic


def p_init(gen_dic, ate_no_se_only=None, cbgate=None, atet=None, bgate=None,
           bt_yes=None, choice_based_sampling=None, choice_based_probs=None,
           ci_level=None, cluster_std=None, cond_var=None,
           gates_minus_previous=None, gates_smooth=None,
           gates_smooth_bandwidth=None, gates_smooth_no_evalu_points=None,
           gatet=None, gate_no_evalu_points=None,  bgate_sample_share=None,
           iate=None, iate_se=None, iate_m_ate=None, knn=None, knn_const=None,
           knn_min_k=None, nw_bandw=None, nw_kern=None, max_cats_z_vars=None,
           max_weight_share=None, se_boot_ate=None, se_boot_gate=None,
           se_boot_iate=None, qiate=None, qiate_se=None, qiate_m_mqiate=None,
           qiate_m_opp=None, qiate_no_of_quantiles=None, se_boot_qiate=None,
           qiate_smooth=None, qiate_smooth_bandwidth=None,
           qiate_bias_adjust=None,
           iv_aggregation_method=None):
    """Initialise dictionary with parameters of effect predictions."""
    atet, gatet = atet is True, gatet is True
    ate_no_se_only = ate_no_se_only is True
    if ate_no_se_only:
        atet = gatet = cbgate = bgate = bt_yes = cluster_std = False
        gates_smooth = iate = iate_se = iate_m_ate = se_boot_ate = False
        qiate = qiate_se = qiate_m_mqiate = qiate_m_opp = False
        se_boot_gate = se_boot_iate = se_boot_qiate = False
    if gatet:
        atet = True
    if qiate:
        iate = True
    cbgate, bgate, bt_yes = cbgate is True,  bgate is True, bt_yes is True
    if choice_based_sampling is True:
        if gen_dic['d_type'] != 'discrete':
            raise NotImplementedError('No choice based sample with continuous'
                                      ' treatments.')
    else:
        choice_based_sampling, choice_based_probs = False, 1
    if ci_level is None or not 0.5 < ci_level < 0.99999999:
        ci_level = 0.95
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
    if gate_no_evalu_points is None or gate_no_evalu_points < 2:
        gate_no_evalu_points = 50
    else:
        gate_no_evalu_points = round(gate_no_evalu_points)
    iate = iate is not False
    iate_se = iate_se is True
    iate_m_ate = iate_m_ate is True
    if iate is False:
        iate_se = iate_m_ate = False

    qiate = qiate is True
    qiate_se = qiate_se is True
    qiate_m_mqiate = qiate_m_mqiate is True
    qiate_m_opp = qiate_m_opp is True
    if qiate is False:
        qiate_se = qiate_m_mqiate = qiate_m_opp = False
    if (qiate_no_of_quantiles is None
        or not isinstance(qiate_no_of_quantiles, int)
            or qiate_no_of_quantiles < 10):
        qiate_no_of_quantiles = 99
    qiate_quantiles = (np.arange(qiate_no_of_quantiles) / qiate_no_of_quantiles
                       + 0.5 / qiate_no_of_quantiles)
    qiate_smooth = qiate_smooth is not False
    if qiate_smooth_bandwidth is None or qiate_smooth_bandwidth <= 0:
        qiate_smooth_bandwidth = 1

    qiate_bias_adjust = qiate_bias_adjust is not False
    if qiate_bias_adjust or qiate_se:
        iate_se = True
    # if not (isinstance(qiate_bias_adjust_draws, (int, float))
    #         and qiate_bias_adjust_draws > 0):
    qiate_bias_adjust_draws = 1  # Current bias adjustment does not need draws

    knn = knn is not False
    if knn_min_k is None or knn_min_k < 0:
        knn_min_k = 10                    # minimum number of neighbours in k-NN
    if knn_const is None or knn_const < 0:
        knn_const = 1                     # k: const. in # of neighbour estimat
    if nw_bandw is None or nw_bandw < 0:  # multiplier
        nw_bandw = 1                      # times Silverman's optimal bandwidth
    if nw_kern is None or nw_kern != 2:   # kernel for NW:
        nw_kern = 1                       # 1: Epanechikov 2: Normal
    se_boot_ate = mcf_init_update.bootstrap(se_boot_ate, 49, 199, cluster_std)
    se_boot_gate = mcf_init_update.bootstrap(se_boot_gate, 49, 199, cluster_std)
    se_boot_iate = mcf_init_update.bootstrap(se_boot_iate, 49, 199, cluster_std)
    se_boot_qiate = mcf_init_update.bootstrap(se_boot_qiate, 49, 199,
                                              cluster_std
                                              )
    if max_weight_share is None or max_weight_share <= 0:
        max_weight_share = 0.05
    q_w = [0.5, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    # Assign variables to dictionary

    valid_iv_methods = ('local', 'global',)
    default_method = ('local', 'global',)
    if isinstance(iv_aggregation_method, (str, float, int,)):
        if iv_aggregation_method in valid_iv_methods:
            iv_aggregation_method = (iv_aggregation_method,)
        else:
            raise ValueError(
                f'{iv_aggregation_method} is not a valid method for IV '
                f'estimation. Valid methods are {" ".join(valid_iv_methods)}.'
                )
    elif isinstance(iv_aggregation_method, (list, tuple,)):
        if not set(iv_aggregation_method) == set(valid_iv_methods):
            raise ValueError(
                f'{iv_aggregation_method} is not a valid method for IV '
                f'estimation. Valid methods are {" ".join(valid_iv_methods)}.'
                )
    else:
        iv_aggregation_method = default_method

    dic = {
        'ate_no_se_only': ate_no_se_only, 'cbgate': cbgate, 'atet': atet,
        'bgate': bgate, 'bt_yes': bt_yes, 'ci_level': ci_level,
        'choice_based_sampling': choice_based_sampling,
        'choice_based_probs': choice_based_probs, 'cluster_std': cluster_std,
        'cond_var': cond_var, 'bgate_sample_share': bgate_sample_share,
        'gates_minus_previous': gates_minus_previous,
        'gates_smooth': gates_smooth,
        'gates_smooth_bandwidth': gates_smooth_bandwidth,
        'gates_smooth_no_evalu_points': gates_smooth_no_evalu_points,
        'gatet': gatet, 'gate_no_evalu_points': gate_no_evalu_points,
        'iate':  iate, 'iate_se': iate_se, 'iate_m_ate': iate_m_ate,
        'knn': knn, 'knn_const': knn_const,
        'knn_min_k': knn_min_k, 'nw_bandw': nw_bandw, 'nw_kern': nw_kern,
        'max_cats_z_vars': max_cats_z_vars,
        'max_weight_share': max_weight_share,
        'qiate': qiate,  'qiate_se': qiate_se,
        'qiate_m_mqiate': qiate_m_mqiate, 'qiate_m_opp': qiate_m_opp,
        'qiate_quantiles': qiate_quantiles, 'qiate_smooth': qiate_smooth,
        'qiate_smooth_bandwidth': qiate_smooth_bandwidth,
        'qiate_bias_adjust': qiate_bias_adjust,
        'qiate_bias_adjust_draws': qiate_bias_adjust_draws,
        'se_boot_ate': se_boot_ate, 'se_boot_gate': se_boot_gate,
        'se_boot_iate': se_boot_iate, 'se_boot_qiate': se_boot_qiate,
        'q_w': q_w,
        'iv_aggregation_method': iv_aggregation_method,
        }
    # Define paths to save figures for plots of effects
    if gen_dic['with_output']:
        dic = mcf_sys.get_fig_path(dic, gen_dic['outpath'], 'ate_iate',
                                   gen_dic['with_output'])
        dic = mcf_sys.get_fig_path(dic, gen_dic['outpath'], 'gate',
                                   gen_dic['with_output'])
        if qiate:
            dic = mcf_sys.get_fig_path(dic, gen_dic['outpath'], 'qiate',
                                       gen_dic['with_output'])
        if cbgate:
            dic = mcf_sys.get_fig_path(dic, gen_dic['outpath'], 'cbgate',
                                       gen_dic['with_output'])
        if bgate:
            dic = mcf_sys.get_fig_path(dic, gen_dic['outpath'], 'bgate',
                                       gen_dic['with_output'])
    return dic


def post_init(p_dic, bin_corr_threshold=None, bin_corr_yes=None,
              est_stats=None, kmeans_no_of_groups=None, kmeans_max_tries=None,
              kmeans_replications=None, kmeans_yes=None, kmeans_single=None,
              kmeans_min_size_share=None, random_forest_vi=None,
              relative_to_first_group_only=None, plots=None, tree=None):
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
    kmeans_single = kmeans_single is True
    if kmeans_replications is None or kmeans_replications < 0:
        kmeans_replications = 10
    else:
        kmeans_replications = round(kmeans_replications)
    if kmeans_max_tries is None:
        kmeans_max_tries = 1000
    kmeans_max_tries = max(kmeans_max_tries, 10)
    if (kmeans_min_size_share is None
        or not isinstance(kmeans_min_size_share, (int, float))
            or not 0 < kmeans_min_size_share < 33):
        kmeans_min_size_share = 1

    add_pred_to_data_file = est_stats
    random_forest_vi = random_forest_vi is not False

    tree = tree is not False
    tree_depths = (2, 3, 4, 5)

    # Put everything in the dictionary
    dic = {
        'bin_corr_threshold': bin_corr_threshold, 'bin_corr_yes': bin_corr_yes,
        'est_stats': est_stats, 'kmeans_no_of_groups': kmeans_no_of_groups,
        'kmeans_max_tries': kmeans_max_tries,
        'kmeans_replications': kmeans_replications, 'kmeans_yes': kmeans_yes,
        'kmeans_single': kmeans_single,
        'kmeans_min_size_share': kmeans_min_size_share,
        'random_forest_vi': random_forest_vi, 'plots': plots,
        'relative_to_first_group_only': relative_to_first_group_only,
        'add_pred_to_data_file': add_pred_to_data_file,
        'tree': tree, 'tree_depths': tree_depths
        }
    return dic


def sens_init(p_dict, cbgate=None, bgate=None, gate=None, iate=None,
              iate_se=None, scenarios=None, cv_k=None,
              replications=2, reference_population=None, iate_df=None):
    """Initialise parameters of post-estimation analysis."""
    dic = {}
    # Check if types of inputs are ok
    if cv_k is not None and not isinstance(cv_k, (int, float)):
        raise TypeError('Number of folds for cross-validation must be integer,'
                        'float or None')
    if replications is not None and not isinstance(replications, (int, float)):
        raise TypeError('Number of replication must be integer, float or None')
    if scenarios is not None and not isinstance(scenarios, (list, tuple, str)):
        raise TypeError('Names of scenarios must be string or None')
    if cbgate is not None and not isinstance(cbgate, bool):
        raise TypeError('cbgate must be boolean or None')
    if bgate is not None and not isinstance(bgate, bool):
        raise TypeError('bgate must be boolean or None')
    if gate is not None and not isinstance(gate, bool):
        raise TypeError('gate must be boolean or None')
    if iate is not None and not isinstance(iate, bool):
        raise TypeError('iate must be boolean or None')
    if iate_se is not None and not isinstance(iate_se, bool):
        raise TypeError('iate_se must be boolean or None')
    if reference_population is not None and not isinstance(
            reference_population, (int, float)):
        raise TypeError('reference_population must be boolean or None')

    # Assign default values
    dic['cbgate'] = cbgate is True
    dic['bgate'] = bgate is True
    dic['gate'] = gate is True or dic['cbgate'] or dic['bgate']
    if iate_df is not None:
        dic['iate'] = True
    else:
        dic['iate'] = iate is True
    dic['iate_se'] = iate_se is True
    if dic['cbgate'] and not p_dict['cbgate']:
        raise ValueError('p_cbgate must be set to True if sens_cbgate is True')
    if dic['gate'] and not p_dict['bgate']:
        raise ValueError('p_bgate must be set to True if sens_bgate is True')

    if reference_population is None:
        dic['reference_population'] = None
    else:
        dic['reference_population'] = reference_population
    if cv_k is None or cv_k < 0.5:
        dic['cv_k'] = 5
    else:
        dic['cv_k'] = round(cv_k)

    if replications is None or replications < 0.5:
        dic['replications'] = 2
    else:
        dic['replications'] = round(replications)

    if scenarios is None:
        dic['scenarios'] = ('basic',)
    elif isinstance(scenarios, str):
        dic['scenarios'] = (scenarios,)
    else:
        raise ValueError(f'Senitivity Scenario {scenarios} not implemented')
    eligible_scenarios = ('basic',)
    wrong_scenarios = [scen for scen in dic['scenarios']
                       if scen not in eligible_scenarios]
    if wrong_scenarios:
        raise ValueError(f'{wrong_scenarios}'
                         f' {"are" if len(wrong_scenarios) > 1 else "is"}'
                         ' ineligable for sensitivity analysis')
    return dic
