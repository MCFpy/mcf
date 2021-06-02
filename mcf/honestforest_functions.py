"""Created on Sat Sep 12 12:59:15 2020.

Contains the functions needed for the running all parts of the programme

@author: MLechner
# -*- coding: utf-8 -*-
"""
import copy
import sys
import math
import time
import pprint
from concurrent import futures
import numpy as np
import pandas as pd
import psutil
import mcf.general_purpose as gp
import mcf.mcf_data_functions as mcf_data
import mcf.mcf_weight_functions as mcf_w
import mcf.mcf_vi_functions as mcf_vi
import mcf.mcf_gate_functions as mcf_gate
import mcf.mcf_forest_functions as mcf_forest
import mcf.honestforest_f2 as hf


def honestforest(
    pfad, outpfad, temppfad, datpfad, indata, preddata, outfiletext, id_name,
    y_name, y_tree_name, x_name_ord, x_name_unord, x_name_always_in_ord,
    x_name_always_in_unord, x_name_remain_unord, x_name_remain_ord,
    z_name_mgate, cluster_name, w_name, mp_parallel=-1, mp_type_vim=-1,
    direct_output_to_file=-1, screen_covariates=-1, n_min_grid=-1,
    check_perfectcorr=-1, n_min_min=-1, clean_data_flag=-1, min_dummy_obs=-1,
    boot=-1, n_min_max=-1, weighted=-1, subsample_factor=-1, m_min_share=-1,
    m_grid=-1, stop_empty=-1, m_random_poisson=-1, alpha_reg_min=-1,
    alpha_reg_max=-1, alpha_reg_grid=-1, random_thresholds=-1, knn_min_k=-1,
    share_forest_sample=-1, descriptive_stats=-1, m_max_share=-1,
    variable_importance_oob=-1, knn_const=-1, nw_kern_flag=-1,
    cond_var_flag=-1, knn_flag=-1, nw_bandw=-1, panel_data=-1,
    max_cats_cont_vars=-1, cluster_std=-1, fs_yes=-1, fs_other_sample_share=-1,
    fs_other_sample=-1, panel_in_rf=-1, fs_rf_threshold=-1, with_output=1,
    max_save_values=-1, max_weight_share=-1, smaller_sample=0,
    seed_sample_split=-1, save_forest=-1, orf=-1, orf_marg=-1, fontsize=-1,
    dpi=-1, ci_level=-1, marg_no_evaluation_points=-1, no_filled_plot=-1,
    weight_as_sparse=0, mp_with_ray=-1, mp_ray_objstore_multiplier=-1,
    verbose=-1, _no_ray_in_forest_building=False, _train_mcf=True,
        _pred_mcf=True):
    """Compute Honest Random Forest based on mcf module (therefore slow)."""
    time1 = time.time()
    # use smaller random sample (usually for testing purposes)
    if 0 < smaller_sample < 1:
        gp.randomsample(datpfad, indata + '.csv', 'smaller_idata.csv',
                        smaller_sample, True)
        if preddata != indata:
            gp.randomsample(datpfad, indata + '.csv', 'smaller_pdata.csv',
                            smaller_sample, True)
            preddata = 'smaller_preddata'
        else:
            preddata = 'smaller_indata'
        indata = 'smaller_indata'
    v_dict = {
        'id_name': id_name, 'cluster_name': cluster_name, 'w_name': w_name,
        'y_tree_name': y_tree_name, 'y_name': y_name, 'x_name_ord': x_name_ord,
        'x_name_unord': x_name_unord, 'x_name_always_in_ord':
        x_name_always_in_ord, 'x_name_always_in_unord': x_name_always_in_unord,
        'x_name_remain_ord': x_name_remain_ord, 'x_name_remain_unord':
        x_name_remain_unord, 'z_name_mgate': z_name_mgate,
        'x_name_amgate': []
        }
    c_dict = {
        'print_to_file': direct_output_to_file, 'pfad': pfad,
        'outpfad': outpfad, 'temppfad': temppfad, 'datpfad': datpfad,
        'indata': indata, 'preddata': preddata, 'outfiletext': outfiletext,
        'screen_covariates': screen_covariates, 'n_min_grid': n_min_grid,
        'check_perfectcorr': check_perfectcorr, 'n_min_min': n_min_min,
        'clean_data_flag': clean_data_flag, 'min_dummy_obs': min_dummy_obs,
        'boot': boot, 'w_yes': weighted, 'n_min_max': n_min_max, 'nw_bandw':
        nw_bandw, 'nw_kern': nw_kern_flag, 'subsample_factor':
        subsample_factor, 'm_min_share': m_min_share, 'm_grid': m_grid,
        'stop_empty': stop_empty, 'm_random_poisson': m_random_poisson,
        'alpha_reg_min': alpha_reg_min, 'alpha_reg_max': alpha_reg_max,
        'alpha_reg_grid': alpha_reg_grid, 'random_thresholds':
        random_thresholds, 'knn_const': knn_const, 'share_forest_sample':
        share_forest_sample, 'knn': knn_flag, 'desc_stat': descriptive_stats,
        'm_max_share': m_max_share, 'var_import_oob': variable_importance_oob,
        'cond_var': cond_var_flag, 'panel_data': panel_data,
        'max_cats_cont_vars': max_cats_cont_vars, 'cluster_std': cluster_std,
        'fs_yes': fs_yes, 'fs_other_sample_share': fs_other_sample_share,
        'fs_other_sample': fs_other_sample, 'panel_in_rf': panel_in_rf,
        'fs_rf_threshold': fs_rf_threshold, 'with_output': with_output,
        'no_parallel': mp_parallel, 'mp_type_vim': mp_type_vim,
        'max_save_values': max_save_values, 'max_weight_share':
        max_weight_share, 'knn_min_k': knn_min_k, 'save_forest': save_forest,
        'orf': orf, 'orf_marg': orf_marg, 'fig_fontsize': fontsize,
        'fig_dpi': dpi, 'fig_ci_level': ci_level, 'gmate_no_evaluation_points':
        marg_no_evaluation_points, 'no_filled_plot': no_filled_plot,
        'weight_as_sparse': weight_as_sparse, 'mp_with_ray': mp_with_ray,
        'mp_ray_objstore_multiplier': mp_ray_objstore_multiplier,
        'verbose': verbose, 'no_ray_in_forest_building':
        _no_ray_in_forest_building, 'train_mcf': _train_mcf,
        'pred_mcf': _pred_mcf}
    c_dict, v_dict = get_defaults(c_dict, v_dict)

    c_dict['mp_with_ray'] = False

    # Some descriptive stats of input and direction of output file
    if c_dict['with_output']:
        if c_dict['print_to_file']:
            orig_stdout = sys.stdout
            outfiletext = open(c_dict['outfiletext'], 'w')
            sys.stdout = outfiletext
        print('\nParameters for Honest Forest Estimation:')
        pprint.pprint(c_dict)
        pprint.pprint(v_dict)
        if c_dict['desc_stat']:
            gp.print_descriptive_stats_file(c_dict['indata'], 'all',
                                            c_dict['print_to_file'])
            gp.check_all_vars_in_data(c_dict['indata'],
                                      v_dict['names_to_check_train'])
        if c_dict['indata'] != c_dict['preddata']:
            if c_dict['desc_stat']:
                gp.print_descriptive_stats_file(
                    c_dict['preddata'], 'all', c_dict['print_to_file'])
                gp.check_all_vars_in_data(c_dict['preddata'],
                                          v_dict['names_to_check_pred'])
    else:
        c_dict['print_to_file'] = False

# Prepare data: Recode categorical variables to prime numbers, cont. vars
    (v_dict, var_x_type, var_x_values, c_dict, indata_with_z, predata_with_z,
     _, _, _, _, _, _, _, _
     ) = mcf_data.create_xz_variables(v_dict, c_dict, regrf=True)
    # Remove missing and keep only variables needed for further analysis
    if c_dict['clean_data_flag']:
        namen1_to_inc = gp.add_var_names(
            v_dict['id_name'], v_dict['y_name'], v_dict['cluster_name'],
            v_dict['w_name'], v_dict['y_tree_name'], v_dict['x_name'])
        indata2 = gp.clean_reduce_data(
            indata_with_z, c_dict['indata2_temp'],
            namen1_to_inc, c_dict['with_output'],
            c_dict['desc_stat'], c_dict['print_to_file'])
        if indata_with_z != predata_with_z:
            namen2_to_inc = gp.add_var_names(
                v_dict['id_name'], v_dict['cluster_name'], v_dict['w_name'],
                v_dict['x_name'])
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
    if c_dict['screen_covariates']:
        _, x_variables_out = gp.screen_variables(
            indata2, v_dict['x_name'], c_dict['check_perfectcorr'],
            c_dict['min_dummy_obs'], c_dict['with_output'])
        if x_variables_out != []:
            (v_dict, var_x_type, var_x_values, c_dict
             ) = mcf_data.adjust_variables(
                v_dict, var_x_type, var_x_values, c_dict, x_variables_out,
                True)
    # Sample splitting
    share1 = (1 - c_dict['fs_other_sample_share']
              ) * c_dict['share_forest_sample']
    share2 = 1 - c_dict['fs_other_sample_share'] - share1
    tree_sample, fill_y_sample, fs_sample = gp.sample_split_2_3(
        indata2, c_dict['tree_sample_temp'], share1,
        c_dict['fill_y_sample_temp'], share2, c_dict['fs_sample_temp'],
        c_dict['fs_other_sample_share'], seed_sample_split,
        c_dict['with_output'])
    if (c_dict['fs_yes'] == 1) and (c_dict['fs_other_sample'] == 0):
        fs_sample = tree_sample
    if c_dict['indata'] == c_dict['preddata']:
        preddata2 = fill_y_sample
        if c_dict['with_output']:
            print('\nBoth input samples are the same and sample splitting is',
                  'activated. Therefore, the part of the input that is used',
                  'for estimating the forest is not used as reference sample',
                  'to get more reliable inference')
# Pre-analysis feature selection
    time2 = time.time()
    if c_dict['fs_yes']:
        var_fs = copy.deepcopy(v_dict)
        # Analyse features
        if c_dict['with_output']:
            gp.statistics_covariates(fs_sample, var_x_type)
            print('\n\nFeature selection')
        cfs = copy.deepcopy(c_dict)
        if not isinstance(cfs['grid_n_min'], int):
            cfs['grid_n_min'] = cfs['grid_n_min'][0]
        cfs['m_grid'] = 1
        fs_f, x_name_mcf = mcf_forest.build_forest(
            fs_sample, var_fs, var_x_type, var_x_values, cfs, True)
        vi_i, vi_g, vi_ag, name = mcf_vi.variable_importance(
            fs_sample, fs_f, var_fs, var_x_type, var_x_values, cfs, x_name_mcf,
            True)
        v_dict, var_x_type, var_x_values = mcf_forest.fs_adjust_vars(
            vi_i, vi_g, vi_ag, v_dict, var_x_type, var_x_values, name, cfs,
            True)
        del cfs, vi_i, vi_g, vi_ag, name, var_fs
    if c_dict['with_output'] and c_dict['desc_stat']:
        mcf_forest.structure_of_node_tabl()
    if c_dict['orf']:
        if v_dict['y_name'] != v_dict['y_tree_name']:
            raise Exception('ORF allows only one outcome variable')
        c_dict, v_dict = orf_number_of_est(tree_sample, fill_y_sample,
                                           v_dict, c_dict)
        no_of_estimations = len(v_dict['orf_y_name'])
    else:
        no_of_estimations = 1
    time3 = time.time()
    t43 = 0
    t54 = 0
    t65 = 0
    t76 = 0
    t87 = 0
    weights_orf = []
    y_f_orf = []
    forest_orf = []
    for no_of_orf_var in range(no_of_estimations):
        if c_dict['orf']:
            current_y = [v_dict['orf_y_name'][no_of_orf_var]]
            v_dict['y_tree_name'] = current_y[:]
            v_dict['y_name'] = current_y[:]
        time3n = time.time()
# Estimate forest structure
        forest, x_name_mcf = mcf_forest.build_forest(
            tree_sample, v_dict, var_x_type, var_x_values, c_dict, True)
        time4 = time.time()
        t43 += time4 - time3n
# Variable importance
        if c_dict['var_import_oob'] and c_dict['with_output']:
            mcf_vi.variable_importance(tree_sample, forest, v_dict, var_x_type,
                                       var_x_values, c_dict, x_name_mcf, True)
        forest = mcf_forest.remove_oob_from_leaf0(forest)
        time5 = time.time()
        t54 += time5 - time4
# Filling of trees with indices of outcomes:
        forest, _, _ = mcf_forest.fill_trees_with_y_indices_mp(
            forest, fill_y_sample, v_dict, var_x_type, var_x_values, c_dict,
            x_name_mcf, True)
        time6 = time.time()
        t65 += time6 - time5
        weights, y_f, _, cl_f, w_f = mcf_w.get_weights_mp(
            forest, preddata2, fill_y_sample, v_dict, c_dict, x_name_mcf, True)
        if c_dict['orf']:
            y_f_orf.append(copy.deepcopy(y_f))
            weights_orf.append(copy.deepcopy(weights))  # add list of weights
            if c_dict['save_forest']:
                forest_orf.append(copy.deepcopy(forest))
        time7 = time.time()
        t76 += time7 - time6
    if c_dict['orf']:
        v_dict['y_name'] = v_dict['old_y_name']
        v_dict['y_tree_name'] = v_dict['old_y_name']
        weights = weights_orf
        y_f = y_f_orf
        if c_dict['save_forest']:
            forest = forest_orf
    if c_dict['save_forest']:
        gp.save_load(
            c_dict['save_forest_file'],
            (weights, forest, y_f, cl_f, w_f, c_dict, v_dict), save=True,
            output=c_dict['with_output'])
    if c_dict['marg_plots'] and (not c_dict['orf']) and c_dict['with_output']:
        mcf_gate.marg_gates_est(
            forest, fill_y_sample, preddata2, v_dict, c_dict, x_name_mcf,
            var_x_type, var_x_values, regrf=True)
    del forest
    time8a = time.time()
    if c_dict['orf']:
        # pred_outf, y_pred, y_pred_se, names_pred, names_pred_se =predict_orf(
        #     weights, preddata2, y_f, cl_f, w_f, v_dict, c_dict)
        predict_orf(weights, preddata2, y_f, cl_f, w_f, v_dict, c_dict)
    else:
        # pred_outf, y_pred, y_pred_se, names_pred, names_pred_se = predict_hf(
        #     weights, preddata2, y_f, cl_f, w_f, v_dict, c_dict)
        hf.predict_hf(weights, preddata2, y_f, cl_f, w_f, v_dict, c_dict)
    time8 = time.time()
    t87 += time8 - time8a
    time_string = ['Data preparation:                ',
                   'Feature preselection:            ',
                   'Estimate forest structure:       ',
                   'Variable importance              ',
                   'Fill tree with outcomes:         ',
                   'Weight computation:              ',
                   'Marginal predictive plots        ',
                   'Prediction:                      ',
                   'Total time:                      ']
    time_difference = [time2 - time1, time3 - time2, t43, t54, t65, t76,
                       time8a-time7, t87, time8 - time1]
    if c_dict['with_output']:
        print('\n')
        if c_dict['no_parallel'] < 2:
            print('No parallel processing')
        else:
            print('Multiprocessing')
            print('Number of cores used: ', c_dict['no_parallel'])
            if c_dict['mp_type_vim'] == 1:
                print('MP for variable importance was variable based')
            else:
                print('MP for variable importance was bootstrap based')
        gp.print_timing(time_string, time_difference)  # print to file
    if c_dict['print_to_file']:
        sys.stdout = orig_stdout
        outfiletext.close()  # last line of procedure


def get_defaults(c_dict, v_dict):
    """Large procedure to get all parameters and variables needed for HF.

    Parameters
    ----------
    c_dict : Dictory of parameters specified by user
    v_dict : Dictory of list of variable names specified by user

    Returns
    -------
    cn: Updated dictionary of parameters
    vn: Updated list of variables.

    """
    cnew_dict = copy.deepcopy(c_dict)
    # Define full path for various new files
    pred_sample_with_pred = (c_dict['datpfad'] + '/' + c_dict['indata']
                             + 'Predpred' + '.csv')
    cnew_dict['outfiletext'] = (c_dict['outpfad'] + '/' + c_dict['outfiletext']
                                + '.out')
    outfile_predict = c_dict['outpfad'] + '/' + c_dict['indata'] + 'CATE.csv'
    out_allindat = c_dict['outpfad'] + '/' + c_dict['indata'] + 'Predall.csv'
    cnew_dict['indata'] = c_dict['datpfad'] + '/' + c_dict['indata'] + '.csv'
    cnew_dict['preddata'] = (c_dict['datpfad'] + '/' + c_dict['preddata']
                             + '.csv')
    indat_with_pred = c_dict['outpfad'] + '/' + c_dict['indata'] + '_a_p.csv'
    save_forest_file = c_dict['outpfad'] + '/' + c_dict['indata'] + '.pickle'
    indat_temp = c_dict['temppfad'] + '/' + 'indat_temp' + '.csv'
    indata2_temp = c_dict['temppfad'] + '/' + 'indata2_temp' + '.csv'
    preddata2_temp = c_dict['temppfad'] + '/' + 'preddata2_temp' + '.csv'
    preddata3_temp = c_dict['temppfad'] + '/' + 'preddata3_temp' + '.csv'
    pred_eff_temp = c_dict['temppfad'] + '/' + 'pred_eff_temp' + '.csv'
    fs_sample_tmp = c_dict['temppfad'] + '/' + 'fs_sample_tmp' + '.csv'
    tree_sample_tmp = c_dict['temppfad'] + '/' + 'tree_sample_tmp' + '.csv'
    fill_y_sample_tmp = c_dict['temppfad'] + '/' + 'fill_y_sample_tmp' + '.csv'
    tree_sample_nn = c_dict['temppfad'] + '/' + 'tree_sample_NN' + '.csv'
    temporary_file = c_dict['temppfad'] + '/' + 'temporaryfile' + '.csv'
    if c_dict['print_to_file'] != 1:
        cnew_dict['print_to_file'] = False
    else:
        cnew_dict['print_to_file'] = True
    if c_dict['verbose'] == 0:
        cnew_dict['verbose'] = False
    else:
        cnew_dict['verbose'] = True
    mp_automatic = False
    if c_dict['no_parallel'] < -0.5:
        cnew_dict['no_parallel'] = round(psutil.cpu_count() * 0.9)
        mp_automatic = True
    elif -0.5 <= c_dict['no_parallel'] <= 1.5:
        cnew_dict['no_parallel'] = 1
    else:
        cnew_dict['no_parallel'] = round(c_dict['no_parallel'])
    sys_share = 0.7 * getattr(psutil.virtual_memory(), 'percent') / 100
    if c_dict['screen_covariates'] != 0:
        cnew_dict['screen_covariates'] = True
    else:
        cnew_dict['screen_covariates'] = False
    if c_dict['check_perfectcorr'] == 0:
        cnew_dict['check_perfectcorr'] = False
    else:
        cnew_dict['check_perfectcorr'] = True
    if c_dict['min_dummy_obs'] < 1:
        cnew_dict['min_dummy_obs'] = 10
    else:
        cnew_dict['min_dummy_obs'] = round(c_dict['min_dummy_obs'])
    if c_dict['clean_data_flag'] == 0:
        cnew_dict['clean_data_flag'] = False
    else:
        cnew_dict['clean_data_flag'] = True

    if (c_dict['alpha_reg_min'] < 0) or (c_dict['alpha_reg_min'] >= 0.4):
        cnew_dict['alpha_reg_min'] = 0.1
        if c_dict['alpha_reg_min'] >= 0.4:
            print('Values of 0.4 and larger do not make sense for alpha',
                  'regularisation. alpha regularisation parameter set to ',
                  'default value. {:8.3f}'.format(cnew_dict['alpha_reg_min']))
    else:
        cnew_dict['alpha_reg_min'] = c_dict['alpha_reg_min']
    if (c_dict['alpha_reg_max'] < 0) or (c_dict['alpha_reg_max'] >= 0.4):
        cnew_dict['alpha_reg_max'] = 0.2
        if c_dict['alpha_reg_max'] >= 0.4:
            print('Values of 0.4 and larger do not make sense for alpha',
                  'regularisation. alpha regularisation parameter set to ',
                  'default value. {:8.3f}'.format(cnew_dict['alpha_reg_max']))
    else:
        cnew_dict['alpha_reg_max'] = c_dict['alpha_reg_max']
    cnew_dict['alpha_reg_grid'] = round(c_dict['alpha_reg_grid'])
    if cnew_dict['alpha_reg_grid'] < 1:
        cnew_dict['alpha_reg_grid'] = 2
    if cnew_dict['alpha_reg_min'] >= cnew_dict['alpha_reg_max']:
        cnew_dict['alpha_reg_grid'] = 1
        cnew_dict['alpha_reg_max'] = cnew_dict['alpha_reg_min']
    if cnew_dict['alpha_reg_grid'] == 1:
        alpha_reg = (cnew_dict['alpha_reg_max']
                     + cnew_dict['alpha_reg_min']) / 2
    elif cnew_dict['alpha_reg_grid'] == 2:
        alpha_reg = np.hstack((cnew_dict['alpha_reg_min'],
                               cnew_dict['alpha_reg_max']))
    else:
        alpha_reg = np.linspace(
            cnew_dict['alpha_reg_min'], cnew_dict['alpha_reg_max'],
            cnew_dict['alpha_reg_grid'])
        alpha_reg = list(np.unique(alpha_reg))

    if ((c_dict['share_forest_sample'] < 0.01)
            or (c_dict['share_forest_sample'] > 0.99)):
        cnew_dict['share_forest_sample'] = 0.5
    if c_dict['panel_data'] == 1:
        cnew_dict['cluster_std'] = True
        c_dict['cluster_std'] = True
        if c_dict['panel_in_rf'] == 1:
            cnew_dict['panel_in_rf'] = True
        else:  # default
            cnew_dict['panel_in_rf'] = False
        panel = True
    else:
        cnew_dict['panel_data'] = False
        panel = False
        cnew_dict['panel_in_rf'] = False
    if c_dict['cluster_std'] == 1:
        cnew_dict['cluster_std'] = True
    else:
        cnew_dict['cluster_std'] = False
    if not (cnew_dict['cluster_std'] or panel):
        cnew_dict['cluster_std'] = False
    data = pd.read_csv(cnew_dict['indata'], header=0)
    n_sample = len(data.index)
    if c_dict['mp_type_vim'] != 1 or c_dict['mp_type_vim'] != 2:
        if n_sample < 20000:
            cnew_dict['mp_type_vim'] = 1  # MP over var's, fast, lots of memory
        else:
            cnew_dict['mp_type_vim'] = 2  # MP over bootstraps.
    cnew_dict['n_min_min'] = round(c_dict['n_min_min'])  # grid min.leafsize
    if cnew_dict['n_min_min'] < 1:
        if cnew_dict['n_min_min'] == -1:
            cnew_dict['n_min_min'] = round((n_sample**0.4)/10)
            if cnew_dict['n_min_min'] < 5:
                cnew_dict['n_min_min'] = 5
        else:
            cnew_dict['n_min_min'] = round((n_sample**0.4)/20)
            if cnew_dict['n_min_min'] < 3:
                cnew_dict['n_min_min'] = 3
    cnew_dict['n_min_max'] = round(c_dict['n_min_max'])
    if cnew_dict['n_min_max'] < 1:
        if cnew_dict['n_min_max'] == -1:
            cnew_dict['n_min_max'] = round(math.sqrt(n_sample)/5)
            if cnew_dict['n_min_max'] < 5:
                cnew_dict['n_min_max'] = 5
        else:
            cnew_dict['n_min_max'] = round(math.sqrt(n_sample)/10)
            if cnew_dict['n_min_max'] < 3:
                cnew_dict['n_min_max'] = 3
    cnew_dict['n_min_grid'] = round(c_dict['n_min_grid'])
    if cnew_dict['n_min_grid'] < 1:
        cnew_dict['n_min_grid'] = 2
    if cnew_dict['n_min_min'] == cnew_dict['n_min_max']:
        cnew_dict['n_min_grid'] = 1
    elif cnew_dict['n_min_max'] < cnew_dict['n_min_min']:
        cnew_dict['n_min_grid'] = 1
        cnew_dict['n_min_max'] = cnew_dict['n_min_min']
    if cnew_dict['n_min_grid'] == 1:
        n_min = round((cnew_dict['n_min_max'] + cnew_dict['n_min_min']) / 2)
    elif cnew_dict['n_min_grid'] == 2:
        n_min = np.hstack((cnew_dict['n_min_min'], cnew_dict['n_min_max']))
    else:
        n_min = np.linspace(cnew_dict['n_min_min'], cnew_dict['n_min_max'],
                            cnew_dict['n_min_grid'])
        n_min = list(np.unique(np.round(n_min)))
    # Select grid for number of parameters
    if (c_dict['m_min_share'] <= 0) or (c_dict['m_min_share'] > 1):
        cnew_dict['m_min_share'] = 0.1     # Hastie et al suggest p/3, sqrt(p)
        if c_dict['m_min_share'] == -2:    # around page 588
            cnew_dict['m_min_share'] = 0.2
    if (c_dict['m_max_share'] <= 0) or (c_dict['m_max_share'] > 1):
        cnew_dict['m_max_share'] = 0.66
        if c_dict['m_max_share'] == -2:
            cnew_dict['m_max_share'] = 0.8
    # Instead of selecting fixed number, select mean of Poisson distribution
    if (c_dict['m_random_poisson'] == 1) or (c_dict['m_random_poisson'] == -1):
        cnew_dict['m_random_poisson'] = True
    else:
        cnew_dict['m_random_poisson'] = False
    if c_dict['m_grid'] < 1:  # values of grid for m
        cnew_dict['m_grid'] = 3
    else:
        cnew_dict['m_grid'] = round(c_dict['m_grid'])

    if c_dict['w_yes'] == 1:  # Weighting
        cnew_dict['w_yes'] = True
    else:
        c_dict['w_yes'] = 0
        cnew_dict['w_yes'] = False

    if c_dict['boot'] > 0:   # Number of bootstrap replications
        cnew_dict['boot'] = round(c_dict['boot'])
    else:
        cnew_dict['boot'] = 1000
    # Operational parameters of forest building
    if c_dict['max_cats_cont_vars'] < 1:
        cnew_dict['max_cats_cont_vars'] = n_sample+1
    else:
        cnew_dict['max_cats_cont_vars'] = round(c_dict['max_cats_cont_vars'])
    if c_dict['stop_empty'] < 1:
        cnew_dict['stop_empty'] = 25
    else:
        cnew_dict['stop_empty'] = round(c_dict['stop_empty'])
    if c_dict['random_thresholds'] < 0:     # Saves computation time
        cnew_dict['random_thresholds'] = 20
    if c_dict['max_weight_share'] <= 0:
        cnew_dict['max_weight_share'] = 0.05
    # Parameters for the variance estimation
    if c_dict['cond_var'] < 0:       # 0: variance estimation uses Var(wY),
        cnew_dict['cond_var'] = True   # 1: conditional mean&variances are used
    else:
        cnew_dict['cond_var'] = False
    if c_dict['knn'] != 1:
        cnew_dict['knn'] = False
    else:
        cnew_dict['knn'] = True
    if c_dict['knn_min_k'] < 0:     # minimum # of neighbours in k-NN estimat.
        cnew_dict['knn_min_k'] = 10
    if c_dict['knn_min_k'] < 0:     # minimum # of neighbours in k-NN estimat.
        cnew_dict['knn_min_k'] = 10
    if c_dict['knn_const'] < 0:  # k: Konstant in # of neighbour estimation
        cnew_dict['knn_const'] = 1
    if c_dict['nw_bandw'] < 0:   # bandwidth for NW estimation multiplier of
        cnew_dict['nw_bandw'] = 1  # silverman's optimal bandwidth
    if c_dict['nw_kern'] != 2:  # kernel for NW est.: 1: Epanechikov 2: Normal
        cnew_dict['nw_kern'] = 1
    if c_dict['desc_stat'] != 0:            # Decriptive statistics
        cnew_dict['desc_stat'] = True
    else:
        cnew_dict['desc_stat'] = False
    if not c_dict['var_import_oob'] == 0:
        cnew_dict['var_import_oob'] = True
    else:                    # importance measure
        cnew_dict['var_import_oob'] = False
    # Feature preselection
    if c_dict['fs_yes'] <= 0:
        cnew_dict['fs_yes'] = False
    else:
        cnew_dict['fs_yes'] = True
    if c_dict['fs_rf_threshold'] <= 0:
        cnew_dict['fs_rf_threshold'] = 0
    if (c_dict['fs_other_sample'] < 0) or (c_dict['fs_other_sample'] == 1):
        cnew_dict['fs_other_sample'] = True
    else:
        cnew_dict['fs_other_sample'] = False
    if ((c_dict['fs_other_sample_share'] < 0)
            or (c_dict['fs_other_sample_share'] > 0.5)):
        cnew_dict['fs_other_sample_share'] = 0.2
    if ((cnew_dict['fs_other_sample'] is False)
            or (cnew_dict['fs_yes'] is False)):
        cnew_dict['fs_other_sample_share'] = 0
    # size of subsampling samples
    subsam_share = 2 * ((n_sample / 2)**0.85) / (n_sample / 2)
    if subsam_share > 0.67:
        subsam_share = 0.67
    if c_dict['subsample_factor'] <= 0:
        cnew_dict['subsample_factor'] = 1
    subsam_share = subsam_share * cnew_dict['subsample_factor']
    if subsam_share > 0.8:
        subsam_share = 0.8
    if subsam_share < 1e-4:
        subsam_share = 1e-4
    q_w = [0.5, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]  # Weight analysis
    if c_dict['save_forest'] != 1:
        cnew_dict['save_forest'] = False
    else:
        cnew_dict['save_forest'] = True
    if c_dict['orf'] != 1:
        cnew_dict['orf'] = False
    else:
        cnew_dict['orf'] = True
    if (c_dict['orf_marg'] != 0) and cnew_dict['orf']:
        cnew_dict['orf_marg'] = True
    else:
        cnew_dict['orf_marg'] = False

    if (c_dict['fig_fontsize'] > 0.5) and (c_dict['fig_fontsize'] < 7.5):
        cnew_dict['fig_fontsize'] = round(c_dict['fig_fontsize'])
    else:
        cnew_dict['fig_fontsize'] = 2
    all_fonts = ('xx-small', 'x-small', 'small', 'medium', 'large', 'x-large',
                 'xx-large')
    for idx, i_lab in enumerate(all_fonts):
        if cnew_dict['fig_fontsize'] == idx + 1:
            cnew_dict['fig_fontsize'] = i_lab
    if c_dict['fig_dpi'] < 10:
        cnew_dict['fig_dpi'] = 500
    else:
        cnew_dict['fig_dpi'] = round(c_dict['fig_dpi'])
    if (c_dict['fig_ci_level'] < 0.5) or (c_dict['fig_ci_level'] > 0.999999):
        cnew_dict['fig_ci_level'] = 0.90
    if c_dict['no_filled_plot'] < 5:
        cnew_dict['no_filled_plot'] = 20
    else:
        cnew_dict['no_filled_plot'] = round(c_dict['no_filled_plot'])
    if c_dict['gmate_no_evaluation_points'] < 2:
        cnew_dict['gmate_no_evaluation_points'] = 50
    else:
        cnew_dict['gmate_no_evaluation_points'] = round(
            c_dict['gmate_no_evaluation_points'])

    if c_dict['mp_with_ray'] == 0:
        cnew_dict['mp_with_ray'] = False
    else:
        cnew_dict['mp_with_ray'] = True

    if c_dict['mp_with_ray'] == 0:
        cnew_dict['mp_with_ray'] = False
    else:
        cnew_dict['mp_with_ray'] = True

    data_train = pd.read_csv(cnew_dict['indata'], header=0)
    n_train = len(data_train.index)
    memory = psutil.virtual_memory()
    # Obs., and number of trees as determinants of obj.store when forest build
    min_obj_str_n = (n_train / 60000) * (cnew_dict['boot'] / 1000) * (
        cnew_dict['m_grid'] * cnew_dict['n_min_grid'] * cnew_dict[
            'alpha_reg_grid'] / 12) * (120 * 1024 * 1024 * 1024)
    mem_object_store = min(memory.available*0.5, min_obj_str_n)
    if c_dict['mp_ray_objstore_multiplier'] is None:
        c_dict['mp_ray_objstore_multiplier'] = -1
    if c_dict['mp_ray_objstore_multiplier'] > 0:
        mem_object_store *= c_dict['mp_ray_objstore_multiplier']
    if mem_object_store > 0.5 * memory.available:
        mem_object_store = 0.5 * memory.available
    if mem_object_store < 78643200:
        mem_object_store = 78643200
    mem_object_store_larger_n = 10000
    if c_dict['no_ray_in_forest_building'] is True:
        cnew_dict['no_ray_in_forest_building'] = True
    else:
        cnew_dict['no_ray_in_forest_building'] = False

    cnew_dict['weight_as_sparse'] = False
    cnew_dict['no_of_treat'] = None
    cnew_dict['d_values'] = None
    cnew_dict['boot_by_boot'] = 10

    cn_add = {'pred_sample_with_pred': pred_sample_with_pred,
              'outfile_predict':    outfile_predict,
              'out_allindat':       out_allindat,
              'indat_with_pred':    indat_with_pred,
              'indat_temp':         indat_temp,
              'indata2_temp':       indata2_temp,
              'preddata2_temp':     preddata2_temp,
              'preddata3_temp':     preddata3_temp,
              'pred_eff_temp':      pred_eff_temp,
              'fs_sample_temp':     fs_sample_tmp,
              'tree_sample_temp':   tree_sample_tmp,
              'fill_y_sample_temp': fill_y_sample_tmp,
              'tree_sample_nn':     tree_sample_nn,
              'temporary_file':     temporary_file,
              'panel':              panel,
              'grid_n_min':         n_min,
              'save_forest_file':   save_forest_file,
              'title_variance':     'Weight-based variance',
              'add_pred_to_data_file': True,
              'subsam_share':       subsam_share,
              'q_w':                q_w,
              'sys_share':          sys_share,
              'mp_automatic':       mp_automatic,
              'grid_alpha_reg':     alpha_reg,
              'mem_object_store':   mem_object_store,
              'mem_object_store_1': mem_object_store,
              'mem_object_store_2': mem_object_store,
              'mem_object_store_3': mem_object_store,
              'mem_object_store_larger_n': mem_object_store_larger_n,
              'mp_weights_tree_batch': 0
              }
    cnew_dict.update(cn_add)
    vnew_dict = copy.deepcopy(v_dict)
    if cnew_dict['cluster_std'] or cnew_dict['panel']:
        if len(v_dict['cluster_name']) > 1:
            print('More than one name for cluster variable.')
            sys.exit()
        vnew_dict['cluster_name'] = gp.cleaned_var_names(
            vnew_dict['cluster_name'])
    else:
        vnew_dict['cluster_name'] = []
    vnew_dict['y_tree_name'] = gp.cleaned_var_names(vnew_dict['y_tree_name'])
    vnew_dict['y_name'].extend(vnew_dict['y_tree_name'])
    vnew_dict['y_name'] = gp.cleaned_var_names(vnew_dict['y_name'])
    vnew_dict['x_name_ord'] = gp.cleaned_var_names(vnew_dict['x_name_ord'])
    vnew_dict['x_name_unord'] = gp.cleaned_var_names(vnew_dict['x_name_unord'])
    vnew_dict['x_name_always_in_ord'] = gp.cleaned_var_names(
        vnew_dict['x_name_always_in_ord'])
    vnew_dict['x_name_always_in_unord'] = gp.cleaned_var_names(
        vnew_dict['x_name_always_in_unord'])
    vnew_dict['x_name_remain_ord'] = gp.cleaned_var_names(
        vnew_dict['x_name_remain_ord'])
    vnew_dict['x_name_remain_unord'] = gp.cleaned_var_names(
        vnew_dict['x_name_remain_unord'])
    if cnew_dict['w_yes'] == 0:
        vnew_dict['w_name'] = []
    else:
        vnew_dict['w_name'] = gp.cleaned_var_names(vnew_dict['w_name'])
    vnew_dict['id_name'] = gp.cleaned_var_names(vnew_dict['id_name'])

    x_name = copy.deepcopy(vnew_dict['x_name_ord'] + vnew_dict['x_name_unord'])
    x_name = gp.cleaned_var_names(x_name)
    x_name_in_tree = gp.cleaned_var_names(copy.deepcopy(
        vnew_dict['x_name_always_in_ord']
        + vnew_dict['x_name_always_in_unord']))
    x_name_remain = gp.cleaned_var_names(copy.deepcopy(
        vnew_dict['x_name_remain_ord'] + vnew_dict['x_name_remain_unord']
        + x_name_in_tree))
    x_name_remain = gp.cleaned_var_names(x_name_remain)
    if x_name_in_tree != []:
        x_name_remain = gp.cleaned_var_names(copy.deepcopy(
            x_name_in_tree + x_name_remain))
        x_name = gp.cleaned_var_names(copy.deepcopy(x_name_in_tree + x_name))
    x_name_always_in = gp.cleaned_var_names(copy.deepcopy(
        vnew_dict['x_name_always_in_ord']
        + vnew_dict['x_name_always_in_unord']))
    name_ordered = gp.cleaned_var_names(copy.deepcopy(
        vnew_dict['x_name_ord'] + vnew_dict['x_name_always_in_ord']
        + vnew_dict['x_name_remain_ord']))
    name_unordered = gp.cleaned_var_names(copy.deepcopy(
        vnew_dict['x_name_unord'] + vnew_dict['x_name_always_in_unord']
        + vnew_dict['x_name_remain_unord']))
    if not ((name_ordered == []) or (name_unordered == [])):
        if any([value for value in name_ordered if value in name_unordered]):
            print('Remove overlap in ordered and unordered variables')
            sys.exit()
    v_dict['x_name_margplots'] = gp.cleaned_var_names(
        v_dict['z_name_mgate'])
    vnew_dict['z_name_mgate'] = list(
        set(v_dict['z_name_mgate']).intersection(x_name))
    if not vnew_dict['z_name_mgate']:  # no names left
        marg_plots = False
    else:
        marg_plots = True
    cnew_dict.update({'marg_plots': marg_plots})
    # Define variables for consistency check in data sets
    names_to_check_train = vnew_dict['y_name'] + x_name
    names_to_check_pred = x_name
    vn_add = {
        'x_name_remain':        x_name_remain,
        'x_name':               x_name,
        'x_name_in_tree':       x_name_in_tree,
        'x_name_always_in':     x_name_always_in,
        'name_ordered':         name_ordered,
        'name_unordered':       name_unordered,
        'names_to_check_train': names_to_check_train,
        'names_to_check_pred':  names_to_check_pred}
    vnew_dict.update(vn_add)
    return cnew_dict, vnew_dict


def predict_orf(weights, data_file, y_orf_data, cl_data, w_data, v_dict,
                c_dict):
    """Compute predictions & their std.errors and save to file (ORF).

    Parameters
    ----------
    weights : List of lists. For every obs, positive weights are saved.
    pred_data : String. csv-file with data to make predictions for.
    y_orf_data : List of Numpy arrays. All outcome variables.
    cl_data : Numpy array. Cluster variable.
    w_data : Numpy array. Sampling weights.
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.

    Returns
    -------
    post_estimation_file : String. Name of files with predictions.
    y_pred : Numpy array. Predictions.
    y_pred_se: Numpy array. Standard errors of predictions.
    names_y_pred: List of strings: All names of Predictions in file.
    names_y_pred_se: List of strings: All names of SE of predictions in file.

    """
    if c_dict['with_output'] and not c_dict['print_to_file']:
        print('\nComputing predictions')
    n_x = len(weights[0])
    n_y = np.size(y_orf_data[0], axis=0)
    v_dict['orf_y_name'].append(v_dict['old_y_name'][0] + '0')
    no_of_out = len(v_dict['orf_y_name'])
    pred_y = np.empty((n_x, no_of_out))
    pred_y_se = np.empty((n_x, no_of_out))
    no_cases = 0
    for y_idx, y_name in enumerate(v_dict['orf_y_name']):
        if c_dict['with_output'] and not c_dict['print_to_file']:
            print(y_name)
        larger_0 = 0
        equal_0 = 0
        mean_pos = 0
        std_pos = 0
        gini_all = 0
        gini_pos = 0
        share_censored = 0
        share_largest_q = np.zeros(3)
        sum_larger = np.zeros(len(c_dict['q_w']))
        obs_larger = np.zeros(len(c_dict['q_w']))
        if not c_dict['w_yes']:
            w_data = None
        if c_dict['cluster_std']:
            no_of_cluster = np.size(np.unique(cl_data))
        else:
            no_of_cluster = None
        if y_idx == 0:
            weights_n = [[[[0]]]] * n_x
            y_orf_n = None
        else:
            weights_n = weights[y_idx-1]
            y_orf_n = y_orf_data[y_idx-1]
            # y_orf_n = y_orf_n.flatten()
            y_orf_n = y_orf_n.reshape(-1)
        if y_idx == no_of_out-1:
            weights_p = [[[[0]]]] * n_x
            y_orf_p = None
        else:
            weights_p = weights[y_idx]
            y_orf_p = y_orf_data[y_idx]
            # y_orf_p = y_orf_p.flatten()
        if c_dict['no_parallel'] < 1.5:
            maxworkers = 1
        else:
            if c_dict['mp_automatic']:
                maxworkers = gp.find_no_of_workers(c_dict['no_parallel'],
                                                   c_dict['sys_share'])
            else:
                maxworkers = c_dict['no_parallel']
        if c_dict['with_output']:
            print('Number of parallel processes: ', maxworkers)
            if maxworkers == 1:
                for idx in range(n_x):
                    ret_all_i = pred_func1_for_mp_orf(
                        idx, weights_p[idx][:][0], weights_n[idx][:][0],
                        cl_data, no_of_cluster, w_data, y_orf_p, y_orf_n, n_y,
                        c_dict)
                    pred_y[idx, y_idx] = ret_all_i[1]
                    pred_y_se[idx, y_idx] = ret_all_i[2]
                    larger_0 += ret_all_i[3][0]
                    equal_0 += ret_all_i[3][1]
                    mean_pos += ret_all_i[3][2]
                    std_pos += ret_all_i[3][3]
                    gini_all += ret_all_i[3][4]
                    gini_pos += ret_all_i[3][5]
                    share_largest_q += ret_all_i[3][6]
                    sum_larger += ret_all_i[3][7]
                    obs_larger += ret_all_i[3][8]
                    share_censored += ret_all_i[4]
                    no_cases += 1
                    if c_dict['with_output'] and not c_dict['print_to_file']:
                        gp.share_completed(idx+1, n_x)
            else:
                with futures.ProcessPoolExecutor(max_workers=maxworkers
                                                 ) as fpp:
                    ret_fut = {fpp.submit(
                        pred_func1_for_mp_orf, idx, weights_p[idx][:][0],
                        weights_n[idx][:][0], cl_data, no_of_cluster, w_data,
                        y_orf_p, y_orf_n, n_y, c_dict):
                            idx for idx in range(n_x)}
                    for jdx, fer in enumerate(futures.as_completed(ret_fut)):
                        ret_all_i = fer.result()
                        del ret_fut[fer]
                        iix = ret_all_i[0]
                        pred_y[iix, y_idx] = ret_all_i[1]
                        pred_y_se[iix, y_idx] = ret_all_i[2]
                        larger_0 += ret_all_i[3][0]
                        equal_0 += ret_all_i[3][1]
                        mean_pos += ret_all_i[3][2]
                        std_pos += ret_all_i[3][3]
                        gini_all += ret_all_i[3][4]
                        gini_pos += ret_all_i[3][5]
                        share_largest_q += ret_all_i[3][6]
                        sum_larger += ret_all_i[3][7]
                        obs_larger += ret_all_i[3][8]
                        share_censored += ret_all_i[4]
                        no_cases += 1
                        share_censored += ret_all_i[4]
                        if c_dict['with_output'] and not c_dict[
                                'print_to_file']:
                            gp.share_completed(jdx+1, n_x)
        larger_0 /= no_cases
        equal_0 /= no_cases
        mean_pos /= no_cases
        std_pos /= no_cases
        gini_all /= no_cases
        gini_pos /= no_cases
        share_largest_q /= no_cases
        sum_larger /= no_cases
        obs_larger /= no_cases
        share_censored /= no_cases
        if c_dict['with_output']:
            print('\n')
            print('=' * 80)
            print('Analysis of weights (normalised to add to 1): ',
                  'Predicted', '(stats are averaged over all predictions)')
            hf.print_weight_stat_pred(larger_0 / n_x, equal_0 / n_x,
                                      mean_pos / n_x, std_pos / n_x,
                                      gini_all / n_x, gini_pos / n_x,
                                      share_largest_q / n_x, sum_larger / n_x,
                                      obs_larger / n_x, c_dict, share_censored)
    if c_dict['with_output']:
        hf.print_pred(None, pred_y, pred_y_se, w_data, c_dict, v_dict)
    # Add results to data file
    name_pred = []
    idx = 0
    for o_name in v_dict['orf_y_name']:
        name_pred += [o_name]
    name_pred_y = [s + '_pred' for s in name_pred]
    name_pred_y_se = [s + '_pred_se' for s in name_pred]
    pred_y_df = pd.DataFrame(data=pred_y, columns=name_pred_y)
    pred_y_se_df = pd.DataFrame(data=pred_y_se, columns=name_pred_y_se)
    data_df = pd.read_csv(data_file)
    df_list = [data_df, pred_y_df, pred_y_se_df]
    data_file_new = pd.concat(df_list, axis=1)
    gp.delete_file_if_exists(c_dict['pred_sample_with_pred'])
    data_file_new.to_csv(c_dict['pred_sample_with_pred'], index=False)
    if c_dict['with_output']:
        gp.print_descriptive_stats_file(
            c_dict['pred_sample_with_pred'], 'all', c_dict['print_to_file'])
    return (c_dict['pred_sample_with_pred'], pred_y, pred_y_se, name_pred_y,
            name_pred_y_se)


def pred_func1_for_mp_orf(idx, weights_p_idx, weights_n_idx, cl_data,
                          no_of_cluster, w_data, y_p_data, y_n_data,
                          n_y, c_dict):
    """
    Compute function to be looped over observations for Multiprocessing (ORF).

    Parameters
    ----------
    idx : Int. Counter.
    weights_p_idx : List of int. Indices of non-zero weights. Positive.
    weights_n_idx : List of int. Indices of non-zero weights. Negative.
    cl_data : Numpy vector. Cluster variable.
    no_of_cluster : Int. Number of clusters.
    w_data : Numpy vector. Sampling weights.
    y : Numpy array. Outcome variable.
    n_y : Int. Length of outcome data.
    c_dict : Dict. Parameters.

    Returns
    -------
    i: Int. Counter.
    y_pred_i: Numpy array.
    y_pred_se_i: Numpy array.
    l1_to_9: Tuple of lists.
    """
    if c_dict['with_output']:
        print('procedure not yet fully implemented')
        if cl_data is not None:
            print('Clustervars not yet used', no_of_cluster)
    if len(weights_p_idx[0]) > 1:
        w_p_index = weights_p_idx[0]  # Indices of non-zero weights
        w_p_i = np.array(weights_p_idx[1], copy=True)
    else:
        w_p_index = None
        w_p_i = None
    if len(weights_n_idx[0]) > 1:
        w_n_index = weights_n_idx[0]  # Indices of non-zero weights
        w_n_i = np.array(weights_n_idx[1], copy=True)
    else:
        w_n_index = None
        w_n_i = None
    if (w_n_i is None) and (w_p_i is None):
        raise Exception('Both weights are None. Impossible.')
    if c_dict['w_yes']:
        if w_p_i is not None:
            # w_p_t = w_data[w_p_index].flatten()
            # w_p_i = w_p_i * w_p_t
            w_p_i = w_p_i * w_data[w_p_index].reshape(-1)
        if w_n_i is not None:
            # w_n_t = w_data[w_n_index].flatten()
            # w_n_i = w_n_i * w_n_t
            w_n_i = w_n_i * w_data[w_n_index].reshape(-1)
    # else:
    #     w_p_t = None
    #     w_n_t = None
    if w_p_i is not None:
        w_p_i = w_p_i / np.sum(w_p_i)
    if w_n_i is not None:
        w_n_i = w_n_i / np.sum(w_n_i)
    if c_dict['max_weight_share'] < 1:
        if w_p_i is not None:
            w_p_i, _, share_p_i = gp.bound_norm_weights(
                w_p_i, c_dict['max_weight_share'])
        if w_n_i is not None:
            w_n_i, _, share_n_i = gp.bound_norm_weights(
                w_n_i, c_dict['max_weight_share'])
    if w_n_i is not None:
        w_n_all_i = np.zeros(n_y)
        w_n_all_i[w_n_index] = w_n_i.flatten()
    if w_p_i is None:
        share_i = share_n_i
        y_pred_idx = 1 - np.sum(w_n_all_i * y_n_data)
    else:
        share_i = share_p_i
        w_p_all_i = np.zeros(n_y)
        w_p_all_i[w_p_index] = w_p_i.flatten()
        if w_n_i is None:
            y_pred_idx = np.sum(w_p_all_i * y_p_data)
        else:
            y_pred_idx = np.sum(w_p_all_i * y_p_data - w_n_all_i * y_n_data)
    # ret = gp.weight_var(w_i, y_data, cl_data, c_dict, norm=False,
    #                     weights=w_data)
    # pred_y_idx = np.copy(ret[0])
    # pred_y_se_idx = np.sqrt(ret[1])
    if y_pred_idx < 0:
        y_pred_idx = 0
    elif y_pred_idx > 1:
        y_pred_idx = 1
    y_pred_se_idx = -1
    # if c_dict['cluster_std']:
    #     ret2 = gp.aggregate_cluster_pos_w(cl_data, w_i, y_data,
    #                                       sweights=w_data, norma=False)
    #     w_add = np.copy(ret2[0])
    # else:
    #     w_add = np.copy(ret[2])
    if w_p_i is None:
        l1_to_9 = hf.analyse_weights_pred(w_n_all_i, c_dict)
    else:
        l1_to_9 = hf.analyse_weights_pred(w_p_all_i, c_dict)
    return idx, y_pred_idx, y_pred_se_idx, l1_to_9, share_i


def orf_number_of_est(csv_file1, csv_file2, v_dict, c_dict):
    """
    Generate ORF specific outcome variables and save to file.

    Parameters
    ----------
    csv_file1, csv_file2 : Strings. Name of csv data files. Will be appended.
    v_dict : Dict. Variables.
    c_dict : Dict. Controls.

    Returns
    -------
    c_dict : Dict.
    v_dict : Dict.
    """
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    y1_df = df1[v_dict['y_name']]
    y2_df = df2[v_dict['y_name']]
    count = y1_df.value_counts()
    y_values1 = np.int16(np.flip(np.unique(y1_df.to_numpy())))
    y_values2 = np.int16(np.flip(np.unique(y2_df.to_numpy())))
    if np.any(y_values1 != y_values2):
        raise Exception('ORF: Values of dependent variables not identical',
                        'in both samples', y_values1, y_values2)
    new_names = []
    for y_val in y_values1[:-1]:
        print(y_val)
        new_n = v_dict['y_name'][0] + str(y_val)
        df1[new_n] = (y1_df >= (y_val-1e-15)).astype('int')
        df2[new_n] = (y2_df >= (y_val-1e-15)).astype('int')
        new_names.append(new_n)
    v_add = {'old_y_name': v_dict['y_name'], 'orf_y_name': new_names}
    v_dict.update(v_add)
    gp.delete_file_if_exists(csv_file1)
    gp.delete_file_if_exists(csv_file2)
    df1.to_csv(csv_file1, index=False)
    df2.to_csv(csv_file2, index=False)
    if c_dict['desc_stat'] and c_dict['with_output']:
        print('=' * 80)
        print('New binary outcome variabes for ORF')
        print(count)
        gp.print_descriptive_stats_file(csv_file1, new_names,
                                        c_dict['print_to_file'])
        gp.print_descriptive_stats_file(csv_file2, new_names,
                                        c_dict['print_to_file'])
    return c_dict, v_dict
