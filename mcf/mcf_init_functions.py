"""Created on Fri Apr  3 11:05:15 2020.

Contains the functions needed for the running all parts of the programme
@author: MLechner
-*- coding: utf-8 -*-
"""
import copy
import math
from pathlib import Path
import os
import numpy as np
import pandas as pd
import psutil
from mcf import general_purpose as gp


def make_user_variable(
    id_name, cluster_name, w_name, d_name, y_tree_name, y_name, x_name_ord,
    x_name_unord, x_name_always_in_ord, z_name_list,
    x_name_always_in_unord, z_name_split_ord, z_name_split_unord,
    z_name_mgate, z_name_amgate, x_name_remain_ord, x_name_remain_unord,
        x_balance_name_ord, x_balance_name_unord):
    """Put variable names in dictionary."""
    def check_none(name):
        if name is None:
            return []
        return name

    variable_dict = {'id_name': check_none(id_name),
                     'cluster_name': check_none(cluster_name),
                     'w_name': check_none(w_name),
                     'd_name': check_none(d_name),
                     'y_tree_name': check_none(y_tree_name),
                     'y_name': check_none(y_name),
                     'x_name_ord': check_none(x_name_ord),
                     'x_name_unord': check_none(x_name_unord),
                     'x_name_always_in_ord': check_none(x_name_always_in_ord),
                     'z_name_list': check_none(z_name_list),
                     'x_name_always_in_unord': check_none(
                         x_name_always_in_unord),
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


def get_controls(c_dict, v_dict):
    """Large procedure to get all parameters and variables needed for MCF.

    Parameters
    ----------
    c_dict : Dictory of parameters specified by user
    v_dict : Dictory of list of variable names specified by user

    Returns
    -------
    cn: Updated dictionary of parameters
    vn: Updated list of variables.

    """
    def sub_size(n_train, subsample_share_mult, max_share):
        if subsample_share_mult is None:
            subsample_share_mult = -1
        if subsample_share_mult <= 0:
            subsample_share_mult = 1
        subsam_share = 2 * ((n_train / 2)**0.85) / (n_train / 2)
        subsam_share = min(subsam_share, 0.67)
        subsam_share = subsam_share * subsample_share_mult
        subsam_share = min(subsam_share, max_share)
        subsam_share = max(subsam_share, 1e-4)
        return subsam_share

    path_programme_run = str(Path(__file__).parent.absolute())
    if c_dict['datpfad'] is None:
        c_dict['datpfad'] = path_programme_run
    if c_dict['outpfad'] is None:
        c_dict['outpfad'] = path_programme_run + '/out'
    out_temp = c_dict['outpfad']
    for i in range(1000):
        if not c_dict['with_output']:
            break
        if os.path.isdir(out_temp):
            if not ((c_dict['with_output'] is False)
                    or (c_dict['verbose'] is False)):
                print("Directory for output %s already exists" % out_temp,
                      "A new directory is created for the output.")
            out_temp = c_dict['outpfad'] + str(i)
        else:
            try:
                os.mkdir(out_temp)
            except OSError as oserr:
                raise Exception(
                    "Creation of the directory %s failed" % out_temp
                    ) from oserr
            else:
                if not ((c_dict['with_output'] is False)
                        or (c_dict['verbose'] is False)):
                    print("Successfully created the directory %s"
                          % out_temp)
                if out_temp != c_dict['outpfad']:
                    c_dict['outpfad'] = out_temp
                break
    if (c_dict['indata'] is None) and c_dict['train_mcf']:
        raise Exception('Filename of indata must be specified')
    if (c_dict['preddata'] is None) and c_dict['train_mcf']:
        c_dict['preddata'] = c_dict['indata']
    if c_dict['outfiletext'] is None:
        if c_dict['train_mcf']:
            c_dict['outfiletext'] = c_dict['indata']
        else:
            c_dict['outfiletext'] = c_dict['preddata']
    cnew_dict = copy.deepcopy(c_dict)

    temppfad = c_dict['outpfad'] + '/_tempmcf_'
    if os.path.isdir(temppfad):
        file_list = os.listdir(temppfad)
        if file_list:
            for temp_file in file_list:
                os.remove(os.path.join(temppfad, temp_file))
        if not ((c_dict['with_output'] is False)
                or (c_dict['verbose'] is False)):
            print("Temporary directory  %s already exists" % temppfad)
            if file_list:
                print('All files deleted.')
    else:
        try:
            os.mkdir(temppfad)
        except OSError as oserr:
            raise Exception("Creation of the directory %s failed" % temppfad
                            ) from oserr
        else:
            if not ((c_dict['with_output'] is False)
                    or (c_dict['verbose'] is False)):
                print("Successfully created the directory %s" % temppfad)
    fig_pfad_jpeg = c_dict['outpfad'] + '/' + 'fig_jpeg'
    fig_pfad_csv = c_dict['outpfad'] + '/' + 'fig_csv'
    fig_pfad_pdf = c_dict['outpfad'] + '/' + 'fig_pdf'
    if c_dict['pred_mcf'] and c_dict['with_output']:
        if not os.path.isdir(fig_pfad_jpeg):
            os.mkdir(fig_pfad_jpeg)
        if not os.path.isdir(fig_pfad_csv):
            os.mkdir(fig_pfad_csv)
        if not os.path.isdir(fig_pfad_pdf):
            os.mkdir(fig_pfad_pdf)
    if c_dict['train_mcf']:
        pred_sample_with_pred = (c_dict['outpfad'] + '/' + c_dict['indata']
                                 + 'Predpred' + '.csv')
    else:
        pred_sample_with_pred = (c_dict['outpfad'] + '/' + c_dict['preddata']
                                 + 'Predpred' + '.csv')
    cnew_dict['outfiletext'] = (c_dict['outpfad'] + '/' + c_dict['outfiletext']
                                + '.txt')
    # out_allindat = c_dict['outpfad'] + '/' + c_dict['indata'] + 'Predall.csv'
    if c_dict['save_forest_files'] is None:
        save_forest_file_pickle = (
            c_dict['outpfad'] + '/' + c_dict['indata'] + '_save_pred.pickle')
        save_forest_file_csv = (
            c_dict['outpfad'] + '/' + c_dict['indata'] + '_save_pred.csv')
        save_forest_file_ps = (
            c_dict['outpfad'] + '/' + c_dict['indata'] + '_save_predps.npy')
        save_forest_file_d_train_tree = (
            c_dict['outpfad'] + '/' + c_dict['indata'] + '_save_predd.npy')
    else:
        save_forest_file_pickle = c_dict['save_forest_files'] + '.pickle'
        save_forest_file_csv = c_dict['save_forest_files'] + '.csv'
        save_forest_file_ps = c_dict['save_forest_files'] + 'ps.npy'
        save_forest_file_d_train_tree = c_dict['save_forest_files'] + 'd.npy'
    if c_dict['train_mcf']:
        cnew_dict['indata'] = (c_dict['datpfad'] + '/' + c_dict['indata']
                               + '.csv')
    if c_dict['pred_mcf']:
        cnew_dict['preddata'] = (c_dict['datpfad'] + '/' + c_dict['preddata']
                                 + '.csv')
    indat_temp = temppfad + '/' + 'indat_temp' + '.csv'
    indata2_temp = temppfad + '/' + 'indata2_temp' + '.csv'
    preddata2_temp = temppfad + '/' + 'preddata2_temp' + '.csv'
    preddata3_temp = temppfad + '/' + 'preddata3_temp' + '.csv'
    off_support_temp = temppfad + '/' + 'off_support_temp' + '.csv'
    pred_eff_temp = temppfad + '/' + 'pred_eff_temp' + '.csv'
    fs_sample_tmp = temppfad + '/' + 'fs_sample_tmp' + '.csv'
    tree_sample_tmp = temppfad + '/' + 'tree_sample_tmp' + '.csv'
    fill_y_sample_tmp = temppfad + '/' + 'fill_y_sample_tmp' + '.csv'
    tree_sample_nn = temppfad + '/' + 'tree_sample_NN' + '.csv'
    temporary_file = temppfad + '/' + 'temporaryfile' + '.csv'
    nonlc_sample = temppfad + '/' + 'nonlc_sample' + '.csv'
    lc_sample = temppfad + '/' + 'lc_sample' + '.csv'
    cnew_dict['desc_stat'] = not c_dict['desc_stat'] is False
    if c_dict['w_yes'] is True:  # Weighting
        cnew_dict['w_yes'] = True
    else:
        c_dict['w_yes'] = False
        cnew_dict['w_yes'] = False

    if c_dict['output_type'] is None:
        c_dict['output_type'] = 2
    if c_dict['output_type'] == 0:
        print_to_file = False
        print_to_terminal = True
    elif c_dict['output_type'] == 1:
        print_to_file = True
        print_to_terminal = False
    else:
        print_to_file = True
        print_to_terminal = True
    cnew_dict['verbose'] = not c_dict['verbose'] is False
    if c_dict['screen_covariates'] is False:
        cnew_dict['screen_covariates'] = False
    else:
        cnew_dict['screen_covariates'] = True
    if c_dict['check_perfectcorr'] is False:
        cnew_dict['check_perfectcorr'] = False
    else:
        cnew_dict['check_perfectcorr'] = True
    if c_dict['min_dummy_obs'] is None:
        c_dict['min_dummy_obs'] = -1
    if c_dict['min_dummy_obs'] < 1:
        cnew_dict['min_dummy_obs'] = 10
    else:
        cnew_dict['min_dummy_obs'] = round(c_dict['min_dummy_obs'])
    if c_dict['clean_data_flag'] is False:
        cnew_dict['clean_data_flag'] = False
    else:
        cnew_dict['clean_data_flag'] = True
    if c_dict['alpha_reg_min'] is None:
        c_dict['alpha_reg_min'] = -1
    if (c_dict['alpha_reg_min'] < 0) or (c_dict['alpha_reg_min'] >= 0.4):
        cnew_dict['alpha_reg_min'] = 0.1
        if c_dict['alpha_reg_min'] >= 0.4:
            print('Values of 0.4 and larger do not make sense for alpha',
                  'regularisation. alpha regularisation parameter set to ',
                  'default value. {:8.3f}'.format(cnew_dict['alpha_reg_min']))
    else:
        cnew_dict['alpha_reg_min'] = c_dict['alpha_reg_min']
    if c_dict['alpha_reg_max'] is None:
        c_dict['alpha_reg_max'] = -1
    if (c_dict['alpha_reg_max'] < 0) or (c_dict['alpha_reg_max'] >= 0.4):
        cnew_dict['alpha_reg_max'] = 0.1
        if c_dict['alpha_reg_max'] >= 0.4:
            print('Values of 0.4 and larger do not make sense for alpha',
                  'regularisation. alpha regularisation parameter set to ',
                  'default value. {:8.3f}'.format(cnew_dict['alpha_reg_max']))
    else:
        cnew_dict['alpha_reg_max'] = c_dict['alpha_reg_max']
    if c_dict['alpha_reg_grid'] is None:
        cnew_dict['alpha_reg_grid'] = 1
    else:
        cnew_dict['alpha_reg_grid'] = round(c_dict['alpha_reg_grid'])
    if cnew_dict['alpha_reg_grid'] < 1:
        cnew_dict['alpha_reg_grid'] = 1
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
        cnew_dict['alpha_reg_grid'] = len(alpha_reg)
    if c_dict['share_forest_sample'] is None:
        cnew_dict['share_forest_sample'] = 0.5
    else:
        if (c_dict['share_forest_sample'] < 0.01) or (
                c_dict['share_forest_sample'] > 0.99):
            cnew_dict['share_forest_sample'] = 0.5
    if c_dict['panel_data'] is True:
        cnew_dict['cluster_std'] = True
        c_dict['cluster_std'] = True
        if c_dict['panel_in_rf'] is False:
            cnew_dict['panel_in_rf'] = False
        else:  # default
            cnew_dict['panel_in_rf'] = True
        panel = True
    else:
        cnew_dict['panel_data'] = False
        panel = False
        cnew_dict['panel_in_rf'] = False
    if c_dict['train_mcf']:
        data_train = pd.read_csv(cnew_dict['indata'], header=0)
    else:
        data_train = pd.read_csv(save_forest_file_csv, header=0)
    v_dict['d_name'][0] = gp.adjust_var_name(v_dict['d_name'][0],
                                             data_train.columns.tolist())
    d_dat_pd = data_train[v_dict['d_name']].squeeze()
    if d_dat_pd.dtype == 'object':
        d_dat_pd = d_dat_pd.astype('category')
        print(d_dat_pd.cat.categories)
        if cnew_dict['with_output']:
            print('Automatic recoding of treatment variable')
            numerical_codes = pd.unique(d_dat_pd.cat.codes)
            print(numerical_codes)
        d_dat_pd = d_dat_pd.cat.codes
    d_dat = d_dat_pd.to_numpy()
    d_values = np.unique(d_dat)
    d_values = np.int16(np.round(d_values))
    d_values = d_values.tolist()
    no_of_treatments = len(d_values)     # Number of treatments
    if c_dict['boot'] is None:
        c_dict['boot'] = -1
    if c_dict['boot'] > 0:   # Number of bootstrap replications
        cnew_dict['boot'] = round(c_dict['boot'])
    else:
        cnew_dict['boot'] = 1000
    if c_dict['train_mcf']:
        if c_dict['save_forest'] is True:
            cnew_dict['save_forest'] = True
        else:
            cnew_dict['save_forest'] = False
        n_train = len(data_train.index)
        if c_dict['mp_type_vim'] != 1 and c_dict['mp_type_vim'] != 2:
            if n_train < 20000:
                cnew_dict['mp_type_vim'] = 1  # MP over var's, fast, lots RAM
            else:
                cnew_dict['mp_type_vim'] = 2  # MP over bootstraps.
        vcount = data_train.groupby(v_dict['d_name']).size()
        n_d = vcount.to_numpy()         # Number of obs in treatments
        if n_train != n_d.sum():
            raise Exception(
                "Counting treatments does not work. Stop treatment")
        n_d_subsam = n_d.min() * cnew_dict['share_forest_sample']
        if c_dict['n_min_min'] is None:
            c_dict['n_min_min'] = -1
            cnew_dict['n_min_min'] = -1
        else:
            cnew_dict['n_min_min'] = round(c_dict['n_min_min'])   # grid n_min
        if cnew_dict['n_min_min'] < 1:
            if cnew_dict['n_min_min'] == -1:
                cnew_dict['n_min_min'] = round((n_d_subsam**0.4)/10)
                if cnew_dict['n_min_min'] < 5:
                    cnew_dict['n_min_min'] = 5
            else:
                cnew_dict['n_min_min'] = round((n_d_subsam**0.4)/20)
                if cnew_dict['n_min_min'] < 3:
                    cnew_dict['n_min_min'] = 3
        if cnew_dict['n_min_min'] < len(d_values):
            cnew_dict['n_min_min'] = len(d_values)
            text_to_print = ('Required to have support. Smallest ',
                             ' leafsize set to: ', cnew_dict['n_min_min'])
        else:
            text_to_print = None
        if c_dict['n_min_max'] is None:
            c_dict['n_min_max'] = -1
            cnew_dict['n_min_max'] = -1
        else:
            cnew_dict['n_min_max'] = round(c_dict['n_min_max'])
        if cnew_dict['n_min_max'] < 1:
            if cnew_dict['n_min_max'] == -1:
                cnew_dict['n_min_max'] = round(math.sqrt(n_d_subsam) / 5)
                if cnew_dict['n_min_max'] < 5:
                    cnew_dict['n_min_max'] = 5
            else:
                cnew_dict['n_min_max'] = round(math.sqrt(n_d_subsam) / 10)
                if cnew_dict['n_min_max'] < 3:
                    cnew_dict['n_min_max'] = 3
        if c_dict['n_min_grid'] is None:
            cnew_dict['n_min_grid'] = 0
        else:
            cnew_dict['n_min_grid'] = round(c_dict['n_min_grid'])
        if cnew_dict['n_min_grid'] < 1:
            cnew_dict['n_min_grid'] = 1
        if cnew_dict['n_min_min'] == cnew_dict['n_min_max']:
            cnew_dict['n_min_grid'] = 1
        elif cnew_dict['n_min_max'] < cnew_dict['n_min_min']:
            cnew_dict['n_min_grid'] = 1
            cnew_dict['n_min_max'] = cnew_dict['n_min_min']
        if cnew_dict['n_min_grid'] == 1:
            n_min = round(cnew_dict['n_min_min'])
        elif cnew_dict['n_min_grid'] == 2:
            n_min = np.hstack((cnew_dict['n_min_min'], cnew_dict['n_min_max']))
        else:
            n_min = np.linspace(cnew_dict['n_min_min'], cnew_dict['n_min_max'],
                                cnew_dict['n_min_grid'])
            n_min = list(np.unique(np.round(n_min)))
            cnew_dict['n_min_grid'] = len(n_min)
        # Select grid for number of parameters
        if c_dict['m_min_share'] is None:
            c_dict['m_min_share'] = -1
        if (c_dict['m_min_share'] <= 0) or (c_dict['m_min_share'] > 1):
            cnew_dict['m_min_share'] = 0.1   # Hastie et al suggest p/3,sqrt(p)
            if c_dict['m_min_share'] == -2:  # around page 588
                cnew_dict['m_min_share'] = 0.2
        if c_dict['m_max_share'] is None:
            c_dict['m_max_share'] = -1
        if (c_dict['m_max_share'] <= 0) or (c_dict['m_max_share'] > 1):
            cnew_dict['m_max_share'] = 0.66
            if c_dict['m_max_share'] == -2:
                cnew_dict['m_max_share'] = 0.8
        # Instead of selecting fixed number, select mean of Poisson distrib.
        if c_dict['m_random_poisson'] is False:
            cnew_dict['m_random_poisson'] = False
        else:
            cnew_dict['m_random_poisson'] = True
        if c_dict['m_grid'] is None:
            c_dict['m_grid'] = -1
        if c_dict['m_grid'] < 1:  # values of grid for m
            cnew_dict['m_grid'] = 2
        else:
            cnew_dict['m_grid'] = round(c_dict['m_grid'])
        if c_dict['max_cats_cont_vars'] is None:
            c_dict['max_cats_cont_vars'] = -1
        if c_dict['max_cats_cont_vars'] < 1:
            cnew_dict['max_cats_cont_vars'] = n_train + 1
        else:
            cnew_dict['max_cats_cont_vars'] = round(
                c_dict['max_cats_cont_vars'])
        if c_dict['stop_empty'] is None:
            c_dict['stop_empty'] = -1
        if c_dict['stop_empty'] < 1:
            cnew_dict['stop_empty'] = 1
        else:
            cnew_dict['stop_empty'] = round(c_dict['stop_empty'])
        if c_dict['random_thresholds'] is None:
            c_dict['random_thresholds'] = -1
        if c_dict['random_thresholds'] < 0:     # Saves computation time
            cnew_dict['random_thresholds'] = round(4 + n_train**0.2)
            # Feature preselection
        cnew_dict['fs_yes'] = bool(c_dict['fs_yes'])
        if c_dict['fs_rf_threshold'] is None:
            c_dict['fs_rf_threshold'] = -1
        if c_dict['fs_rf_threshold'] <= 0:
            cnew_dict['fs_rf_threshold'] = 0
        if c_dict['fs_other_sample'] is False:
            cnew_dict['fs_other_sample'] = False
        else:
            cnew_dict['fs_other_sample'] = True
        if c_dict['fs_other_sample_share'] is None:
            c_dict['fs_other_sample_share'] = -1
        if (c_dict['fs_other_sample_share'] < 0) or (
                c_dict['fs_other_sample_share'] > 0.5):
            cnew_dict['fs_other_sample_share'] = 0.2
        if cnew_dict['fs_other_sample'] is False or (
                cnew_dict['fs_yes'] is False):
            cnew_dict['fs_other_sample_share'] = 0

        # size of subsampling samples         n/2: size of forest sample
        subsam_share_forest = sub_size(
            n_train, c_dict['subsample_factor_forest'], 0.67)
        if c_dict['subsample_factor_eval'] is None:
            c_dict['subsample_factor_eval'] = False  # Default
        if c_dict['subsample_factor_eval'] is False:
            c_dict['subsample_factor_eval'] = 1000000000
        if c_dict['subsample_factor_eval'] is True:
            c_dict['subsample_factor_eval'] = 1
        if c_dict['subsample_factor_eval'] < 0.01:
            c_dict['subsample_factor_eval'] = 1000000000
        subsam_share_eval = sub_size(
            n_train, c_dict['subsample_factor_eval'], 1)
        # Define method to be used later on
        if c_dict['mce_vart'] is None:
            c_dict['mce_vart'] = -1
        if (c_dict['mce_vart'] == 1) or (c_dict['mce_vart'] == -1):
            mtot = 1
            mtot_no_mce = 0        # MSE + MCE rule
            estimator_str = 'MSE & MCE'
        elif c_dict['mce_vart'] == 2:
            mtot = 2
            mtot_no_mce = 1        # -Var(treatment effect) rule
            estimator_str = '-Var(effect)'
        elif c_dict['mce_vart'] == 0:
            mtot = 3
            mtot_no_mce = 1        # MSE rule
            estimator_str = 'MSE'
        elif c_dict['mce_vart'] == 3:
            mtot = 4               # MSE + MCE rule or penalty function rule
            mtot_no_mce = 0        # (randomly decided)
            estimator_str = 'MSE,MCE or penalty (random)'
        else:
            raise Exception('Inconsistent MTOT definition of  MCE_VarT.')
        if c_dict['mtot_p_diff_penalty'] is None:
            c_dict['mtot_p_diff_penalty'] = -1
        if c_dict['mtot_p_diff_penalty'] == 0:
            if mtot == 4:
                mtot = 1  # No random mixing of rules; prob of MSE+MCE rule== 1
        elif c_dict['mtot_p_diff_penalty'] < 0:  # Default
            if mtot == 4:
                cnew_dict['mtot_p_diff_penalty'] = 0.5
            else:                                   # Approx 1 for N = 1000
                cnew_dict['mtot_p_diff_penalty'] = (
                    2 * ((n_train * subsam_share_forest)**0.9)
                    / (n_train * subsam_share_forest))
                if mtot == 2:
                    cnew_dict['mtot_p_diff_penalty'] = 100 * cnew_dict[
                        'mtot_p_diff_penalty']
                cnew_dict['mtot_p_diff_penalty'] *= np.sqrt(
                    no_of_treatments * (no_of_treatments-1)/2)
        else:
            if mtot == 4:
                if c_dict['mtot_p_diff_penalty'] > 1:  # if accidently scaled %
                    cnew_dict['mtot_p_diff_penalty'] = c_dict[
                        'mtot_p_diff_penalty'] / 100
                    print('Probability of using p-score larger than 1.',
                          'Devided by 100.')
                if cnew_dict['mtot_p_diff_penalty'] > 1:
                    raise Exception('Probability of using p-score larger '
                                    + 'than 1. Programm terminated.')
        if cnew_dict['mtot_p_diff_penalty']:
            estimator_str += ' penalty fct'
        if c_dict['l_centering'] is not False:
            cnew_dict['l_centering'] = True
            if c_dict['l_centering_share'] is None:
                c_dict['l_centering_share'] = -1
            if c_dict['l_centering_cv_k'] is None:
                c_dict['l_centering_cv_k'] = -1
            if c_dict['l_centering_new_sample'] is True:
                cnew_dict['l_centering_new_sample'] = True
            else:
                cnew_dict['l_centering_new_sample'] = False
            if cnew_dict['l_centering_new_sample']:
                if not 0.0999 < c_dict['l_centering_share'] < 0.9001:
                    cnew_dict['l_centering_share'] = 0.25
            else:
                if c_dict['l_centering_cv_k'] < 1:
                    cnew_dict['l_centering_cv_k'] = 5
                else:
                    cnew_dict['l_centering_cv_k'] = int(
                        round(c_dict['l_centering_cv_k']))
            if c_dict['l_centering_uncenter'] is not True:
                cnew_dict['l_centering_uncenter'] = False
        else:
            cnew_dict['l_centering'] = False
            cnew_dict['l_centering_new_sample'] = None
            cnew_dict['l_centering_cv_k'] = None
            cnew_dict['l_centering_uncenter'] = None
        if c_dict['match_nn_prog_score'] is False:
            cnew_dict['match_nn_prog_score'] = False
        else:
            cnew_dict['match_nn_prog_score'] = True
        if c_dict['nn_main_diag_only'] is True:
            cnew_dict['nn_main_diag_only'] = True
        else:
            cnew_dict['nn_main_diag_only'] = False
        if c_dict['no_ray_in_forest_building'] is True:
            cnew_dict['no_ray_in_forest_building'] = True
        else:
            cnew_dict['no_ray_in_forest_building'] = False
        if c_dict['max_cats_z_vars'] is None:
            c_dict['max_cats_z_vars'] = -1
        if c_dict['max_cats_z_vars'] < 1:
            cnew_dict['max_cats_z_vars'] = round(n_train ** 0.3)
        else:
            cnew_dict['max_cats_z_vars'] = round(c_dict['max_cats_z_vars'])
        if c_dict['var_import_oob'] is True:
            cnew_dict['var_import_oob'] = True
        else:                    # importance measure
            cnew_dict['var_import_oob'] = False
    else:
        n_min = None
        subsam_share_eval = None
        subsam_share_forest = None
        mtot = None
        mtot_no_mce = None
        estimator_str = 'Estimates loaded.'
    if c_dict['pred_mcf']:
        if c_dict['cluster_std'] is True:
            cnew_dict['cluster_std'] = True
        else:
            cnew_dict['cluster_std'] = False
        if not (cnew_dict['cluster_std'] or panel):
            cnew_dict['cluster_std'] = False
        data_pred = pd.read_csv(cnew_dict['preddata'], header=0)
        n_pred = len(data_pred.index)
        # Choice based sampling (oversampling of treatments)
        if c_dict['choice_based_yes'] is True:
            cnew_dict['choice_based_yes'] = True
            if len(c_dict['choice_based_probs']) == no_of_treatments:
                if all(v > 0 for v in c_dict['choice_based_probs']):
                    pcb = np.array(c_dict['choice_based_probs'])
                    pcb = pcb / np.sum(pcb) * no_of_treatments
                    cnew_dict['choice_based_probs'] = pcb.tolist()
                    if cnew_dict['with_output'] and cnew_dict['verbose']:
                        print('Choice based sampling active. Normalized',
                              'choice probabilites for', 'treatments:',
                              cnew_dict['choice_based_probs'])
                else:
                    raise Exception('Choice based sampling active. Not'
                                    + 'possible to have zero or negative'
                                    + ' choice probability.')
            else:
                raise Exception('Choice based sampling active. Rows in choice'
                                + ' probabilities does not correspond to'
                                + ' number of treatments.')
        else:
            cnew_dict['choice_based_yes'] = False
            cnew_dict['choice_based_probs'] = 1
        if c_dict['max_weight_share'] is None:
            c_dict['max_weight_share'] = -1
        if c_dict['max_weight_share'] <= 0:
            cnew_dict['max_weight_share'] = 0.05
        # Parameters for the variance estimation
        if c_dict['cond_var'] is False:    # False: variance est. uses Var(wY)
            cnew_dict['cond_var'] = False  # True: cond. mean & variances used
        else:
            cnew_dict['cond_var'] = True
        cnew_dict['knn'] = c_dict['knn'] is not False
        if c_dict['knn_min_k'] is None:
            c_dict['knn_min_k'] = -1
        if c_dict['knn_min_k'] < 0:     # minimum number of neighbours in k-NN
            cnew_dict['knn_min_k'] = 10
        if c_dict['knn_const'] is None:
            c_dict['knn_const'] = -1
        if c_dict['knn_const'] < 0:  # k: const. in # of neighbour estimation
            cnew_dict['knn_const'] = 1
        if c_dict['nw_bandw'] is None:
            c_dict['nw_bandw'] = -1
        if c_dict['nw_bandw'] < 0:   # bandwidth for NW estimat. multiplier of
            cnew_dict['nw_bandw'] = 1  # silverman's optimal bandwidth
        if c_dict['nw_kern'] is None:
            c_dict['nw_kern'] = -1
        if c_dict['nw_kern'] != 2:  # kernel for NW: 1: Epanechikov 2: Normal
            cnew_dict['nw_kern'] = 1
        bnr = 199
        if c_dict['se_boot_ate'] is None:
            c_dict['se_boot_ate'] = bnr if cnew_dict['cluster_std'] else False
        if c_dict['se_boot_gate'] is None:
            c_dict['se_boot_gate'] = bnr if cnew_dict['cluster_std'] else False
        if c_dict['se_boot_iate'] is None:
            c_dict['se_boot_iate'] = 199 if cnew_dict['cluster_std'] else False

        if 0 < c_dict['se_boot_ate'] < 49:   # This includes False
            cnew_dict['se_boot_ate'] = bnr
        elif c_dict['se_boot_ate'] >= 49:
            cnew_dict['se_boot_ate'] = int(c_dict['se_boot_ate'])
        else:
            cnew_dict['se_boot_ate'] = 0
        if 0 < c_dict['se_boot_gate'] < 49:   # This includes False
            cnew_dict['se_boot_gate'] = bnr
        elif c_dict['se_boot_gate'] >= 49:
            cnew_dict['se_boot_gate'] = int(c_dict['se_boot_gate'])
        else:
            cnew_dict['se_boot_gate'] = 0
        if 0 < c_dict['se_boot_iate'] < 49:   # This includes False
            cnew_dict['se_boot_iate'] = bnr
        elif c_dict['se_boot_iate'] >= 49:
            cnew_dict['se_boot_iate'] = int(c_dict['se_boot_iate'])
        else:
            cnew_dict['se_boot_iate'] = 0
        # Balancing test based on weights
        if c_dict['balancing_test_w'] is False:
            cnew_dict['balancing_test_w'] = False
        else:
            cnew_dict['balancing_test_w'] = True
        if (v_dict['x_balance_name_ord'] == []) and (
                v_dict['x_balance_name_unord'] == []):
            cnew_dict['balancing_test_w'] = False
        cnew_dict['atet_flag'] = bool(c_dict['atet_flag'])
        if c_dict['gatet_flag'] is True:
            cnew_dict['gatet_flag'] = True
            cnew_dict['atet_flag'] = True  # GTET with Z=D will be computed
        else:
            cnew_dict['gatet_flag'] = False
        if c_dict['gmate_sample_share'] is None:
            c_dict['gmate_sample_share'] = -1
        if c_dict['gmate_sample_share'] <= 0:
            if n_pred < 1000:
                cnew_dict['gmate_sample_share'] = 1
            else:
                cnew_dict['gmate_sample_share'] = (
                    1000 + ((n_pred-1000) ** 0.75)) / n_pred
        if c_dict['gmate_no_evaluation_points'] is None:
            c_dict['gmate_no_evaluation_points'] = -1
        if c_dict['gmate_no_evaluation_points'] < 2:
            cnew_dict['gmate_no_evaluation_points'] = 50
        else:
            cnew_dict['gmate_no_evaluation_points'] = round(
                c_dict['gmate_no_evaluation_points'])
        if c_dict['smooth_gates'] is False:
            cnew_dict['smooth_gates'] = False
        else:
            cnew_dict['smooth_gates'] = True
        if c_dict['sgates_bandwidth'] is None:
            c_dict['sgates_bandwidth'] = -1
        if c_dict['sgates_bandwidth'] <= 0:
            cnew_dict['sgates_bandwidth'] = 1
        else:
            cnew_dict['sgates_bandwidth'] = c_dict['sgates_bandwidth']
        if c_dict['sgates_no_evaluation_points'] is None:
            c_dict['sgates_no_evaluation_points'] = -1
        if c_dict['sgates_no_evaluation_points'] < 2:
            cnew_dict['sgates_no_evaluation_points'] = 50
        else:
            cnew_dict['sgates_no_evaluation_points'] = round(
                c_dict['sgates_no_evaluation_points'])
        if c_dict['iate_flag'] is False:
            cnew_dict['iate_flag'] = False
            cnew_dict['iate_se_flag'] = False
            c_dict['post_est_stats'] = False
        else:
            cnew_dict['iate_flag'] = True
            if c_dict['iate_se_flag'] is False:
                cnew_dict['iate_se_flag'] = False
            else:
                cnew_dict['iate_se_flag'] = True
        # Post estimation parameters
        if c_dict['post_est_stats'] is False:
            cnew_dict['post_est_stats'] = False
        else:
            cnew_dict['post_est_stats'] = True
        if c_dict['bin_corr_yes'] is False:    # Checking for binary predict's
            cnew_dict['bin_corr_yes'] = False
        else:
            cnew_dict['bin_corr_yes'] = True
        if c_dict['bin_corr_thresh'] is None:
            c_dict['bin_corr_thresh'] = -1
        if c_dict['bin_corr_thresh'] < 0 or c_dict['bin_corr_thresh'] > 1:
            cnew_dict['bin_corr_thresh'] = 0.1  # Minimum threshhold of abs.
        else:                                   # correlation to be displayed
            cnew_dict['bin_corr_thresh'] = c_dict['bin_corr_thresh']
        if c_dict['post_plots'] is False:
            cnew_dict['post_plots'] = False
        else:
            cnew_dict['post_plots'] = True
        if c_dict['relative_to_first_group_only'] is False:
            cnew_dict['relative_to_first_group_only'] = False
        else:
            cnew_dict['relative_to_first_group_only'] = True
        if c_dict['post_km'] is False:
            cnew_dict['post_km'] = False
        else:
            cnew_dict['post_km'] = True
        if c_dict['post_km_no_of_groups'] is None:
            c_dict['post_km_no_of_groups'] = [-1]
        if isinstance(c_dict['post_km_no_of_groups'], (int, float)):
            c_dict['post_km_no_of_groups'] = [c_dict['post_km_no_of_groups']]
        if len(c_dict['post_km_no_of_groups']) == 1:
            if c_dict['post_km_no_of_groups'][0] < 2:
                if n_pred < 10000:
                    middle = 5
                elif n_pred > 100000:
                    middle = 10
                else:
                    middle = 5 + int(round(n_pred/20000))
                if middle < 7:
                    cnew_dict['post_km_no_of_groups'] = [
                        middle-2, middle-1, middle, middle+1, middle+2]
                else:
                    cnew_dict['post_km_no_of_groups'] = [
                        middle-4, middle-2, middle, middle+2, middle+4]
            else:
                cnew_dict['post_km_no_of_groups'] = [round(
                    int(c_dict['post_km_no_of_groups']))]
        else:
            if not isinstance(c_dict['post_km_no_of_groups'], list):
                c_dict['post_km_no_of_groups'] = list(
                        cnew_dict['post_km_no_of_groups'])
                cnew_dict['post_km_no_of_groups'] = [
                    round(int(a)) for a in c_dict['post_km_no_of_groups']]
        if c_dict['post_km_replications'] is None:
            c_dict['post_km_replications'] = -1
        if c_dict['post_km_replications'] < 0:
            cnew_dict['post_km_replications'] = 10
        else:
            cnew_dict['post_km_replications'] = round(
                c_dict['post_km_replications'])
        if c_dict['post_kmeans_max_tries'] is None:
            cnew_dict['post_kmeans_max_tries'] = 1000
        if cnew_dict['post_kmeans_max_tries'] < 10:
            cnew_dict['post_kmeans_max_tries'] = 10
        add_pred_to_data_file = bool(cnew_dict['post_est_stats'])
        if c_dict['post_random_forest_vi'] is False:
            cnew_dict['post_random_forest_vi'] = False
        else:
            cnew_dict['post_random_forest_vi'] = True
        q_w = [0.5, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]  # Weight analysis
        if cnew_dict['fig_fontsize'] is None:
            cnew_dict['fig_fontsize'] = -1
        if cnew_dict['fig_fontsize'] > 0.5 and cnew_dict['fig_fontsize'] < 7.5:
            cnew_dict['fig_fontsize'] = round(cnew_dict['fig_fontsize'])
        else:
            cnew_dict['fig_fontsize'] = 2
        all_fonts = ('xx-small', 'x-small', 'small', 'medium', 'large',
                     'x-large', 'xx-large')
        for i, i_lab in enumerate(all_fonts):
            if cnew_dict['fig_fontsize'] == i + 1:
                cnew_dict['fig_fontsize'] = i_lab
        if c_dict['fig_dpi'] is None:
            c_dict['fig_dpi'] = -1
        if c_dict['fig_dpi'] < 10:
            cnew_dict['fig_dpi'] = 500
        else:
            cnew_dict['fig_dpi'] = round(cnew_dict['fig_dpi'])
        if c_dict['fig_ci_level'] is None:
            c_dict['fig_ci_level'] = -1
        if c_dict['fig_ci_level'] < 0.5 or c_dict['fig_ci_level'] > 0.999999:
            cnew_dict['fig_ci_level'] = 0.90
        if c_dict['no_filled_plot'] is None:
            c_dict['no_filled_plot'] = -1
        if c_dict['no_filled_plot'] < 5:
            cnew_dict['no_filled_plot'] = 20
        else:
            cnew_dict['no_filled_plot'] = round(c_dict['no_filled_plot'])
        if c_dict['show_plots'] is False:
            cnew_dict['show_plots'] = False
        else:
            cnew_dict['show_plots'] = True
        if c_dict['mp_type_weights'] != 2:
            cnew_dict['mp_type_weights'] = 1  # MP over obs, fast, lots of mem
        else:
            cnew_dict['mp_type_weights'] = 2
        if c_dict['weight_as_sparse'] is False:
            cnew_dict['weight_as_sparse'] = False
        else:
            cnew_dict['weight_as_sparse'] = True
        if c_dict['mp_weights_tree_batch'] is None:
            c_dict['mp_weights_tree_batch'] = -1
        if c_dict['mp_weights_tree_batch'] < 1:
            cnew_dict['mp_weights_tree_batch'] = 0
        else:
            cnew_dict['mp_weights_tree_batch'] = round(
                c_dict['mp_weights_tree_batch'])
    else:
        all_fonts = None
        add_pred_to_data_file = None
        q_w = None
    if c_dict['common_support'] is None:
        c_dict['common_support'] = -1
    if c_dict['common_support'] != 0 and c_dict['common_support'] != 2:
        cnew_dict['common_support'] = 1      # Default values
    if cnew_dict['common_support'] > 0:
        if c_dict['support_quantil'] is None:
            c_dict['support_quantil'] = -1
        if c_dict['support_quantil'] < 0 or c_dict['support_quantil'] > 1:
            cnew_dict['support_quantil'] = 1
        if c_dict['support_min_p'] is None:
            c_dict['support_min_p'] = -1
        if c_dict['support_min_p'] < 0 or c_dict['support_min_p'] > 0.5:
            cnew_dict['support_min_p'] = 0.01
    if c_dict['support_max_del_train'] is None:
        c_dict['support_max_del_train'] = -1
    if not 0 < c_dict['support_max_del_train'] <= 1:
        cnew_dict['support_max_del_train'] = 0.33
    if c_dict['mp_with_ray'] is False:
        cnew_dict['mp_with_ray'] = False
    else:
        cnew_dict['mp_with_ray'] = True
    memory = psutil.virtual_memory()
    if c_dict['mp_ray_objstore_multiplier'] is None:
        c_dict['mp_ray_objstore_multiplier'] = -1
    # Obs., and number of trees as determinants of obj.store when forest build
    if c_dict['train_mcf']:
        min_obj_str_n_1 = (n_train / 60000) * (cnew_dict['boot'] / 1000) * (
            cnew_dict['m_grid'] * cnew_dict['n_min_grid'] * cnew_dict[
                'alpha_reg_grid'] / 12) * (120 * 1024 * 1024 * 1024) * 5
        min_obj_str_n_2 = (n_train / 60000) * (cnew_dict['boot'] / 1000) * (
            120 * 1024 * 1024 * 1024)
        mem_object_store_1 = min(memory.available*0.5, min_obj_str_n_1)
        mem_object_store_2 = min(memory.available*0.5, min_obj_str_n_2)
        if c_dict['mp_ray_objstore_multiplier'] > 0:
            mem_object_store_1 *= c_dict['mp_ray_objstore_multiplier']
            mem_object_store_2 *= c_dict['mp_ray_objstore_multiplier']
        if mem_object_store_1 > 0.7 * memory.available:
            mem_object_store_1 = 0.7 * memory.available
        if mem_object_store_2 > 0.5 * memory.available:
            mem_object_store_2 = 0.5 * memory.available
        if n_train < 20000:
            mem_object_store_1 = None             # Forest
            mem_object_store_2 = None             # Variable importance
    else:
        mem_object_store_1 = None
        mem_object_store_2 = None
        n_train = 0
    if c_dict['pred_mcf']:
        min_obj_str_n_3 = (n_pred / 60000) * (120 * 1024 * 1024 * 1024)
        mem_object_store_3 = min(memory.available*0.5, min_obj_str_n_3)
        if c_dict['mp_ray_objstore_multiplier'] > 0:
            mem_object_store_3 *= c_dict['mp_ray_objstore_multiplier']
        if mem_object_store_3 > 0.5 * memory.available:
            mem_object_store_3 = 0.5 * memory.available
        if n_pred < 20000:
            mem_object_store_3 = None             # GATE
    else:
        mem_object_store_3 = 3
        n_pred = 0
    mp_automatic = False
    if c_dict['no_parallel'] is None:
        c_dict['no_parallel'] = -1
    if c_dict['no_parallel'] < -0.5:
        cnew_dict['no_parallel'] = psutil.cpu_count(logical=True) - 2
        mp_automatic = True
    elif -0.5 <= c_dict['no_parallel'] <= 1.5:
        cnew_dict['no_parallel'] = 1
    else:
        cnew_dict['no_parallel'] = round(c_dict['no_parallel'])
    sys_share = 0.7 * getattr(psutil.virtual_memory(), 'percent') / 100

    if c_dict['_mp_ray_del'] is None:
        cnew_dict['_mp_ray_del'] = ('refs',)
    else:
        possible_vals = ('refs', 'rest', 'none')
        if isinstance(c_dict['_mp_ray_del'], (str, list)):
            cnew_dict['_mp_ray_del'] = tuple(c_dict['_mp_ray_del'])
        if len(cnew_dict['_mp_ray_del']) > 2:
            print(cnew_dict['_mp_ray_del'])
            raise Exception('Too many parameters for _mp_ray_del')
        if not isinstance(cnew_dict['_mp_ray_del'], tuple):
            raise Exception('_mp_ray_del is no Tuple')
        if not all(i in possible_vals for i in cnew_dict['_mp_ray_del']):
            raise Exception('Wrong parameters for _mp_ray_del')
    if c_dict['_mp_ray_shutdown'] is None:
        cnew_dict['_mp_ray_shutdown'] = not ((n_train < 20000) and
                                             (n_pred < 20000))
    if cnew_dict['train_mcf'] and cnew_dict['pred_mcf'] and (
            cnew_dict['indata'] == cnew_dict['preddata']):
        if c_dict['reduce_split_sample'] is not True:
            cnew_dict['reduce_split_sample'] = False
        if c_dict['reduce_split_sample']:
            if cnew_dict['reduce_split_sample_pred_share'] is None:
                cnew_dict['reduce_split_sample_pred_share'] = 0.5
        else:
            cnew_dict['reduce_split_sample_pred_share'] = 0
    else:
        if c_dict['reduce_split_sample'] is True:
            raise Exception('No sample split possible. Training file differs'
                            + ' from prediction file (or no training and'
                            + ' prediction).')
        cnew_dict['reduce_split_sample'] = False
        cnew_dict['reduce_split_sample_pred_share'] = 0
    if not 0 <= cnew_dict['reduce_split_sample_pred_share'] <= 1:
        raise Exception('reduce_split_sample_pred_share outside [0,1]')
    if c_dict['reduce_training'] is not True:
        cnew_dict['reduce_training'] = False
        cnew_dict['reduce_training_share'] = 0
    else:
        if c_dict['reduce_training_share'] is None:
            cnew_dict['reduce_training_share'] = 0.5
    if not 0 <= cnew_dict['reduce_training_share'] <= 1:
        raise Exception('reduce_training_share outside [0,1]')
    if c_dict['reduce_prediction'] is not True:
        cnew_dict['reduce_prediction'] = False
        cnew_dict['reduce_prediction_share'] = 0
    else:
        if c_dict['reduce_prediction_share'] is None:
            cnew_dict['reduce_prediction_share'] = 0.5
    if not 0 <= cnew_dict['reduce_prediction_share'] <= 1:
        raise Exception('reduce_prediction_share outside [0,1]')
    if c_dict['reduce_largest_group_train'] is not True:
        cnew_dict['reduce_largest_group_train'] = False
        cnew_dict['reduce_largest_group_train_share'] = 0
    else:
        if c_dict['reduce_largest_group_train_share'] is None:
            cnew_dict['reduce_largest_group_train_share'] = 0.5
    if not 0 <= cnew_dict['reduce_largest_group_train_share'] <= 1:
        raise Exception('reduce_largest_group_train_share outside [0,1]')

    cn_add = {'temppfad':           temppfad,
              'fig_pfad_jpeg':      fig_pfad_jpeg,
              'fig_pfad_csv':       fig_pfad_csv,
              'fig_pfad_pdf':       fig_pfad_pdf,
              'pred_sample_with_pred': pred_sample_with_pred,
              'indat_temp':         indat_temp,
              'indata2_temp':       indata2_temp,
              'preddata2_temp':     preddata2_temp,
              'preddata3_temp':     preddata3_temp,
              'off_support_temp':   off_support_temp,
              'pred_eff_temp':      pred_eff_temp,
              'fs_sample_temp':     fs_sample_tmp,
              'tree_sample_temp':   tree_sample_tmp,
              'fill_y_sample_temp': fill_y_sample_tmp,
              'tree_sample_nn':     tree_sample_nn,
              'temporary_file':     temporary_file,
              'lc_sample':          lc_sample,
              'nonlc_sample':       nonlc_sample,
              'panel':              panel,
              'grid_n_min':         n_min,
              'title_variance':     'Weight-based variance',
              'add_pred_to_data_file': add_pred_to_data_file,
              'subsam_share_forest': subsam_share_forest,
              'subsam_share_eval':  subsam_share_eval,
              'mtot':               mtot,
              'mtot_no_mce':        mtot_no_mce,
              'd_values':           d_values,
              'no_of_treat':        no_of_treatments,
              'q_w':                q_w,
              'sys_share':          sys_share,
              'mp_automatic':       mp_automatic,
              'grid_alpha_reg':     alpha_reg,
              'print_to_terminal':  print_to_terminal,
              'print_to_file':      print_to_file,
              'mem_object_store_1': mem_object_store_1,
              'mem_object_store_2': mem_object_store_2,
              'mem_object_store_3': mem_object_store_3,
              'save_forest_file_pickle': save_forest_file_pickle,
              'save_forest_file_csv': save_forest_file_csv,
              'save_forest_file_ps': save_forest_file_ps,
              'save_forest_file_d_train_tree': save_forest_file_d_train_tree,
              'estimator_str': estimator_str}
    cnew_dict.update(cn_add)
    if c_dict['train_mcf']:
        if v_dict['y_name'] is None or v_dict['y_name'] == []:
            raise Exception('y_name must be specified.')
        if v_dict['d_name'] is None or v_dict['d_name'] == []:
            raise Exception('d_name must be specified.')
        if v_dict['y_tree_name'] is None or v_dict['y_tree_name'] == []:
            v_dict['y_tree_name'] = [v_dict['y_name'][0]]
    if (v_dict['x_name_ord'] is None or v_dict['x_name_ord'] == []) and (
            v_dict['x_name_unord'] is None or v_dict['x_name_unord'] == []):
        raise Exception('x_name_ord or x_name_unord must be specified.')
    vnew_dict = copy.deepcopy(v_dict)
    if cnew_dict['cluster_std'] or cnew_dict['panel']:
        if len(v_dict['cluster_name']) > 1:
            raise Exception('More than one name for cluster variable.')
        vnew_dict['cluster_name'] = gp.cleaned_var_names(
            vnew_dict['cluster_name'])
    else:
        vnew_dict['cluster_name'] = []
    vnew_dict['y_tree_name'] = gp.cleaned_var_names(
        vnew_dict['y_tree_name'])
    vnew_dict['y_name'].extend(vnew_dict['y_tree_name'])
    vnew_dict['y_name'] = gp.cleaned_var_names(vnew_dict['y_name'])
    vnew_dict['x_name_always_in_ord'] = gp.cleaned_var_names(
        vnew_dict['x_name_always_in_ord'])
    vnew_dict['x_name_always_in_unord'] = gp.cleaned_var_names(
        vnew_dict['x_name_always_in_unord'])
    vnew_dict['x_name_remain_ord'] = gp.cleaned_var_names(
        vnew_dict['x_name_remain_ord'])
    vnew_dict['x_name_remain_unord'] = gp.cleaned_var_names(
        vnew_dict['x_name_remain_unord'])
    vnew_dict['x_balance_name_ord'] = gp.cleaned_var_names(
        vnew_dict['x_balance_name_ord'])
    vnew_dict['x_balance_name_unord'] = gp.cleaned_var_names(
        vnew_dict['x_balance_name_unord'])
    vnew_dict['z_name_list'] = gp.cleaned_var_names(
        vnew_dict['z_name_list'])
    vnew_dict['z_name_ord'] = gp.cleaned_var_names(vnew_dict['z_name_ord'])
    vnew_dict['z_name_unord'] = gp.cleaned_var_names(
        vnew_dict['z_name_unord'])
    vnew_dict['d_name'] = gp.cleaned_var_names(vnew_dict['d_name'])
    vnew_dict['x_name_ord'] = gp.cleaned_var_names(vnew_dict['x_name_ord'])
    vnew_dict['x_name_unord'] = gp.cleaned_var_names(vnew_dict['x_name_unord'])
    if cnew_dict['w_yes'] == 0:
        vnew_dict['w_name'] = []
    else:
        if vnew_dict['w_name'] is None:
            raise Exception('No name for sample weights specified.')
        if vnew_dict['w_name'] == []:
            raise Exception('No name for sample weights specified.')
        vnew_dict['w_name'] = gp.cleaned_var_names(vnew_dict['w_name'])
    vnew_dict['id_name'] = gp.cleaned_var_names(vnew_dict['id_name'])

    x_name = copy.deepcopy(vnew_dict['x_name_ord'] + vnew_dict['x_name_unord'])
    x_name = gp.cleaned_var_names(x_name)
    x_name_in_tree = copy.deepcopy(vnew_dict['x_name_always_in_ord']
                                   + vnew_dict['x_name_always_in_unord'])
    x_name_in_tree = gp.cleaned_var_names(x_name_in_tree)
    x_balance_name = gp.cleaned_var_names(copy.deepcopy(
        vnew_dict['x_balance_name_ord']+vnew_dict['x_balance_name_unord']))
    x_name_remain = gp.cleaned_var_names(copy.deepcopy(
        vnew_dict['x_name_remain_ord'] + vnew_dict['x_name_remain_unord']
        + x_name_in_tree + x_balance_name))
    x_name_always_in = gp.cleaned_var_names(copy.deepcopy(
        vnew_dict['x_name_always_in_ord']
        + vnew_dict['x_name_always_in_unord']))
    name_ordered = gp.cleaned_var_names(copy.deepcopy(
        vnew_dict['x_name_ord'] + vnew_dict['x_name_always_in_ord']
        + vnew_dict['x_name_remain_ord']))
    name_unordered = gp.cleaned_var_names(copy.deepcopy(
        vnew_dict['x_name_unord'] + vnew_dict['x_name_always_in_unord']
        + vnew_dict['x_name_remain_unord']))
    if cnew_dict['fs_yes'] and c_dict['train_mcf']:
        if cnew_dict['balancing_test_w']:
            x_name_remain = gp.cleaned_var_names(copy.deepcopy(
                x_balance_name + x_name_remain))
    if x_name_in_tree != []:
        x_name_remain = gp.cleaned_var_names(copy.deepcopy(
            x_name_in_tree + x_name_remain))
        x_name = gp.cleaned_var_names(copy.deepcopy(x_name_in_tree + x_name))
    if not ((name_ordered == []) or (name_unordered == [])):
        if any([value for value in name_ordered if value in name_unordered]):
            raise Exception('Remove overlap in ordered + unordered variables')
    v_dict['z_name_mgate'] = gp.cleaned_var_names(
        v_dict['z_name_mgate'])
    v_dict['z_name_amgate'] = gp.cleaned_var_names(
        v_dict['z_name_amgate'])
    vnew_dict['z_name_mgate'] = list(
        set(v_dict['z_name_mgate']).intersection(x_name))
    vnew_dict['z_name_amgate'] = list(
        set(v_dict['z_name_amgate']).intersection(x_name))
    if (not vnew_dict['z_name_mgate']) and (
            not vnew_dict['z_name_amgate']):   # no names left
        marg_plots = False
    else:
        marg_plots = True
    cnew_dict.update({'marg_plots': marg_plots})
    # Define variables for consistency check in data sets
    names_to_check_train = vnew_dict['d_name'] + vnew_dict['y_name'] + x_name
    names_to_check_pred = x_name
    if ((vnew_dict['z_name_list'] == []) and (vnew_dict['z_name_ord'] == [])
            and (vnew_dict['z_name_unord'] == [])):
        cnew_dict.update({'agg_yes': 0})
        z_name = []
        cnew_dict.update({'gate_yes': False})
    else:
        cnew_dict.update({'agg_yes': 1})
        cnew_dict.update({'gate_yes': True})
        if not vnew_dict['z_name_list'] == []:
            names_to_check_train.extend(vnew_dict['z_name_list'])
            names_to_check_pred.extend(vnew_dict['z_name_list'])
        if not vnew_dict['z_name_ord'] == []:
            names_to_check_train.extend(vnew_dict['z_name_ord'])
            names_to_check_pred.extend(vnew_dict['z_name_ord'])
        if not vnew_dict['z_name_unord'] == []:
            names_to_check_train.extend(vnew_dict['z_name_unord'])
            names_to_check_pred.extend(vnew_dict['z_name_unord'])
        z_name = vnew_dict['z_name_ord'] + vnew_dict['z_name_unord']
    if (cnew_dict['atet_flag'] or cnew_dict['gatet_flag'] or
            cnew_dict['choice_based_yes']) and c_dict['pred_mcf']:
        data2 = pd.read_csv(cnew_dict['preddata'], nrows=2)
        var_names = list(data2.columns)
        var_names_up = [s.upper() for s in var_names]
        if vnew_dict['d_name'][0] not in var_names_up:
            if cnew_dict['with_output'] and c_dict['verbose'] and (
                    cnew_dict['atet_flag'] or cnew_dict['gatet_flag']):
                print('-' * 80)
                add_text = 'Treatment variable not in prediction data. '
                add_text += 'ATET and GATET cannot be computed. '
                print(add_text)
                print('-' * 80)
            else:
                add_text = ''
            cnew_dict['atet_flag'] = False
            cnew_dict['gatet_flag'] = False
            if text_to_print is not None:
                text_to_print = add_text + ' \n' + text_to_print
            else:
                text_to_print = add_text
            if cnew_dict['choice_based_yes']:
                raise Exception('Choice based sampling relates only to ' +
                                'prediction file. It requires treatment ' +
                                'information in prediction file, WHICH IS ' +
                                'MISSING!')
    else:
        text_to_print = None

    if (name_unordered != []) or (len(vnew_dict['y_name']) > 1) or cnew_dict[
            'fs_yes']:
        cnew_dict['l_centering_uncenter'] = False

    vn_add = {
        'x_name_remain':        x_name_remain,
        'x_name':               x_name,
        'z_name':               z_name,
        'x_name_in_tree':       x_name_in_tree,
        'x_balance_name':       x_balance_name,
        'x_name_always_in':     x_name_always_in,
        'name_ordered':         name_ordered,
        'name_unordered':       name_unordered,
        'names_to_check_train': names_to_check_train,
        'names_to_check_pred':  names_to_check_pred}
    vnew_dict.update(vn_add)

    return cnew_dict, vnew_dict, text_to_print


def controls_into_dic(
        mp_parallel, mp_type_vim, output_type, outpfad, datpfad, indata,
        preddata, outfiletext, screen_covariates, n_min_grid,
        check_perfectcorr, n_min_min, clean_data_flag,
        min_dummy_obs, mce_vart, p_diff_penalty, boot, n_min_max,
        support_min_p, weighted, support_check, support_quantil,
        subsample_factor_forest, subsample_factor_eval,
        m_min_share, m_grid, stop_empty, m_random_poisson, alpha_reg_min,
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
        boot_by_boot, obs_by_obs, max_elements_per_split, mp_with_ray,
        mp_ray_objstore_multiplier, verbose, no_ray_in_forest_building,
        predict_mcf, train_mcf, forest_files, match_nn_prog_score,
        se_boot_ate, se_boot_gate, se_boot_iate, support_max_del_train,
        _mp_ray_del, _mp_ray_shutdown,
        reduce_split_sample, reduce_split_sample_pred_share, reduce_training,
        reduce_training_share, reduce_prediction, reduce_prediction_share,
        reduce_largest_group_train, reduce_largest_group_train_share,
        iate_flag, iate_se_flag, l_centering_uncenter):
    """Build dictionary with parameters.

    Parameters
    ----------
    ... : All user defined control parameters.

    Returns
    -------
    controls_dict: Dictionary with a collection of these parameters.
    """
    controls_dict = {
        'output_type': output_type,
        'outpfad': outpfad, 'datpfad': datpfad,
        'indata': indata, 'preddata': preddata, 'outfiletext': outfiletext,
        'screen_covariates': screen_covariates, 'n_min_grid': n_min_grid,
        'check_perfectcorr': check_perfectcorr, 'n_min_min': n_min_min,
        'clean_data_flag': clean_data_flag, 'min_dummy_obs': min_dummy_obs,
        'mce_vart': mce_vart, 'mtot_p_diff_penalty': p_diff_penalty,
        'boot': boot, 'support_min_p': support_min_p, 'w_yes': weighted,
        'common_support': support_check, 'n_min_max': n_min_max,
        'support_quantil': support_quantil, 'knn_min_k': knn_min_k,
        'nw_bandw': nw_bandw, 'nw_kern': nw_kern_flag,
        'subsample_factor_forest': subsample_factor_forest,
        'subsample_factor_eval': subsample_factor_eval,
        'm_min_share': m_min_share,
        'm_grid': m_grid, 'stop_empty': stop_empty,
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
        'post_random_forest_vi': post_random_forest_vi,
        'gmate_no_evaluation_points': gmate_no_evaluation_points,
        'gmate_sample_share': gmate_sample_share, 'no_filled_plot':
        no_filled_plot, 'smooth_gates': smooth_gates,
        'sgates_bandwidth': smooth_gates_bandwidth,
        'sgates_no_evaluation_points': smooth_gates_no_evaluation_points,
        'show_plots': show_plots, 'weight_as_sparse': weight_as_sparse,
        'mp_type_weights': mp_type_weights, 'mp_weights_tree_batch':
        mp_weights_tree_batch, 'boot_by_boot': boot_by_boot, 'obs_by_obs':
        obs_by_obs, 'max_elements_per_split': max_elements_per_split,
        'mp_with_ray': mp_with_ray, 'mp_ray_objstore_multiplier':
        mp_ray_objstore_multiplier, 'verbose': verbose,
        'no_ray_in_forest_building': no_ray_in_forest_building,
        'pred_mcf': predict_mcf, 'train_mcf': train_mcf,
        'save_forest_files': forest_files,
        'se_boot_ate': se_boot_ate, 'se_boot_gate': se_boot_gate,
        'se_boot_iate': se_boot_iate,
        'support_max_del_train': support_max_del_train,
        '_mp_ray_del': _mp_ray_del, '_mp_ray_shutdown': _mp_ray_shutdown,
        'reduce_split_sample': reduce_split_sample,
        'reduce_split_sample_pred_share': reduce_split_sample_pred_share,
        'reduce_training': reduce_training,
        'reduce_training_share': reduce_training_share,
        'reduce_prediction': reduce_prediction,
        'reduce_prediction_share': reduce_prediction_share,
        'reduce_largest_group_train': reduce_largest_group_train,
        'reduce_largest_group_train_share': reduce_largest_group_train_share,
        'iate_flag': iate_flag, 'iate_se_flag': iate_se_flag
            }
    return controls_dict
