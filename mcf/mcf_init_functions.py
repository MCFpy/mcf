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
from mcf import mcf_init_add_functions as mcf_ia


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
    path_programme_run = str(Path(__file__).parent.absolute())
    if c_dict['datpfad'] is None:
        c_dict['datpfad'] = path_programme_run
    if c_dict['outpfad'] is None:
        c_dict['outpfad'] = path_programme_run + '/out'
    out_temp = c_dict['outpfad']
    for i in range(1000):
        if not (c_dict['with_output'] and c_dict['train_mcf']):
            break
        if os.path.isdir(out_temp):
            if not ((c_dict['with_output'] is False)
                    or (c_dict['verbose'] is False)):
                print(f'Directory for output {out_temp} already exists',
                      'A new directory is created for the output.')
            out_temp = c_dict['outpfad'] + str(i)
        else:
            try:
                os.mkdir(out_temp)
            except OSError as oserr:
                raise Exception(
                    f'Creation of the directory {out_temp} failed') from oserr
            else:
                if not ((c_dict['with_output'] is False)
                        or (c_dict['verbose'] is False)):
                    print(f'Successfully created the directory {out_temp}')
                if out_temp != c_dict['outpfad']:
                    c_dict['outpfad'] = out_temp
                break
    if (c_dict['indata'] is None) and c_dict['train_mcf']:
        raise Exception('Filename of indata must be specified')
    if (c_dict['preddata'] is None) and c_dict['train_mcf']:
        c_dict['preddata'] = c_dict['indata']
    if c_dict['outfiletext'] is None:
        c_dict['outfiletext'] = (c_dict['indata'] if c_dict['train_mcf'] else
                                 c_dict['preddata'])
    cnew_dict = copy.deepcopy(c_dict)

    temppfad = c_dict['outpfad'] + '/_tempmcf_'
    if os.path.isdir(temppfad):
        file_list = os.listdir(temppfad)
        if file_list:
            for temp_file in file_list:
                os.remove(os.path.join(temppfad, temp_file))
        if not ((c_dict['with_output'] is False)
                or (c_dict['verbose'] is False)):
            print('Temporary directory {temppfad} already exists')
            if file_list:
                print('All files deleted.')
    else:
        try:
            os.mkdir(temppfad)
        except OSError as oserr:
            raise Exception(f'Creation of the directory {temppfad} failed'
                            ) from oserr
        else:
            if not ((c_dict['with_output'] is False)
                    or (c_dict['verbose'] is False)):
                print(f'Successfully created the directory {temppfad}')
    make_dir = c_dict['pred_mcf'] and c_dict['with_output']
    cnew_dict = mcf_ia.get_fig_path(cnew_dict, 'cs_ate_iate', make_dir)
    cnew_dict = mcf_ia.get_fig_path(cnew_dict, 'gate', make_dir)
    cnew_dict = mcf_ia.get_fig_path(cnew_dict, 'mgate', make_dir)
    cnew_dict = mcf_ia.get_fig_path(cnew_dict, 'amgate', make_dir)
    cnew_dict = mcf_ia.get_fig_path(cnew_dict, 'common_support',
                                    c_dict['with_output'], no_csv=True)
    if c_dict['with_output']:
        dir_for_pred = cnew_dict['cs_ate_iate_fig_pfad_csv'] + '/'
    else:
        dir_for_pred = c_dict['outpfad'] + '/'
    if c_dict['train_mcf']:
        pred_sample_with_pred = dir_for_pred + c_dict['indata'] + 'X_IATE.csv'
    else:
        pred_sample_with_pred = (dir_for_pred + c_dict['preddata']
                                 + 'X_IATE.csv')
    cnew_dict['outfiletext'] = (c_dict['outpfad'] + '/' + c_dict['outfiletext']
                                + '.txt')
    cnew_dict['outfilesummary'] = (c_dict['outpfad'] + '/'
                                   + c_dict['outfiletext'] + '_Summary.txt')
    if c_dict['with_output']:
        gp.delete_file_if_exists(c_dict['outfiletext'])
        gp.delete_file_if_exists(cnew_dict['outfilesummary'])
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
    indat_temp = temppfad + '/' + 'indat_temp.csv'
    indata2_temp = temppfad + '/' + 'indata2_temp.csv'
    preddata2_temp = temppfad + '/' + 'preddata2_temp.csv'
    preddata3_temp = temppfad + '/' + 'preddata3_temp.csv'
    off_support_temp = temppfad + '/' + 'off_support_temp.csv'
    pred_eff_temp = temppfad + '/' + 'pred_eff_temp.csv'
    fs_sample_tmp = temppfad + '/' + 'fs_sample_tmp.csv'
    tree_sample_tmp = temppfad + '/' + 'tree_sample_tmp.csv'
    fill_y_sample_tmp = temppfad + '/' + 'fill_y_sample_tmp.csv'
    tree_sample_nn = temppfad + '/' + 'tree_sample_NN.csv'
    temporary_file = temppfad + '/' + 'temporaryfile.csv'
    nonlc_sample = temppfad + '/' + 'nonlc_sample.csv'
    lc_sample = temppfad + '/' + 'lc_sample.csv'
    cnew_dict['desc_stat'] = not c_dict['desc_stat'] is False

    if c_dict['output_type'] is None:
        c_dict['output_type'] = 2
    if c_dict['output_type'] == 0:
        print_to_file = False
        print_to_terminal = True
    elif c_dict['output_type'] == 1:
        print_to_file = True
        print_to_terminal = False
    else:
        print_to_file = print_to_terminal = True
    cnew_dict['verbose'] = not c_dict['verbose'] is False
    if c_dict['w_yes'] is True:  # Weighting
        cnew_dict['w_yes'] = True
    else:
        c_dict['w_yes'] = cnew_dict['w_yes'] = False

    if (c_dict['d_type'] is None) or (c_dict['d_type'] == 'discrete'):
        cnew_dict['d_type'] = 'discrete'
    elif c_dict['d_type'] == 'continuous':
        cnew_dict['d_type'] = 'continuous'
    else:
        raise Exception(f'{c_dict["d_type"]} is wrong treatment type.')

    cnew_dict['screen_covariates'] = not c_dict['screen_covariates'] is False
    cnew_dict['check_perfectcorr'] = not c_dict['check_perfectcorr'] is False
    if c_dict['min_dummy_obs'] is None:
        c_dict['min_dummy_obs'] = -1
    cnew_dict['min_dummy_obs'] = (10 if c_dict['min_dummy_obs'] < 1
                                  else round(c_dict['min_dummy_obs']))
    cnew_dict['clean_data_flag'] = not c_dict['clean_data_flag'] is False
    if c_dict['alpha_reg_min'] is None:
        c_dict['alpha_reg_min'] = -1
    if (c_dict['alpha_reg_min'] < 0) or (c_dict['alpha_reg_min'] >= 0.4):
        cnew_dict['alpha_reg_min'] = 0.1
        if c_dict['alpha_reg_min'] >= 0.4:
            print('Values of 0.4 and larger do not make sense for alpha',
                  'regularisation. alpha regularisation parameter set to ',
                  f'default value. {cnew_dict["alpha_reg_min"]:8.3f}')
    else:
        cnew_dict['alpha_reg_min'] = c_dict['alpha_reg_min']
    if c_dict['alpha_reg_max'] is None:
        c_dict['alpha_reg_max'] = -1
    if (c_dict['alpha_reg_max'] < 0) or (c_dict['alpha_reg_max'] >= 0.4):
        cnew_dict['alpha_reg_max'] = 0.1
        if c_dict['alpha_reg_max'] >= 0.4:
            print('Values of 0.4 and larger do not make sense for alpha',
                  'regularisation. alpha regularisation parameter set to ',
                  f'default value. {cnew_dict["alpha_reg_max"]:8.3f}')
    else:
        cnew_dict['alpha_reg_max'] = c_dict['alpha_reg_max']
    cnew_dict['alpha_reg_grid'] = (1 if c_dict['alpha_reg_grid'] is None
                                   else round(c_dict['alpha_reg_grid']))
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
        cnew_dict['panel_in_rf'] = not c_dict['panel_in_rf'] is False
        panel = True
    else:
        cnew_dict['panel_data'] = panel = cnew_dict['panel_in_rf'] = False
    data_train = (pd.read_csv(cnew_dict['indata'], header=0)
                  if c_dict['train_mcf']
                  else pd.read_csv(save_forest_file_csv, header=0))
    v_dict['d_name'] = mcf_ia.to_list_if_needed(v_dict['d_name'])
    v_dict['d_name'][0] = gp.adjust_var_name(v_dict['d_name'][0],
                                             data_train.columns.tolist())
    d_dat_pd = data_train[v_dict['d_name']].squeeze()  # pylint: disable=E1136
    d_dat_pd.dropna(inplace=True)
    if d_dat_pd.dtype == 'object':
        d_dat_pd = d_dat_pd.astype('category')
        print(d_dat_pd.cat.categories)
        if cnew_dict['with_output']:
            print('Automatic recoding of treatment variable')
            numerical_codes = pd.unique(d_dat_pd.cat.codes)
            print(numerical_codes)
        d_dat_pd = d_dat_pd.cat.codes
    d_dat = d_dat_pd.to_numpy()
    if cnew_dict['d_type'] == 'continuous':
        d_values = no_of_treatments = None
    else:
        d_values = np.int16(np.round(np.unique(d_dat)))
        d_values = d_values.tolist()
        no_of_treatments = len(d_values)     # Number of treatments
    if c_dict['boot'] is None:
        c_dict['boot'] = -1
    cnew_dict['boot'] = round(c_dict['boot']) if c_dict['boot'] > 0 else 1000
    if c_dict['train_mcf']:
        cnew_dict['save_forest'] = c_dict['save_forest'] is True
        if not c_dict['pred_mcf']:
            cnew_dict['save_forest'] = True
        n_train = len(data_train.index)
        if c_dict['mp_type_vim'] != 1 and c_dict['mp_type_vim'] != 2:
            if n_train < 20000:
                cnew_dict['mp_type_vim'] = 1  # MP over var's, fast, lots RAM
            else:
                cnew_dict['mp_type_vim'] = 2  # MP over bootstraps.
        temp = data_train.groupby(v_dict['d_name'])  # pylint: disable=E1101
        vcount = temp.size()
        n_d = vcount.to_numpy()         # Number of obs in treatments
        if n_train != n_d.sum():
            print('-' * 80)
            print('Counting treatments does not work. ' +
                  f'n_d_sum: {n_d.sum()}, n_train: {n_train}.')
            print('Difference could be due to missing values.')
        n_d_subsam = n_d.min() * cnew_dict['share_forest_sample']
        if c_dict['n_min_min'] is None:
            c_dict['n_min_min'] = cnew_dict['n_min_min'] = -1
        else:
            cnew_dict['n_min_min'] = round(c_dict['n_min_min'])   # grid n_min
        if cnew_dict['n_min_min'] < 1:
            if cnew_dict['n_min_min'] == -1:
                cnew_dict['n_min_min'] = round((n_d_subsam**0.4)/5)
                if cnew_dict['n_min_min'] < 5:
                    cnew_dict['n_min_min'] = 5
            else:
                cnew_dict['n_min_min'] = round((n_d_subsam**0.4)/10)
                if cnew_dict['n_min_min'] < 3:
                    cnew_dict['n_min_min'] = 3
            if cnew_dict['d_type'] == 'discrete':
                cnew_dict['n_min_min'] *= len(d_values)
        if cnew_dict['d_type'] == 'discrete':
            if c_dict['n_min_treat'] is None or c_dict['n_min_treat'] < 1:
                cnew_dict['n_min_treat'] = 3
            else:
                cnew_dict['n_min_treat'] = int(c_dict['n_min_treat'])
            if cnew_dict['n_min_min'] < cnew_dict['n_min_treat'
                                                  ] * len(d_values):
                cnew_dict['n_min_min'] = cnew_dict['n_min_treat'
                                                   ] * len(d_values)
                text_to_print = ('Minimum leaf size adjusted. Smallest ',
                                 ' leafsize set to: ', cnew_dict['n_min_min'])
            else:
                text_to_print = None
        else:
            cnew_dict['n_min_treat'] = 0
        if c_dict['n_min_max'] is None:
            c_dict['n_min_max'] = cnew_dict['n_min_max'] = -1
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
            if cnew_dict['d_type'] == 'discrete':
                cnew_dict['n_min_max'] *= len(d_values)
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
        cnew_dict['max_cats_cont_vars'] = (
            n_train + 1 if c_dict['max_cats_cont_vars'] < 1
            else round(c_dict['max_cats_cont_vars']))
        if c_dict['random_thresholds'] is None:
            c_dict['random_thresholds'] = -1
        if c_dict['random_thresholds'] < 0:     # Saves computation time
            cnew_dict['random_thresholds'] = round(4 + n_train**0.2)
        # Feature preselection
        cnew_dict['fs_yes'] = bool(c_dict['fs_yes'])
        if c_dict['fs_rf_threshold'] is None:
            c_dict['fs_rf_threshold'] = -1
        if c_dict['fs_rf_threshold'] <= 0:
            cnew_dict['fs_rf_threshold'] = 0.0001
        cnew_dict['fs_other_sample'] = not c_dict['fs_other_sample'] is False
        if c_dict['fs_other_sample_share'] is None:
            c_dict['fs_other_sample_share'] = -1
        if (c_dict['fs_other_sample_share'] < 0) or (
                c_dict['fs_other_sample_share'] > 0.5):
            cnew_dict['fs_other_sample_share'] = 0.33
        if cnew_dict['fs_other_sample'] is False or (
                cnew_dict['fs_yes'] is False):
            cnew_dict['fs_other_sample_share'] = 0

        # size of subsampling samples         n/2: size of forest sample
        subsam_share_forest = mcf_ia.sub_size(
            n_train, c_dict['subsample_factor_forest'], 0.67)
        if c_dict['subsample_factor_eval'] is None:
            c_dict['subsample_factor_eval'] = True  # Default
        if c_dict['subsample_factor_eval'] is False:
            c_dict['subsample_factor_eval'] = 1000000000
        if c_dict['subsample_factor_eval'] is True:
            c_dict['subsample_factor_eval'] = 2
        if c_dict['subsample_factor_eval'] < 0.01:
            c_dict['subsample_factor_eval'] = 1000000000
        subsam_share_eval = min(
            subsam_share_forest * c_dict['subsample_factor_eval'], 1)
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
                if cnew_dict['d_type'] == 'discrete':
                    cnew_dict['mtot_p_diff_penalty'] *= np.sqrt(
                        no_of_treatments * (no_of_treatments-1)/2)
        else:
            if mtot == 4:
                if c_dict['mtot_p_diff_penalty'] > 1:  # if accidently scaled %
                    cnew_dict['mtot_p_diff_penalty'] = c_dict[
                        'mtot_p_diff_penalty'] / 100
                    print('Probability of using p-score larger than 1.',
                          'Devided by 100.')
                ass_err = 'Probability of using p-score > 1. Programm stopped.'
                assert 0 <= cnew_dict['mtot_p_diff_penalty'] <= 1, ass_err
        if cnew_dict['mtot_p_diff_penalty']:
            estimator_str += ' penalty fct'
        if c_dict['l_centering'] is not False:
            cnew_dict['l_centering'] = True
            if c_dict['l_centering_share'] is None:
                c_dict['l_centering_share'] = -1
            if c_dict['l_centering_cv_k'] is None:
                c_dict['l_centering_cv_k'] = -1
            cnew_dict['l_centering_new_sample'] = (
                c_dict['l_centering_new_sample'] is True)
            if cnew_dict['l_centering_new_sample']:
                if not 0.0999 < c_dict['l_centering_share'] < 0.9001:
                    cnew_dict['l_centering_share'] = 0.25
            else:
                if c_dict['l_centering_cv_k'] < 1:
                    cnew_dict['l_centering_cv_k'] = 5
                else:
                    cnew_dict['l_centering_cv_k'] = int(
                        round(c_dict['l_centering_cv_k']))
            if c_dict['l_centering_uncenter'] is not False:
                cnew_dict['l_centering_uncenter'] = True
            if c_dict['l_centering_replication'] is not False:
                cnew_dict['l_centering_replication'] = True
        else:
            cnew_dict['l_centering'] = False
            cnew_dict['l_centering_new_sample'] = None
            cnew_dict['l_centering_cv_k'] = None
            cnew_dict['l_centering_uncenter'] = None
        cnew_dict['match_nn_prog_score'] = (
            not c_dict['match_nn_prog_score'] is False)
        cnew_dict['nn_main_diag_only'] = c_dict['nn_main_diag_only'] is True
        cnew_dict['no_ray_in_forest_building'] = (
            c_dict['no_ray_in_forest_building'] is True)
        if c_dict['max_cats_z_vars'] is None:
            c_dict['max_cats_z_vars'] = -1
        cnew_dict['max_cats_z_vars'] = (
            round(n_train ** 0.3) if c_dict['max_cats_z_vars'] < 1 else
            round(c_dict['max_cats_z_vars']))
        cnew_dict['var_import_oob'] = c_dict['var_import_oob'] is True
    else:
        n_min = subsam_share_eval = subsam_share_forest = mtot = None
        mtot_no_mce = None
        estimator_str = 'Estimates loaded.'
    if c_dict['pred_mcf']:
        cnew_dict['cluster_std'] = c_dict['cluster_std'] is True
        if not (cnew_dict['cluster_std'] or panel):
            cnew_dict['cluster_std'] = False
        data_pred = pd.read_csv(cnew_dict['preddata'], header=0)
        n_pred = len(data_pred.index)
        # Choice based sampling (oversampling of treatments)
        if c_dict['choice_based_yes'] is True:
            err_txt = 'No choice based sample with continuous treatments.'
            assert cnew_dict['d_type'] == 'discrete', err_txt
            cnew_dict['choice_based_yes'] = True
            err_t = ('Choice based sampling. Rows in choice probabilities do'
                     + ' not correspond to number of treatments.')
            assert len(c_dict['choice_based_probs']) == no_of_treatments, err_t
            err_t = ('Choice based sampling active. Not possible to have zero'
                     + ' or negative choice probability.')
            assert all(v > 0 for v in c_dict['choice_based_probs']), err_t
            pcb = np.array(c_dict['choice_based_probs'])
            pcb = pcb / np.sum(pcb) * no_of_treatments
            cnew_dict['choice_based_probs'] = pcb.tolist()
            if cnew_dict['with_output'] and cnew_dict['verbose']:
                print('Choice based sampling active. Normalized',
                      'choice probabilites for', 'treatments:',
                      cnew_dict['choice_based_probs'])
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
        cnew_dict['balancing_test_w'] = (
            not c_dict['balancing_test_w'] is False)
        if (v_dict['x_balance_name_ord'] == []) and (
                v_dict['x_balance_name_unord'] == []):
            cnew_dict['balancing_test_w'] = False
        cnew_dict['atet_flag'] = bool(c_dict['atet_flag'])
        if c_dict['gatet_flag'] is True:
            cnew_dict['gatet_flag'] = cnew_dict['atet_flag'] = True
        else:
            cnew_dict['gatet_flag'] = False
        if c_dict['gmate_sample_share'] is None:
            c_dict['gmate_sample_share'] = -1
        if c_dict['gmate_sample_share'] <= 0:
            cnew_dict['gmate_sample_share'] = (
                1 if n_pred < 1000
                else (1000 + ((n_pred-1000) ** 0.75)) / n_pred)
        if c_dict['gmate_no_evaluation_points'] is None:
            c_dict['gmate_no_evaluation_points'] = -1
        cnew_dict['gmate_no_evaluation_points'] = (
            50 if c_dict['gmate_no_evaluation_points'] < 2
            else round(c_dict['gmate_no_evaluation_points']))
        cnew_dict['smooth_gates'] = not c_dict['smooth_gates'] is False
        if c_dict['sgates_bandwidth'] is None:
            c_dict['sgates_bandwidth'] = -1
        cnew_dict['sgates_bandwidth'] = (
            1 if c_dict['sgates_bandwidth'] <= 0
            else c_dict['sgates_bandwidth'])
        if c_dict['sgates_no_evaluation_points'] is None:
            c_dict['sgates_no_evaluation_points'] = -1
        cnew_dict['sgates_no_evaluation_points'] = (
            50 if c_dict['sgates_no_evaluation_points'] < 2
            else round(c_dict['sgates_no_evaluation_points']))
        if c_dict['gates_minus_previous']:
            cnew_dict['gates_minus_previous'] = True
        else:
            cnew_dict['gates_minus_previous'] = False
        if c_dict['iate_flag'] is False:
            cnew_dict['iate_flag'] = cnew_dict['iate_se_flag'] = False
            c_dict['post_est_stats'] = False
        else:
            cnew_dict['iate_flag'] = True
            cnew_dict['iate_se_flag'] = not c_dict['iate_se_flag'] is False
        cnew_dict['iate_eff_flag'] = not c_dict['iate_eff_flag'] is False
        # Post estimation parameters
        cnew_dict['post_est_stats'] = not c_dict['post_est_stats'] is False
        cnew_dict['bin_corr_yes'] = not c_dict['bin_corr_yes'] is False
        if c_dict['bin_corr_thresh'] is None:
            c_dict['bin_corr_thresh'] = -1
        if c_dict['bin_corr_thresh'] < 0 or c_dict['bin_corr_thresh'] > 1:
            cnew_dict['bin_corr_thresh'] = 0.1  # Minimum threshhold of abs.
        else:                                   # correlation to be displayed
            cnew_dict['bin_corr_thresh'] = c_dict['bin_corr_thresh']
        cnew_dict['post_plots'] = not c_dict['post_plots'] is False
        cnew_dict['relative_to_first_group_only'] = (
            not c_dict['relative_to_first_group_only'] is False)
        cnew_dict['post_km'] = not c_dict['post_km'] is False
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
                cnew_dict['post_km_no_of_groups'] = [int(
                    round(c_dict['post_km_no_of_groups']))]
        else:
            if not isinstance(c_dict['post_km_no_of_groups'], list):
                c_dict['post_km_no_of_groups'] = list(
                    cnew_dict['post_km_no_of_groups'])
                cnew_dict['post_km_no_of_groups'] = [
                    round(int(a)) for a in c_dict['post_km_no_of_groups']]
        if c_dict['post_km_replications'] is None:
            c_dict['post_km_replications'] = -1
        cnew_dict['post_km_replications'] = (
            10 if c_dict['post_km_replications'] < 0
            else round(c_dict['post_km_replications']))
        if c_dict['post_kmeans_max_tries'] is None:
            cnew_dict['post_kmeans_max_tries'] = 1000
        if cnew_dict['post_kmeans_max_tries'] < 10:
            cnew_dict['post_kmeans_max_tries'] = 10
        add_pred_to_data_file = bool(cnew_dict['post_est_stats'])
        cnew_dict['post_random_forest_vi'] = (
            not c_dict['post_random_forest_vi'] is False)
        q_w = [0.5, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]  # Weight analysis
        if c_dict['mp_type_weights'] != 2:
            cnew_dict['mp_type_weights'] = 1  # MP over obs, fast, lots of mem
        else:
            cnew_dict['mp_type_weights'] = 2
        cnew_dict['weight_as_sparse'] = not (c_dict['weight_as_sparse']
                                             is False)
        if c_dict['mp_weights_tree_batch'] is None:
            c_dict['mp_weights_tree_batch'] = -1
        cnew_dict['mp_weights_tree_batch'] = (
            0 if c_dict['mp_weights_tree_batch'] < 1
            else round(c_dict['mp_weights_tree_batch']))
    else:
        add_pred_to_data_file = q_w = None
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
    cnew_dict['fig_dpi'] = (
        500 if c_dict['fig_dpi'] < 10 else round(cnew_dict['fig_dpi']))
    cnew_dict['fig_legend_loc'] = 'best'
    if c_dict['fig_ci_level'] is None:
        c_dict['fig_ci_level'] = -1
    if c_dict['fig_ci_level'] < 0.5 or c_dict['fig_ci_level'] > 0.999999:
        cnew_dict['fig_ci_level'] = 0.90
    if c_dict['no_filled_plot'] is None:
        c_dict['no_filled_plot'] = -1
    cnew_dict['no_filled_plot'] = (20 if c_dict['no_filled_plot'] < 5
                                   else round(c_dict['no_filled_plot']))
    cnew_dict['show_plots'] = not c_dict['show_plots'] is False

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
            mem_object_store_1 = mem_object_store_2 = None
    else:
        mem_object_store_1 = mem_object_store_2 = None
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
        cnew_dict['no_parallel'] = round(psutil.cpu_count(logical=True)*0.8)
        mp_automatic = True
    elif -0.5 <= c_dict['no_parallel'] <= 1.5:
        cnew_dict['no_parallel'] = 1
    else:
        cnew_dict['no_parallel'] = round(c_dict['no_parallel'])
    sys_share = 0.7 * getattr(psutil.virtual_memory(), 'percent') / 100

    if c_dict['_ray_or_dask'] == 'dask':
        cnew_dict['_ray_or_dask'] == 'dask'
    else:
        cnew_dict['_ray_or_dask'] == 'ray'

    if c_dict['_mp_ray_del'] is None:
        cnew_dict['_mp_ray_del'] = ('refs',)
    else:
        possible_vals = ('refs', 'rest', 'none')
        if isinstance(c_dict['_mp_ray_del'], (str, list)):
            cnew_dict['_mp_ray_del'] = tuple(c_dict['_mp_ray_del'])
        if len(cnew_dict['_mp_ray_del']) > 2:
            print(cnew_dict['_mp_ray_del'])
            raise Exception('Too many parameters for _mp_ray_del')
        ass_err = '_mp_ray_del is no Tuple'
        assert isinstance(cnew_dict['_mp_ray_del'], tuple), ass_err
        ass_e = 'Wrong parameters for _mp_ray_del'
        assert all(i in possible_vals for i in cnew_dict['_mp_ray_del']), ass_e
    if c_dict['_mp_ray_shutdown'] is None:
        cnew_dict['_mp_ray_shutdown'] = not ((n_train < 100000) and
                                             (n_pred < 100000))
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
        err_txt = ('No sample split possible. Training file differs from'
                   + ' prediction file (or no training and prediction).')
        assert c_dict['reduce_split_sample'] is not True, err_txt
        cnew_dict['reduce_split_sample'] = False
        cnew_dict['reduce_split_sample_pred_share'] = 0
    ass_err = 'Reduce_split_sample_pred_share outside [0,1]'
    assert 0 <= cnew_dict['reduce_split_sample_pred_share'] <= 1, ass_err
    if c_dict['reduce_training'] is not True:
        cnew_dict['reduce_training'] = False
        cnew_dict['reduce_training_share'] = 0
    else:
        if c_dict['reduce_training_share'] is None:
            cnew_dict['reduce_training_share'] = 0.5
    ass_err = 'Reduce_training_share outside [0,1]'
    assert 0 <= cnew_dict['reduce_training_share'] <= 1, ass_err
    if c_dict['reduce_prediction'] is not True:
        cnew_dict['reduce_prediction'] = False
        cnew_dict['reduce_prediction_share'] = 0
    else:
        if c_dict['reduce_prediction_share'] is None:
            cnew_dict['reduce_prediction_share'] = 0.5
    err_txt = 'Reduce_prediction_share outside [0,1]'
    assert 0 <= cnew_dict['reduce_prediction_share'] <= 1, err_txt
    if c_dict['reduce_largest_group_train'] is not True:
        cnew_dict['reduce_largest_group_train'] = False
        cnew_dict['reduce_largest_group_train_share'] = 0
    else:
        if c_dict['reduce_largest_group_train_share'] is None:
            cnew_dict['reduce_largest_group_train_share'] = 0.5
    err_txt = 'reduce_largest_group_train_share outside [0,1]'
    assert 0 <= cnew_dict['reduce_largest_group_train_share'] <= 1, err_txt
    if cnew_dict['d_type'] == 'continuous':
        cnew_dict['atet'] = cnew_dict['gatet'] = False
        grid_nn, grid_w, grid_dr = 10, 10, 100
        cnew_dict['ct_grid_nn'] = mcf_ia.ct_grid(c_dict['ct_grid_nn'], grid_nn)
        cnew_dict['ct_grid_w'] = mcf_ia.ct_grid(c_dict['ct_grid_w'], grid_w)
        cnew_dict['ct_grid_dr'] = mcf_ia.ct_grid(c_dict['ct_grid_dr'], grid_dr)
        if cnew_dict['ct_grid_dr'] < cnew_dict['ct_grid_w']:
            cnew_dict['ct_grid_dr'] = cnew_dict['ct_grid_w']
        cnew_dict['ct_grid_nn_val'] = mcf_ia.grid_val(cnew_dict['ct_grid_nn'],
                                                      d_dat)
        cnew_dict['ct_grid_w_val'] = mcf_ia.grid_val(cnew_dict['ct_grid_w'],
                                                     d_dat)
        cnew_dict['ct_grid_dr_val'] = mcf_ia.grid_val(cnew_dict['ct_grid_dr'],
                                                      d_dat)
        cnew_dict['ct_grid_nn'] = len(cnew_dict['ct_grid_nn_val'])
        cnew_dict['ct_grid_w'] = len(cnew_dict['ct_grid_w_val'])
        cnew_dict['ct_grid_dr'] = len(cnew_dict['ct_grid_dr_val'])
        v_dict['d_grid_nn_name'] = mcf_ia.grid_name(v_dict['d_name'],
                                                    'grid_nn')
        v_dict['d_grid_w_name'] = mcf_ia.grid_name(v_dict['d_name'], 'grid_w')
        v_dict['d_grid_dr_name'] = mcf_ia.grid_name(v_dict['d_name'],
                                                    'grid_dr')
        no_of_treat = len(cnew_dict['ct_grid_nn_val'])
        precision_of_cont_treat = 4
        (cnew_dict['ct_w_to_dr_int_w01'], cnew_dict['ct_w_to_dr_int_w10'],
         cnew_dict['ct_w_to_dr_index_full'], cnew_dict['ct_d_values_dr_list'],
         cnew_dict['ct_d_values_dr_np']) = mcf_ia.interpol_weights(
            cnew_dict['ct_grid_dr'], cnew_dict['ct_grid_w'],
            cnew_dict['ct_grid_w_val'], precision_of_cont_treat)
    else:
        no_of_treat = no_of_treatments
        cnew_dict['ct_grid_nn'] = cnew_dict['ct_grid_nn_val'] = None
        cnew_dict['ct_grid_dr'] = cnew_dict['ct_grid_dr_val'] = None
    if c_dict['support_adjust_limits'] is None:
        cnew_dict['support_adjust_limits'] = (no_of_treat - 2) * 0.05
        err_txt = 'Negative common support adjustment factor not possible.'
    assert cnew_dict['support_adjust_limits'] >= 0, err_txt
    cn_add = {
        'temppfad': temppfad, 'pred_sample_with_pred': pred_sample_with_pred,
        'indat_temp': indat_temp, 'indata2_temp': indata2_temp,
        'preddata2_temp': preddata2_temp, 'preddata3_temp': preddata3_temp,
        'off_support_temp': off_support_temp, 'pred_eff_temp': pred_eff_temp,
        'fs_sample_temp': fs_sample_tmp, 'tree_sample_temp': tree_sample_tmp,
        'fill_y_sample_temp': fill_y_sample_tmp,
        'tree_sample_nn': tree_sample_nn, 'temporary_file': temporary_file,
        'lc_sample': lc_sample, 'nonlc_sample': nonlc_sample, 'panel': panel,
        'grid_n_min': n_min, 'title_variance': 'Weight-based variance',
        'add_pred_to_data_file': add_pred_to_data_file,
        'subsam_share_forest': subsam_share_forest,
        'subsam_share_eval':  subsam_share_eval, 'mtot': mtot,
        'mtot_no_mce': mtot_no_mce, 'd_values': d_values,
        'no_of_treat': no_of_treatments, 'q_w': q_w, 'sys_share': sys_share,
        'mp_automatic': mp_automatic, 'grid_alpha_reg': alpha_reg,
        'print_to_terminal': print_to_terminal, 'print_to_file': print_to_file,
        'mem_object_store_1': mem_object_store_1,
        'mem_object_store_2': mem_object_store_2,
        'mem_object_store_3': mem_object_store_3,
        'save_forest_file_pickle': save_forest_file_pickle,
        'save_forest_file_csv': save_forest_file_csv,
        'save_forest_file_ps': save_forest_file_ps,
        'save_forest_file_d_train_tree': save_forest_file_d_train_tree,
        'estimator_str': estimator_str}
    cnew_dict.update(cn_add)
    # Check for inconsistencies
    if cnew_dict['iate_flag'] is False or cnew_dict['pred_mcf'] is False:
        cnew_dict['iate_eff_flag'] = False
    if cnew_dict['iate_eff_flag'] and not cnew_dict['train_mcf']:
        print('2nd round efficient IATEs are only computed if training is ',
              'activated. They will be disabled.')
        cnew_dict['iate_eff_flag'] = False
    if c_dict['train_mcf']:
        if v_dict['y_name'] is None or v_dict['y_name'] == []:
            raise Exception('y_name must be specified.')
        if v_dict['d_name'] is None or v_dict['d_name'] == []:
            raise Exception('d_name must be specified.')
        if v_dict['y_tree_name'] is None or v_dict['y_tree_name'] == []:
            if isinstance(v_dict['y_name'], (list, tuple)):
                v_dict['y_tree_name'] = [v_dict['y_name'][0]]
            else:
                v_dict['y_tree_name'] = [v_dict['y_name']]
    if (v_dict['x_name_ord'] is None or v_dict['x_name_ord'] == []) and (
            v_dict['x_name_unord'] is None or v_dict['x_name_unord'] == []):
        raise Exception('x_name_ord or x_name_unord must be specified.')
    vnew_dict = copy.deepcopy(v_dict)
    if cnew_dict['cluster_std'] or cnew_dict['panel']:
        ass_err = 'More than 1 name for cluster variable.'
        assert len(v_dict['cluster_name']) < 2, ass_err
        vnew_dict['cluster_name'] = gp.cleaned_var_names(
            vnew_dict['cluster_name'])
    else:
        vnew_dict['cluster_name'] = []
    vnew_dict['y_tree_name'] = mcf_ia.to_list_if_needed(
        vnew_dict['y_tree_name'])
    vnew_dict['y_tree_name'] = gp.cleaned_var_names(
        vnew_dict['y_tree_name'])
    vnew_dict['y_name'] = mcf_ia.to_list_if_needed(vnew_dict['y_name'])
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
        if vnew_dict['w_name'] is None or vnew_dict['w_name'] == []:
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
    if x_name_in_tree:
        x_name_remain = gp.cleaned_var_names(copy.deepcopy(
            x_name_in_tree + x_name_remain))
        x_name = gp.cleaned_var_names(copy.deepcopy(x_name_in_tree + x_name))
    if not ((name_ordered == []) or (name_unordered == [])):
        if any(value for value in name_ordered if value in name_unordered):
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
    text_to_print = None
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
            cnew_dict['atet_flag'] = cnew_dict['gatet_flag'] = False
            if text_to_print is not None:
                text_to_print = add_text + ' \n' + text_to_print
            else:
                text_to_print = add_text
            err_txt = ('Choice based sampling relates only to prediction'
                       + ' file. It requires treatment information in'
                       + ' prediction file, WHICH IS MISSING!')
            assert not cnew_dict['choice_based_yes'], err_txt

    if  cnew_dict['fs_yes']:
        cnew_dict['l_centering_uncenter'] = False
    vn_add = {
        'x_name_remain': x_name_remain, 'x_name': x_name, 'z_name': z_name,
        'x_name_in_tree': x_name_in_tree, 'x_balance_name': x_balance_name,
        'x_name_always_in': x_name_always_in, 'name_ordered': name_ordered,
        'name_unordered': name_unordered,
        'names_to_check_train': names_to_check_train,
        'names_to_check_pred':  names_to_check_pred}
    vnew_dict.update(vn_add)
    return cnew_dict, vnew_dict, text_to_print
