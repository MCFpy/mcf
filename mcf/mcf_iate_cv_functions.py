"""Created on Thu Oct 13 15:46:02 2022.

Contains the functions needed for the running all parts of the programme
@author: MLechner
-*- coding: utf-8 -*-
"""
from pathlib import Path
import os
import time

import pandas as pd
import numpy as np

from mcf import general_purpose as gp
from mcf import mcf_functions as mcf

def check_if_iate_cv(iate_cv_flag, iate_cv_folds, preddata, indata, iate_flag,
                     d_type='discrete'):
    """Check if cross-validation makes sense."""
    if iate_flag is False or d_type != 'discrete':
        return False, None
    if not isinstance(preddata, str):
        return False, None
    if indata.upper() != preddata.upper() or (iate_cv_flag is not True):
        return False, None
    if isinstance(iate_cv_folds, (int, float)):
        iate_cv_folds = round(iate_cv_folds)
        if iate_cv_folds < 2:
            iate_cv_folds = 5
    else:
        iate_cv_folds = 5
    return iate_cv_flag, iate_cv_folds


def iate_cv_proc(controls_dict, variable_dict, seed_sample_split=1234354,
                 with_output=False, file_with_out_path=__file__):
    """Estimate out-of-sample IATEs in a cross-validated fashion."""
    time1 = time.time()
    outpath = str(Path(file_with_out_path).parent.absolute())
    iate_c_dict, iate_v_dict, cv_dict = get_iate_dict(
        controls_dict, variable_dict, outpath)
    if with_output:
        str_p = (f'Estimation of IATEs with {cv_dict["folds"]} fold' +
                 ' cross-fitting')
        print(str_p)
        gp.print_f(cv_dict['outtext_cv'], str_p)
    names_pot_iate = estim_iate_cv(cv_dict, iate_c_dict, iate_v_dict,
                                   seed_sample_split=seed_sample_split,
                                   with_output=controls_dict['with_output'])
    gp.print_descriptive_stats_file(cv_dict['iate_sample'], 'all',
                                    to_file=True, df_instead_of_file=False,
                                    file_to_print=cv_dict['outtext_cv'])
    remove_dir(cv_dict['temppath'], with_output=with_output)
    time_string = ['Additional time for X-fitted IATEs:    ']
    time_difference = [time.time() - time1]
    print_str = gp.print_timing(time_string, time_difference)
    gp.print_f(cv_dict['outtext_cv'], print_str)
    return cv_dict['iate_sample'], names_pot_iate


def estim_iate_cv(cv_dict, iate_c_dict, iate_v_dict,
                  seed_sample_split=122435467, with_output=True,
                  max_train=1e100, called_by_mcf=True):
    """Estimate iates via cross-validation"""
    df_list_cv = split_data_into_folds(cv_dict['indata'], cv_dict['folds'],
                                       seed_sample_split)
    pred_data = []
    obs_pred_all = 0
    for i in range(cv_dict['folds']):
        if with_output:
            print(f'Fold: {i}', end='')
        obs_est, obs_pred = make_estimation_file_folds(
            i, df_list_cv, cv_dict['pred_temp_file'], cv_dict['est_temp_file'])
        print(f'  Obs: Estimation folds: {obs_est},',
              f' prediction folds: {obs_pred}  ATEs: ', end='')
        iate_c_dict['datpfad'] = str(
            Path(cv_dict['est_temp_file']).parent.resolve())
        iate_c_dict['indata'] = str(Path(cv_dict['est_temp_file']).stem)
        iate_c_dict['preddata'] = str(Path(cv_dict['pred_temp_file']).stem)
        iate_c_dict['outpfad'] = cv_dict['temppath']
        iate_c_dict['train_mcf'] = iate_c_dict['predict_mcf'] = True
        if called_by_mcf:
            (ate, _, _, _, _, _, _, _, _, _, _, names_pot_iate, pred_cv
            ) = mcf.modified_causal_forest_master(
                iate_c_dict, iate_v_dict, False, seed_sample_split)
        else:
            iate_dict = {**iate_c_dict, **iate_v_dict}
            (ate, _, _, _, _, _, _, _, _, _, _, names_pot_iate, pred_cv, _, _
            ) = mcf.modified_causal_forest(**iate_dict)
        pred_data.append(pred_cv)
        if with_output:
            print(*ate)
        obs_pred_all += len(pred_cv)
        if obs_pred_all > max_train:
            break
    pred_data_all = pd.concat(pred_data, axis=0)
    if with_output:
        str_p = f'Number of observations on common support: {obs_pred_all}'
        print(str_p)
        if called_by_mcf:
            gp.print_f(cv_dict['outtext_cv'], str_p)
    if obs_pred_all > max_train:
        pred_data_all = pred_data_all.sample(n=int(max_train), replace=False,
                                             axis=0,
                                             random_state=seed_sample_split)
    gp.delete_file_if_exists(cv_dict['iate_sample'])
    pred_data_all.to_csv(cv_dict['iate_sample'], index=False)
    return names_pot_iate


def make_estimation_file_folds(fold_nr, df_list_cv, pred_temp_file,
                               est_temp_file):
    """Create file with estimation and prediction data."""
    df_list_local = df_list_cv.copy()
    pred_data = df_list_local[fold_nr]
    df_list_local.pop(fold_nr)
    est_data = pd.concat(df_list_local, axis=0, copy=False)
    gp.delete_file_if_exists(pred_temp_file)
    gp.delete_file_if_exists(est_temp_file)
    pred_data.to_csv(pred_temp_file, index=False)
    est_data.to_csv(est_temp_file, index=False)
    return len(est_data), len(pred_data)


def split_data_into_folds(csv_file, folds, seed_sample_split):
    """Split data in folds for cross-validation."""
    data = pd.read_csv(csv_file)
    # random shufflinge the rows of the dataframe
    data = data.sample(frac=1, replace=False, random_state=seed_sample_split,
                       axis=0, ignore_index=True)
    df_list_cv = np.array_split(data, folds, axis=0)
    return df_list_cv


def get_iate_dict(controls_dict, variable_dict, outpath):
    """Adjust controls for mcf iate cv estimation and get cv controls."""
    iate_c_dict = controls_dict.copy()
    iate_v_dict = variable_dict.copy()
    iate_c_dict['d_type'] = 'discrete'
    iate_c_dict['atet_flag'] = iate_c_dict['gatet_flag'] = False
    iate_c_dict['balancing_test'] = False
    iate_c_dict['_descriptive_stats'] = False
    iate_c_dict['iate_flag'] = True
    iate_c_dict['post_est_stats'] = False
    iate_c_dict['variable_importance_oob'] = False
    iate_c_dict['_verbose'] = iate_c_dict['_with_output'] = False
    iate_c_dict['_return_iate_sp'] = True
    iate_v_dict['z_name_list'] = []
    iate_v_dict['z_name_ord'] = iate_v_dict['z_name_unord'] = []
    iate_v_dict['z_name_mgate'] = iate_v_dict['z_name_amgate'] = []
    iate_v_dict['x_balance_name_ord'] = []
    iate_v_dict['x_balance_name_unord'] = []
    iate_c_dict['outpath'] = outpath
    iate_c_dict['with_output'] = False
    cv_dict = {}
    cv_dict['indata'] = (iate_c_dict['datpfad'] + '/' + iate_c_dict['indata']
                         + '.csv')
    cv_dict['folds'] = iate_c_dict['iate_cv_folds']
    cv_dict['with_output'] = iate_c_dict['with_output']
    cv_dict['outpath'] = iate_c_dict['outpath']

    cv_dict['temppath'] = outpath + '/_tempcvmcf_'
    cv_dict['pred_temp_file'] = (cv_dict['temppath'] + '/'
                                 + iate_c_dict['indata'] + 'PRED.csv')
    cv_dict['est_temp_file'] = (cv_dict['temppath'] + '/'
                                + iate_c_dict['indata'] + 'EST.csv')
    create_dir(cv_dict['temppath'], cv_dict['with_output'],
               create_new_if_exists=False)
    cv_dict['iate_sample'] = (outpath + '/' + iate_c_dict['indata']
                              + '_IATE_CV.csv')
    outpath_red = str(Path(outpath).parents[1])
    cv_dict['outtext_cv'] = (outpath_red + '/' + iate_c_dict['outfiletext'] +
                             '_IATE_CV.txt')
    return iate_c_dict, iate_v_dict, cv_dict


def create_dir(path, with_output=True, create_new_if_exists=True):
    """Create directory and all its files."""
    if os.path.isdir(path):
        if create_new_if_exists:
            temppath = path
            for i in range(1000):
                if os.path.isdir(temppath):
                    temppath = path + str(i)
                else:
                    break
            if path != temppath:
                if with_output:
                    print(f'Directory for output {path} already ' +
                          f'exists. {temppath} is created for the output.')
                path = temppath
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError as oserr:
            raise Exception(
                f'Creation of the directory {path} failed') from oserr
        else:
            if with_output:
                print(f'Successfully created the directory {path}')
    return path


def remove_dir(path, with_output=True):
    """Remove directory and all its files."""
    if os.path.isdir(path):
        for temp_file in os.listdir(path):
            os.remove(os.path.join(path, temp_file))
        try:
            os.rmdir(path)
        except OSError:
            if with_output:
                print(f'Removal of the temorary directory {path:s} failed')
        else:
            if with_output:
                print(f'Successfully removed the directory {path:s}')
    else:
        if with_output:
            print('Temporary directory does not exist.')
