"""
Procedures needed for ATE estimation.

Created on Thu Dec  8 15:48:57 2020.

@author: MLechner

# -*- coding: utf-8 -*-
"""
import copy
import pandas as pd
import numpy as np
from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est


def local_centering_new_sample(lc_csvfile, nonlc_csvfile, v_dict,
                               var_x_type_dict, c_dict):
    """
    Generate centered variables and add to file.

    Parameters
    ----------
    lc_csvfile : String. csv-file to estimate RF.
    nonlc_csvfile : String. csv-file to be used for centering.
    v_dict : Dict. Variable names.
    var_x_type_dict : Dictionary with variables and type.
    c_dict : Dict. Controls.

    Returns
    -------
    new_csv_file : String. csv-file to which centered variables are added.
    old_y_name : List of strings. Names of variables to be centered.
    new_y_name : List of strings. Names of centered variables.

    """
    if c_dict['with_output'] and c_dict['verbose']:
        print('\nLocal centering with Random Forests estimated in',
              'independent sample')
    # 1) Create dummies of ordinal variables
    # for each element in y_name do
    #    2) estimate RF on l_cent_sample
    #    3) predict y for nonlc_sample
    #    4) subtract predicted from observed for nonlc_sample
    #    5) Create new variable, update centered_y_name
    # 6) add new data to nonlc_sample and write to file under same name
    max_workers = copy.copy(c_dict['no_parallel'])
    lc_data_df = pd.read_csv(lc_csvfile)
    nonlc_data_df = pd.read_csv(nonlc_csvfile)
    x_names = var_x_type_dict.keys()
    lc_x_df = lc_data_df[x_names]
    lc_y_df = lc_data_df[v_dict['y_name']]
    nonlc_x_df = nonlc_data_df[x_names]
    nonlc_y_df = nonlc_data_df[v_dict['y_name']]
    if c_dict['with_output']:
        print()
        print('Independent sample used for local centering.')
        print('Number of observations used only for computing E(y|x): ',
              len(lc_y_df.index))
    names_unordered = []
    for x_name in x_names:
        if var_x_type_dict[x_name] > 0:
            names_unordered.append(x_name)
    if names_unordered:  # List is not empty
        lc_x_dummies = pd.get_dummies(lc_x_df, columns=names_unordered)
        nonlc_x_dummies = pd.get_dummies(nonlc_x_df, columns=names_unordered)
        x_names_in_both = np.intersect1d(lc_x_dummies.columns,
                                         nonlc_x_dummies.columns)
        lc_x_dummies = lc_x_dummies[x_names_in_both]
        nonlc_x_dummies = nonlc_x_dummies[x_names_in_both]
        lc_x_df = pd.concat([lc_x_df, lc_x_dummies], axis=1)
        nonlc_x_df = pd.concat([nonlc_x_df, nonlc_x_dummies], axis=1)
    x_train = lc_x_df.to_numpy()
    x_pred = nonlc_x_df.to_numpy()
    y_m_yx = np.empty(np.shape(nonlc_y_df))
    centered_y_name = []
    for indx, y_name in enumerate(v_dict['y_name']):
        y_train = lc_y_df[y_name].to_numpy()
        y_nonlc = nonlc_y_df[y_name].to_numpy()
        y_pred, _, _, _, _, _, _ = gp_est.RandomForest_scikit(
            x_train, y_train, x_pred, y_name=y_name, boot=c_dict['boot'],
            n_min=c_dict['grid_n_min'], no_features=c_dict['m_grid'],
            workers=max_workers, pred_p_flag=True,
            pred_t_flag=False, pred_oob_flag=False, with_output=True)
        y_m_yx[:, indx] = y_nonlc - y_pred  # centered outcomes
        centered_y_name.append(y_name + 'LC')
    y_m_yx_df = pd.DataFrame(data=y_m_yx, columns=centered_y_name)
    nonlc_data_df = pd.concat([nonlc_data_df, y_m_yx_df], axis=1)
    gp.delete_file_if_exists(nonlc_csvfile)
    nonlc_data_df.to_csv(nonlc_csvfile, index=False)
    if c_dict['with_output']:
        all_y_name = v_dict['y_name'][:]
        for name in centered_y_name:
            all_y_name.append(name)
        gp.print_descriptive_stats_file(
            nonlc_csvfile, all_y_name, c_dict['print_to_file'])
    return nonlc_csvfile, v_dict['y_name'], centered_y_name


def local_centering_cv(datafiles, v_dict, var_x_type_dict, c_dict):
    """
    Compute local centering for cross-validation.

    Parameters
    ----------
    datafiles : Tuple of Strings. Names of datafiles.
    v_dict : Dict. Variable names.
    var_x_type_dict : Dictionary with variables and type.
    c_dict : Dict. Controls.

    Returns
    -------
    old_y_name : List of strings. Names of variables to be centered.
    new_y_name : List of strings. Names of centered variables.

    """
    max_workers = copy.copy(c_dict['no_parallel'])
    if c_dict['with_output']:
        print()
        print('Cross-validation used for local centering.',
              ' {:2} folds used.'. format(c_dict['l_centering_cv_k']))
    seed = 9324561
    rng = np.random.default_rng(seed)
    add_yx_names = True
    centered_y_name = []
    names_unordered = []
    for file_name in datafiles:
        data_df = pd.read_csv(file_name)
        x_names = var_x_type_dict.keys()
        x_df = data_df[x_names]
        y_df = data_df[v_dict['y_name']]
        obs = len(y_df.index)
        if add_yx_names:
            for x_name in x_names:
                if var_x_type_dict[x_name] > 0:
                    names_unordered.append(x_name)
        if names_unordered:  # List is not empty
            x_dummies = pd.get_dummies(x_df, columns=names_unordered)
            x_df = pd.concat([x_df, x_dummies], axis=1)
        index = np.arange(obs)       # indices
        rng.shuffle(index)
        index_folds = np.array_split(index, c_dict['l_centering_cv_k'])
        x_np = x_df.to_numpy()
        y_np = y_df.to_numpy()
        y_m_yx = np.empty(np.shape(y_df))
        for fold_pred in range(c_dict['l_centering_cv_k']):
            fold_train = [x for i,
                          x in enumerate(index_folds) if i != fold_pred]
            index_train = np.hstack(fold_train)
            index_pred = index_folds[fold_pred]
            x_pred = x_np[index_pred]
            x_train = x_np[index_train]
            for indx, y_name in enumerate(v_dict['y_name']):
                y_train = y_np[index_train, indx]
                y_pred_rf, _, _, _, _, _, _ = gp_est.RandomForest_scikit(
                    x_train, y_train, x_pred, y_name=y_name,
                    boot=c_dict['boot'], n_min=c_dict['grid_n_min'],
                    no_features=c_dict['m_grid'], workers=max_workers,
                    pred_p_flag=True, pred_t_flag=False, pred_oob_flag=False,
                    with_output=True)
                y_m_yx[index_pred, indx] = y_np[index_pred, indx] - y_pred_rf
                if add_yx_names:
                    centered_y_name.append(y_name + 'LC')
            add_yx_names = False
        y_m_yx_df = pd.DataFrame(data=y_m_yx, columns=centered_y_name)
        new_data_df = pd.concat([data_df, y_m_yx_df], axis=1)
        gp.delete_file_if_exists(file_name)
        new_data_df.to_csv(file_name, index=False)
        if c_dict['with_output']:
            all_y_name = v_dict['y_name'][:]
            for name in centered_y_name:
                all_y_name.append(name)
            gp.print_descriptive_stats_file(
                file_name, all_y_name, c_dict['print_to_file'])
    return v_dict['y_name'], centered_y_name
