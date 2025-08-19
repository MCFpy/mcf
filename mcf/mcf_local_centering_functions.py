"""Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for local centering.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from mcf import mcf_data_functions as mcf_data
from mcf import mcf_estimation_generic_functions as mcf_gf
from mcf.mcf_general import get_key_values_in_list, to_numpy_big_data
from mcf.mcf_general_sys import print_mememory_statistics
from mcf import mcf_print_stats_functions as mcf_ps


def local_centering(mcf_, tree_df, fill_y_df=None, train=True, seed=9324561,
                    title=''
                    ):
    """Locally center all outcome variables."""
    lc_dic, gen_dic, var_dic = mcf_.lc_dict, mcf_.gen_dict, mcf_.var_dict
    var_x_type, data_train_dict = mcf_.var_x_type, mcf_.data_train_dict
    cf_dic, int_dic = mcf_.cf_dict, mcf_.int_dict
    if gen_dic['with_output']:
        print_lc_header(gen_dic, lc_dic, title=title)

    y_0_dtype = (np.float64 if len(tree_df) < int_dic['obs_bigdata']
                 else np.float32)

    x_name, x_type = get_key_values_in_list(var_x_type)
    # Get correct dataframes depending on task and cross-validation
    if train:
        if lc_dic['cs_cv']:   # Crossvalidate ... only tree data is used
            tree_mcf_df, fill_y_mcf_df = tree_df.copy(), fill_y_df
        else:  # Use lc_dic['cs_share'] of data for lc (and cs) estim. only
            # Take the same share of obs. from both input samples
            if fill_y_df is None:
                tree_mcf_df, tree_lc_df = train_test_split(
                    tree_df, test_size=lc_dic['cs_share'], random_state=seed)
            else:
                tree_mcf_df, tree_lc_df = train_test_split(
                    tree_df,
                    test_size=lc_dic['cs_share'],
                    random_state=seed
                    )
                fill_y_mcf_df, fill_y_lc_df = train_test_split(
                    fill_y_df,
                    test_size=lc_dic['cs_share'],
                    random_state=seed
                    )
            data_lc_df = pd.concat((tree_lc_df, fill_y_lc_df), axis=0)
            x_lc_df, _ = mcf_data.get_x_data(data_lc_df, x_name)
        if fill_y_df is not None:
            x_fy_df, _ = mcf_data.get_x_data(fill_y_mcf_df, x_name)
    else:
        tree_mcf_df, fill_y_mcf_df = tree_df, None

    x_tree_df, obs_mcf = mcf_data.get_x_data(tree_mcf_df, x_name)

    # Create dummies for categorical variables
    names_unordered = [x_name[j] for j, val in enumerate(x_type) if val > 0]
    # Get covariates (as dummies if categorical)
    if names_unordered:  # List is not empty; predict and train
        x_tree_df, dummy_names = mcf_data.dummies_for_unord(
            x_tree_df, names_unordered, data_train_dict=data_train_dict)

    x_tree_np = to_numpy_big_data(x_tree_df, int_dic['obs_bigdata'])
    print_mememory_statistics(gen_dic, 'Local centering: Prior to estimation')
    lc_r2_txt = None
    if train:
        max_workers = 1 if int_dic['replication'] else gen_dic['mp_parallel']
        if names_unordered:
            if fill_y_df is not None:
                x_fy_df, _ = mcf_data.dummies_for_unord(
                    x_fy_df, names_unordered, data_train_dict=data_train_dict)
            if not lc_dic['cs_cv']:
                x_lc_df, _ = mcf_data.dummies_for_unord(
                    x_lc_df, names_unordered, data_train_dict=data_train_dict)
        if not lc_dic['cs_cv']:
            x_lc_np = to_numpy_big_data(x_lc_df, int_dic['obs_bigdata'])
        if fill_y_df is not None:
            x_fy_np = to_numpy_big_data(x_fy_df, int_dic['obs_bigdata'])

        y_tree_np = to_numpy_big_data(tree_mcf_df[var_dic['y_name']],
                                      int_dic['obs_bigdata'])

        # Find best estimator for first outcome
        (estimator, params, txt_sel, _, transform_x, txt_mse
         ) = mcf_gf.best_regression(
            x_tree_np,  y_tree_np[:, 0].ravel(), estimator=lc_dic['estimator'],
            boot=cf_dic['boot'], seed=seed, max_workers=max_workers,
            test_share=lc_dic['cs_share'],
            cross_validation_k=lc_dic['cs_cv_k'] if lc_dic['cs_cv'] else 0,
            obs_bigdata=int_dic['obs_bigdata'])
        print_mememory_statistics(gen_dic,
                                  'Local centering training: '
                                  'After best_regression')
        forests_all = []
        y_x_tree = np.zeros((len(x_tree_np), len(var_dic['y_name'])),
                            dtype=y_0_dtype)
        if fill_y_df is not None:
            y_x_fy = np.zeros((len(x_fy_np), len(var_dic['y_name'])),
                              dtype=y_0_dtype)
        if lc_dic['cs_cv']:
            index = np.arange(obs_mcf)       # indices
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(index)
            index_folds = np.array_split(index, lc_dic['cs_cv_k'])
            for fold_pred in range(lc_dic['cs_cv_k']):
                fold_train = [x for idx, x in enumerate(index_folds)
                              if idx != fold_pred]
                index_train = np.hstack(fold_train)
                index_pred = index_folds[fold_pred]
                if transform_x:
                    scaler, x_train, x_pred = mcf_gf.scale(
                        x_tree_np[index_train], x_tree_np[index_pred])
                    if fill_y_df is not None:
                        x_fy_np_ = scaler.transform(x_fy_np)
                else:
                    x_train = x_tree_np[index_train]
                    x_pred = x_tree_np[index_pred]
                    if fill_y_df is not None:
                        x_fy_np_ = x_fy_np.copy()
                    scaler = None
                forests_y = []
                for idx, y_name in enumerate(var_dic['y_name']):
                    y_train = y_tree_np[index_train, idx].ravel()
                    y_obj = mcf_gf.regress_instance(estimator, params)
                    if y_obj is None:
                        mean = np.average(y_train)
                        y_x_tree[index_pred, idx] = mean
                        if fill_y_df is not None:
                            y_x_fy[:, idx] += mean
                        y_obj = mean
                    else:
                        y_obj.fit(x_train, y_train)
                        y_x_tree[index_pred, idx] = y_obj.predict(x_pred)
                        if fill_y_df is not None:
                            y_x_fy[:, idx] += y_obj.predict(x_fy_np_)
                    forests_y.append((deepcopy(y_obj), deepcopy(scaler),))

                    print_mememory_statistics(gen_dic,
                                              'Local centering training: '
                                              'During cross-fitting')

                forests_all.append(deepcopy(forests_y))
            if gen_dic['with_output']:
                for idx, y_name in enumerate(var_dic['y_name']):
                    txt_return = print_lc_info(
                        y_name, False, gen_dic, txt_sel,
                        y_x_tree[:, idx].ravel(), y_tree_np[:, idx].ravel())
                    if idx == 0:
                        lc_r2_txt = txt_return
            if fill_y_df is not None:
                y_x_fy /= lc_dic['cs_cv_k']
        else:
            forests_y = []
            if transform_x:
                scaler, x_lc_np_, x_tree_np_ = mcf_gf.scale(x_lc_np,
                                                            x_test=x_tree_np)
                if fill_y_df is not None:
                    x_fy_np_ = scaler.transform(x_fy_np)
            else:
                x_lc_np_, x_tree_np_ = x_lc_np, x_tree_np
                if fill_y_df is not None:
                    x_fy_np_ = x_fy_np
                scaler = None
            for idx, y_name in enumerate(var_dic['y_name']):
                y_lc_np = to_numpy_big_data(
                    data_lc_df[y_name], int_dic['obs_bigdata']).ravel()
                y_obj = mcf_gf.regress_instance(estimator, params)
                if y_obj is None:
                    mean = np.average(y_lc_np)
                    y_x_tree[:, idx] = mean
                    if fill_y_df is not None:
                        y_x_fy[:, idx] = mean
                    y_obj = mean
                else:
                    y_obj.fit(x_lc_np_, y_lc_np)
                    y_x_tree[:, idx] = y_obj.predict(x_tree_np_)
                    if fill_y_df is not None:
                        y_x_fy[:, idx] = y_obj.predict(x_fy_np_)
                forests_y.append((deepcopy(y_obj), deepcopy(scaler)))
                if gen_dic['with_output']:
                    txt_return = print_lc_info(
                        y_name, False, gen_dic, txt_sel,
                        y_x_tree[:, idx].ravel(),
                        to_numpy_big_data(tree_mcf_df[y_name],
                                          int_dic['obs_bigdata']).ravel())
                    if idx == 0:
                        lc_r2_txt = txt_return
            print_mememory_statistics(gen_dic,
                                      'Local centering Prediction: '
                                      'During prediction')
            forests_all.append(deepcopy(forests_y))
        if isinstance(var_dic['y_name'], str):
            var_dic['y_name'] = [var_dic['y_name']]
        y_cent_name = [name + '_lc' for name in var_dic['y_name']]
        y_x_name = [name + '_Ey_x' for name in var_dic['y_name']]
        var_dic['y_name_lc'], var_dic['y_name_ey_x'] = y_cent_name, y_x_name

        # Use predictions to center outcomes
        y_m_yx_tree = to_numpy_big_data(tree_mcf_df[var_dic['y_name']],
                                        int_dic['obs_bigdata']) - y_x_tree
        # Transfer back to pandas DataFrames
        y_m_yx_tree_df = pd.DataFrame(data=y_m_yx_tree, columns=y_cent_name)
        y_x_tree_df = pd.DataFrame(data=y_x_tree, columns=y_x_name)
        # Reset indices for existing dataframes
        tree_mcf_df = tree_mcf_df.reset_index(drop=True)
        tree_add_y_lc_df = pd.concat((tree_mcf_df, y_m_yx_tree_df, y_x_tree_df
                                      ), axis=1)
        if fill_y_df is not None:
            y_m_yx_fy = to_numpy_big_data(fill_y_mcf_df[var_dic['y_name']],
                                          int_dic['obs_bigdata']) - y_x_fy
            y_m_yx_fy_df = pd.DataFrame(data=y_m_yx_fy, columns=y_cent_name)
            y_x_fy_df = pd.DataFrame(data=y_x_fy, columns=y_x_name)
            fill_y_mcf_df = fill_y_mcf_df.reset_index(drop=True)
            fill_add_y_lc_df = pd.concat(
                (fill_y_mcf_df, y_m_yx_fy_df, y_x_fy_df), axis=1)

        if gen_dic['with_output']:
            mcf_ps.print_mcf(gen_dic, txt_mse, summary=False)
            all_y_name = [*var_dic['y_name'], *y_cent_name, *y_x_name]
            if fill_y_df is None:
                mcf_ps.print_descriptive_df(
                    gen_dic, tree_add_y_lc_df, varnames=all_y_name,
                    summary=False)
            else:
                mcf_ps.print_descriptive_df(
                    gen_dic,
                    pd.concat((tree_add_y_lc_df, fill_add_y_lc_df), axis=0),
                    varnames=all_y_name, summary=False)

        lc_dic['forests'] = forests_all
        var_dic = adjust_y_names(var_dic, gen_dic, var_dic['y_name'],
                                 y_cent_name, summary=True)
        y_x_df = None
    else:  # Predict Ey|X
        y_x_np = np.zeros((len(x_tree_np), len(var_dic['y_name'])),
                          dtype=y_0_dtype)
        if lc_dic['forests'] == []:
            raise ValueError('Estimator used for centering are not saved for '
                             'prediction.')
        for forest in lc_dic['forests']:   # Loops over training folds
            for idx, forest_y in enumerate(forest):  # Loops over outcomes
                scaler = forest_y[1]
                if scaler is not None:
                    x_tree_np_ = scaler.transform(x_tree_np)
                else:
                    x_tree_np_ = x_tree_np
                if isinstance(forest_y[0], float):
                    y_x_np[:, idx] += forest_y[0]
                else:
                    y_x_np[:, idx] += forest_y[0].predict(x_tree_np_)
        y_x_np /= len(lc_dic['forests'])
        tree_add_y_lc_df = fill_add_y_lc_df = None
        y_x_df = pd.DataFrame(data=y_x_np, columns=var_dic['y_name_ey_x'])
    if train:
        mcf_.lc_dict = lc_dic
        mcf_.var_dict = var_dic
    if fill_y_df is None:
        fill_add_y_lc_df = None
    print_mememory_statistics(gen_dic,
                              'Local centering: End of function.')
    return tree_add_y_lc_df, fill_add_y_lc_df, y_x_df, lc_r2_txt


def print_lc_header(gen_dic, lc_dic, title=''):
    """Print the header."""
    mcf_ps.print_mcf(gen_dic, '=' * 100
                     + f'\nLocal Centering  {title}', summary=True)
    if gen_dic['verbose']:
        txt = '\nLocal centering is done '
        if lc_dic['cs_cv']:
            txt += 'by cross-validation'
        else:
            txt += 'in an independent random sample'
        mcf_ps.print_mcf(gen_dic, txt, summary=False)


def print_lc_info(y_name, y_as_classifier, gen_dic, txt_sel, y_pred, y_true,
                  summary=True):
    """Compute some basic info on outcome regression in local centering."""
    if y_as_classifier:
        fit = accuracy_score(y_true, y_pred, normalize=True)
        method = 'Accuracy score'
    else:
        fit = r2_score(y_true, y_pred)
        method = 'R2'
    txt = '\n' + 100 * '-' + '\nFit of Ey|x for local centering (Best method: '
    txt += (txt_sel + '). ' + method + f' for {y_name}: {fit:5.2%} \n'
            + '- ' * 50)
    mcf_ps.print_mcf(gen_dic, txt, summary=summary)
    txt_return = txt_sel + ' of Ey|x (' + method + f') for {y_name}: {fit:5.2%}'

    return txt_return


def adjust_y_names(var_dic, gen_dic, y_name_old, y_name_new, summary=True):
    """
    Switch variables names of y in dictionary.

    Parameters
    ----------
    var_dict : Dictionary. Variables.
    y_name_old : List of strings. Old variable names.
    y_name_new : List of strings. New variable names.
    with_output : Boolean.

    Returns
    -------
    var_dict : Dict. Modified variable names.

    """
    var_dic['y_tree_name_unc'] = var_dic['y_tree_name'].copy()
    for indx, y_name in enumerate(y_name_old):
        if (var_dic['y_tree_name'] is None
            or var_dic['y_tree_name'] == []
                or y_name == var_dic['y_tree_name'][0]):
            var_dic['y_tree_name'] = [y_name_new[indx]]
            break
    var_dic['y_name'] = y_name_new
    if gen_dic['with_output']:
        txt = '\n' + 'New variable to build trees in RF: '
        txt += f' {var_dic["y_tree_name"][0]}'
        mcf_ps.print_mcf(gen_dic, txt, summary=summary)

    return var_dic
