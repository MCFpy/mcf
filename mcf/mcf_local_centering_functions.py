"""Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for local centering.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from mcf import mcf_data_functions as data
from mcf import mcf_general as gp
from mcf import mcf_print_stats_functions as ps


def local_centering(mcf_, tree_df, fill_y_df, train=True):
    """Locally center all outcome variables."""
    lc_dic, gen_dic, var_dic = mcf_.lc_dict, mcf_.gen_dict, mcf_.var_dict
    var_x_type, data_train_dict = mcf_.var_x_type, mcf_.data_train_dict
    cf_dic = mcf_.cf_dict
    ps.print_mcf(gen_dic, '=' * 100 + '\nLocal Centering' + '\n', summary=True)
    seed = 9324561
    if gen_dic['with_output'] and gen_dic['verbose']:
        txt = '\nLocal centering with Random Forests estimated '
        if lc_dic['cs_cv']:
            txt += 'by cross-validation'
        else:
            txt += 'in an independent random sample'
        ps.print_mcf(gen_dic, txt, summary=False)
    x_name, x_type = gp.get_key_values_in_list(var_x_type)
    # Get correct dataframes depending on task and cross-validation
    if train:
        if lc_dic['cs_cv']:   # Crossvalidate ... only tree data is used
            tree_mcf_df, fill_y_mcf_df = tree_df.copy(), fill_y_df
        else:  # Use lc_dic['cs_share'] of data for lc (and cs) estim. only
            # Take the same share of obs. from both input samples
            tree_mcf_df, tree_lc_df = train_test_split(
                tree_df, test_size=lc_dic['cs_share'], random_state=seed)
            fill_y_mcf_df, fill_y_lc_df = train_test_split(
                fill_y_df, test_size=lc_dic['cs_share'], random_state=seed)
            data_lc_df = pd.concat([tree_lc_df, fill_y_lc_df], axis=0)
            x_lc_df, _ = data.get_x_data(data_lc_df, x_name)
        x_fy_df, _ = data.get_x_data(fill_y_mcf_df, x_name)
    else:
        tree_mcf_df, fill_y_mcf_df = tree_df, None
    x_tree_df, obs_mcf = data.get_x_data(tree_mcf_df, x_name)
    names_unordered = [x_name[j] for j, val in enumerate(x_type) if val > 0]
    # Get covariates
    if names_unordered:  # List is not empty; predict and train
        x_tree_df, dummy_names = data.dummies_for_unord(
            x_tree_df, names_unordered, data_train_dict=data_train_dict)
    x_tree_np = x_tree_df.to_numpy()
    if train:
        if names_unordered:  # List is not empty
            x_fy_df, _ = data.dummies_for_unord(
                x_fy_df, names_unordered, data_train_dict=data_train_dict)
            if not lc_dic['cs_cv']:
                x_lc_df, _ = data.dummies_for_unord(
                    x_lc_df, names_unordered, data_train_dict=data_train_dict)
        if not lc_dic['cs_cv']:
            x_lc_np = x_lc_df.to_numpy()
        x_fy_np = x_fy_df.to_numpy()
        max_workers = 1 if gen_dic['replication'] else gen_dic['mp_parallel']
        params = {'n_estimators': cf_dic['boot'], 'max_features': 'sqrt',
                  'bootstrap': True, 'oob_score': False, 'n_jobs': max_workers,
                  'random_state': seed, 'verbose': False}
        forests_all = []
        y_x_tree = np.zeros((len(x_tree_np), len(var_dic['y_name'])))
        y_x_fy = np.zeros((len(x_fy_np), len(var_dic['y_name'])))
        if lc_dic['cs_cv']:
            index = np.arange(obs_mcf)       # indices
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(index)
            index_folds = np.array_split(index, lc_dic['cs_cv_k'])
            y_tree_np = tree_mcf_df[var_dic['y_name']].to_numpy()
            for fold_pred in range(lc_dic['cs_cv_k']):
                fold_train = [x for idx, x in enumerate(index_folds)
                              if idx != fold_pred]
                index_train = np.hstack(fold_train)
                index_pred = index_folds[fold_pred]
                x_train = x_tree_np[index_train]
                x_pred = x_tree_np[index_pred]
                forests_y = []
                for idx, y_name in enumerate(var_dic['y_name']):
                    y_train = y_tree_np[index_train, idx].ravel()
                    if y_as_classifier := len(np.unique(y_tree_np)) < 10:
                        y_rf_obj = RandomForestClassifier(**params)
                    else:
                        y_rf_obj = RandomForestRegressor(**params)
                    y_rf_obj.fit(x_train, y_train)
                    y_x_tree[index_pred, idx] = y_rf_obj.predict(x_pred)
                    y_x_fy[:, idx] += y_rf_obj.predict(x_fy_np)
                    forests_y.append(deepcopy(y_rf_obj))
            if gen_dic['with_output']:
                for idx, y_name in enumerate(var_dic['y_name']):
                    print_lc_info(y_name, y_as_classifier, gen_dic,
                                  y_x_tree[:, idx].ravel(),
                                  y_tree_np[:, idx].ravel())
                forests_all.append(deepcopy(forests_y))
            y_x_fy /= lc_dic['cs_cv_k']
        else:
            forests_y = []
            for idx, y_name in enumerate(var_dic['y_name']):
                y_lc_df = data_lc_df[y_name]
                if y_as_classifier := y_lc_df.nunique() < 10:
                    y_rf_obj = RandomForestClassifier(**params)
                else:
                    y_rf_obj = RandomForestRegressor(**params)
                y_rf_obj.fit(x_lc_np, y_lc_df.to_numpy().ravel())
                y_x_tree[:, idx] = y_rf_obj.predict(x_tree_np)
                y_x_fy[:, idx] = y_rf_obj.predict(x_fy_np)
                forests_y.append(deepcopy(y_rf_obj))
                if gen_dic['with_output']:
                    print_lc_info(y_name, y_as_classifier, gen_dic,
                                  y_x_tree[:, idx].ravel(),
                                  tree_mcf_df[y_name].to_numpy().ravel())
            forests_all.append(deepcopy(forests_y))
        if isinstance(var_dic['y_name'], (list, tuple)):
            y_name = [y_name]
        y_cent_name = [name + '_LC' for name in var_dic['y_name']]
        y_x_name = [name + '_EY_X' for name in var_dic['y_name']]
        var_dic['y_name_lc'], var_dic['y_name_ey_x'] = y_cent_name, y_x_name
        y_m_yx_tree = tree_mcf_df[var_dic['y_name']].to_numpy() - y_x_tree
        y_m_yx_fy = fill_y_mcf_df[var_dic['y_name']].to_numpy() - y_x_fy
        y_m_yx_tree_df = pd.DataFrame(data=y_m_yx_tree, columns=y_cent_name)
        y_m_yx_fy_df = pd.DataFrame(data=y_m_yx_fy, columns=y_cent_name)
        y_x_tree_df = pd.DataFrame(data=y_x_tree, columns=y_x_name)
        y_x_fy_df = pd.DataFrame(data=y_x_fy, columns=y_x_name)
        tree_mcf_df = tree_mcf_df.reset_index(drop=True)
        fill_y_mcf_df = fill_y_mcf_df.reset_index(drop=True)
        tree_add_y_lc_df = pd.concat((tree_mcf_df, y_m_yx_tree_df, y_x_tree_df
                                      ), axis=1)
        fill_add_y_lc_df = pd.concat((fill_y_mcf_df, y_m_yx_fy_df, y_x_fy_df),
                                     axis=1)
        if gen_dic['with_output']:
            all_y_name = [*var_dic['y_name'], *y_cent_name, *y_x_name]
            ps.print_descriptive_df(
                gen_dic, pd.concat((tree_add_y_lc_df, fill_add_y_lc_df),
                                   axis=0), varnames=all_y_name, summary=True)
        lc_dic['forests'] = forests_all
        var_dic = adjust_y_names(var_dic, gen_dic, var_dic['y_name'],
                                 y_cent_name, summary=True)
        y_x_df = None
    else:  # Predict Ey|X
        y_x_np = np.zeros((len(x_tree_np), len(var_dic['y_name'])))
        for forest in lc_dic['forests']:
            for idx, forest_y in enumerate(forest):
                y_x_np[:, idx] += forest_y.predict(x_tree_np)
        y_x_np /= len(lc_dic['forests'])
        tree_add_y_lc_df = fill_add_y_lc_df = None
        y_x_df = pd.DataFrame(data=y_x_np, columns=var_dic['y_name_ey_x'])
    if train:
        mcf_.lc_dict = lc_dic
        mcf_.var_dict = var_dic
    return tree_add_y_lc_df, fill_add_y_lc_df, y_x_df


def print_lc_info(y_name, y_as_classifier, gen_dic, y_pred, y_true,
                  summary=True):
    """Compute some basic info on outcome regression in local centering."""
    if y_as_classifier:
        fit = accuracy_score(y_true, y_pred, normalize=True)
        method = 'Accuracy score '
    else:
        fit = r2_score(y_true, y_pred)
        method = 'R2 '
    txt = '\n' + 100 * '-' + '\nFit of Ey|x for local centering.   '
    txt += method + f'for {y_name}: {fit:5.2%} \n' + '- ' * 50
    ps.print_mcf(gen_dic, txt, summary=summary)


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
        ps.print_mcf(gen_dic, txt, summary=summary)
    return var_dic
