"""Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for feature selection.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import copy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

from mcf import mcf_print_stats_functions as ps


def feature_selection(mcf_, data_df):
    """
    Feature selection using scikit-learn.

    Step 1: Set-aside a random sample for feature selection analysis (option)
    Step 2: Estimate classifier forest for treatment
    Step 3: Estimate regression (>=10 values) or classifier (<10 values) forest
            for outcome
            Estimation is 75%, prediction on 25% random sample
    Step 4: Permutate variable groups; create variable importance measure based
            on r2 or accuary; compute relative loss compared to baseline.
    Step 5: Deselect variables if loss is less than threshold
    Step 6: Do not deselect variable if on list of protected variables. Do not
            deselect variable if highly correlated with other variable also
            on list of variables to deselect.
    """
    var_dic, gen_dic, fs_dic = mcf_.var_dict, mcf_.gen_dict, mcf_.fs_dict
    var_x_type, var_x_values = copy(mcf_.var_x_type), copy(mcf_.var_x_values)
    boot = mcf_.cf_dict['boot']
    if gen_dic['with_output']:
        ps.print_mcf(gen_dic, 'Feature selection', summary=False)
    # 1. Set aside sample for feature selection
    if fs_dic['other_sample']:
        data_fs_df, data_mcf_df = train_test_split(
            data_df, test_size=fs_dic['other_sample_share'], random_state=42)
    else:
        data_mcf_df, data_fs_df = data_df.copy(), data_df.copy()
    # Create dummies
    all_names = var_x_type.keys()
    # Remove all covariates that are discretized for GATE estimation
    x_cat_names = [name + 'CATV'
                   for name in all_names if name + 'CATV' in all_names]
    x_names = [name for name in all_names if name not in x_cat_names]
    x_names_org = x_names.copy()
    names_unordered = []
    for x_name in x_names:
        if var_x_type[x_name] > 0:
            names_unordered.append(x_name)
    if names_unordered:  # List is not empty
        unordered_dummy_names = {}  # Dict contains dummy name correspondence
        for idx, name in enumerate(names_unordered):
            x_dummies = pd.get_dummies(data_fs_df[name], columns=[name])
            x_dummies_names = [name + str(ddd) for ddd in x_dummies.columns]
            unordered_dummy_names[name] = x_dummies.columns = x_dummies_names
            if idx == 0:
                x_all_dummies = x_dummies
            else:
                x_all_dummies = pd.concat([x_all_dummies, x_dummies], axis=1,
                                          copy=True)
        data_fs_df = pd.concat([data_fs_df, x_all_dummies], axis=1, copy=True)
        # Remove names of unordered variables
        x_names = [name for name in x_names if name not in names_unordered]
        # Add their dummy names
        x_names.extend(x_all_dummies.columns)
    # Test sample is used for variable importance calculations
    (y_train_df, y_test_df, d_train_df, d_test_df, x_train_df, x_test_df
     ) = train_test_split(
         data_fs_df[var_dic['y_tree_name']], data_fs_df[var_dic['d_name']],
         data_fs_df[x_names], test_size=0.25, random_state=42)
    max_workers = 1 if gen_dic['replication'] else gen_dic['mp_parallel']
    params = {'n_estimators': boot, 'max_features': 'sqrt', 'bootstrap': True,
              'oob_score': False, 'n_jobs': max_workers, 'random_state': 42,
              'verbose': False}
    d_rf_obj = RandomForestClassifier(**params)
    if y_as_classifier := int(y_train_df.nunique().iloc[0]) < 10:
        y_rf_obj = RandomForestClassifier(**params)
    else:
        y_rf_obj = RandomForestRegressor(**params)
    d_rf_obj.fit(x_train_df.to_numpy(), d_train_df.to_numpy().ravel())
    y_rf_obj.fit(x_train_df.to_numpy(), y_train_df.to_numpy().ravel())
    d_pred = d_rf_obj.predict(x_test_df.to_numpy())
    y_pred = y_rf_obj.predict(x_test_df.to_numpy())
    d_np = d_test_df.to_numpy().ravel()
    y_np = y_test_df.to_numpy().ravel()
    score_d_full = accuracy_score(d_np, d_pred, normalize=True)
    score_y_full = score_for_y(y_np, y_pred, y_as_classifier)
    # Compute variable importance for all variables (dummies as group)
    x_names_to_delete = []
    if gen_dic['with_output']:
        vi_information = pd.DataFrame(columns=x_names_org,
                                      index=['score_w/o_x_y', 'score_w/o_x_d',
                                             'rel_diff_y_%', 'rel_diff_d_%'])
    for name in x_names_org:
        if name in names_unordered:
            names_to_shuffle = unordered_dummy_names[name]
        else:
            names_to_shuffle = name
        x_all_rnd_df = x_test_df.copy().reset_index(drop=True)
        x_rnd_df = x_test_df[names_to_shuffle].sample(frac=1, random_state=42)
        x_all_rnd_df[names_to_shuffle] = x_rnd_df.reset_index(drop=True)
        d_pred_rnd = d_rf_obj.predict(x_all_rnd_df.to_numpy())
        y_pred_rnd = y_rf_obj.predict(x_all_rnd_df.to_numpy())
        d_score = accuracy_score(d_np, d_pred_rnd, normalize=True)
        y_score = score_for_y(y_np, y_pred_rnd, y_as_classifier)
        d_rel_diff = (score_d_full - d_score) / score_d_full
        y_rel_diff = (score_y_full - y_score) / score_y_full
        if ((d_rel_diff < fs_dic['rf_threshold'])
                and (y_rel_diff < fs_dic['rf_threshold'])):
            x_names_to_delete.append(name)
        if gen_dic['with_output']:
            vi_information[name] = [y_score, d_score,
                                    y_rel_diff*100, d_rel_diff*100]
    ps.print_mcf(gen_dic, '=' * 100 + '\nFeature selection' + '\n' + '- ' * 50,
                 summary=True)
    if gen_dic['with_output']:
        ps.print_mcf(gen_dic, f'\nFull score y: {score_y_full:6.3f} '
                     f'Full score d: {score_d_full:6.3f} '
                     f'Threshold in %: {fs_dic["rf_threshold"]:4.2%}\n',
                     summary=False)
        with pd.option_context('display.max_rows', None,
                               'display.expand_frame_repr', True,
                               'chop_threshold', 1e-13):
            ps.print_mcf(gen_dic, vi_information.transpose().sort_values(
                by=['rel_diff_y_%', 'rel_diff_d_%'], ascending=False),
                         summary=False)
    if x_names_to_delete:
        forbidden_to_delete_vars = (
            var_dic['x_name_always_in'] + var_dic['x_name_remain']
            + var_dic['z_name'] + var_dic['z_name_list'])
        names_to_remove = [name for name in x_names_to_delete
                           if name not in forbidden_to_delete_vars]
        if names_to_remove[:-1]:
            # Check if two vars to remove are highly correlated.
            # Use full data for this exercise.
            weg_corr = data_df[names_to_remove].corr()
            names_weg, do_not_remove = names_to_remove.copy(), []
            if gen_dic['with_output']:
                with pd.option_context('display.max_rows', None,
                                       'display.expand_frame_repr', True,
                                       'display.width', 120,
                                       'chop_threshold', 1e-13):
                    ps.print_mcf(gen_dic, '\nCorrelation of variables to be '
                                 'deleted\n', weg_corr, summary=False)
            for idx, name_weg in enumerate(names_weg[:-1]):
                for name_weg2 in names_weg[idx+1:]:
                    if np.abs(weg_corr.loc[name_weg, name_weg2]) > 0.5:
                        do_not_remove.append(name_weg)
            names_to_remove = [name for name in names_weg
                               if name not in do_not_remove]
        if names_to_remove:
            for name_weg in names_to_remove:
                var_x_type.pop(name_weg)
                var_x_values.pop(name_weg)
                var_dic['x_name'].remove(name_weg)
        if gen_dic['with_output']:
            ps.print_mcf(
                gen_dic, '\nVariables deleted: ' + ' '.join(names_to_remove)
                + '\nVariables kept:    ' + ' '.join(var_dic['x_name'])
                + '\n' + '-' * 100, summary=True)
    else:
        if gen_dic['with_output']:
            ps.print_mcf(gen_dic, '\nNo variables removed in feature'
                         ' selection' + '\n' + '-' * 100, summary=True)
    mcf_.var_dict = var_dic
    mcf_.var_x_type, mcf_.var_x_values = var_x_type, var_x_values
    return data_mcf_df


def score_for_y(y_true, y_pred, y_as_classifier):
    """Compute score dependending on type of y."""
    if y_as_classifier:
        return accuracy_score(y_true, y_pred, normalize=True)
    return r2_score(y_true, y_pred)
