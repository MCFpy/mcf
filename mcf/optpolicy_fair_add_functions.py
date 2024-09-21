"""Created on Wed May  1 16:35:19 2024.

Functions for correcting scores w.r.t. protected variables.

@author: MLechner

# -*- coding: utf-8 -*-
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from mcf import mcf_print_stats_functions as mcf_ps


def score_combinations(fair_scores_np, fair_scores_name, scores_np,
                       scores_name):
    """Find and create combinations of scores that can be tested."""
    fs_np, fs_name, s_np, s_name = check_if_variation(
        fair_scores_np, fair_scores_name, scores_np, scores_name)

    obs, no_scores = s_np.shape
    no_of_new_scores = int(no_scores * (no_scores - 1) / 2)
    test_np = np.zeros((obs, no_of_new_scores))
    test_fair_np = np.zeros_like(test_np)
    test_name, test_fair_name = [], []
    index = 0
    for idx in range(0, no_scores):
        for jdx in range(idx+1, no_scores):
            test_np[:, index] = s_np[:, jdx] - s_np[:, idx]
            test_fair_np[:, index] = fs_np[:, jdx] - fs_np[:, idx]
            test_name.append(s_name[jdx] + '_m_' + s_name[idx])
            test_fair_name.append(fs_name[jdx] + '_m_' + fs_name[idx])
            index += 1

    return test_np, test_fair_np, test_name, test_fair_name


def check_if_variation(fair_scores_np, fairscores_name, scores_np, scores_name):
    """Delete scores without variation."""
    const = 1e-8
    score_const = (np.std(fair_scores_np, axis=0)
                   + np.std(scores_np, axis=0)) > const

    positions_0 = np.where(score_const)[0]
    if not score_const.all():
        fs_np = fair_scores_np[:, positions_0].copy()
        s_np = scores_np[:, positions_0].copy()
        fs_name = [fairscores_name[idx] for idx in positions_0]
        s_name = [scores_name[idx] for idx in positions_0]
    else:
        fs_np, fs_name = fair_scores_np.copy(), fairscores_name[:]
        s_np, s_name = scores_np.copy(), scores_name[:]

    return fs_np, fs_name, s_np, s_name


def change_variable_names_fair(optp_, fairscore_name):
    """Change variable names to account for fair scores."""
    if optp_.var_dict['polscore_desc_name'] is not None:
        optp_.var_dict['polscore_desc_name'].extend(
            optp_.var_dict['polscore_name'].copy())
    else:
        optp_.var_dict['polscore_desc_name'] = optp_.var_dict[
            'polscore_name'].copy()
    # Remove duplicates from list without changing order
    optp_.var_dict['polscore_desc_name'] = remove_duplicates(
        optp_.var_dict['polscore_desc_name'])

    if (optp_.var_dict['vi_x_name'] is not None
            and optp_.var_dict['protected_ord_name'] is not None):
        optp_.var_dict['vi_x_name'].extend(
            optp_.var_dict['protected_ord_name'])
    elif (optp_.var_dict['vi_x_name'] is None
          and optp_.var_dict['protected_ord_name'] is not None):
        optp_.var_dict['vi_x_name'] = optp_.var_dict['protected_ord_name']

    if (optp_.var_dict['vi_to_dummy_name'] is not None
            and optp_.var_dict['protected_unord_name'] is not None):
        optp_.var_dict['vi_to_dummy_name'].extend(
            optp_.var_dict['protected_unord_name'])
    elif (optp_.var_dict['vi_to_dummy_name'] is None
          and optp_.var_dict['protected_ord_name'] is not None):
        optp_.var_dict['vi_to_dummy_name'] = optp_.var_dict[
            'protected_unord_name']

    if optp_.var_dict['vi_x_name'] is not None:
        optp_.var_dict['vi_x_name'] = unique_list(optp_.var_dict['vi_x_name'])
    if optp_.var_dict['vi_to_dummy_name'] is not None:
        optp_.var_dict['vi_to_dummy_name'] = unique_list(
            optp_.var_dict['vi_to_dummy_name'])
    optp_.var_dict['polscore_name'] = fairscore_name


def remove_duplicates(lst):
    """Remove duplicates from list without changing order."""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def unique_list(list_tuple):
    """Remove duplicate elements from list without changing order."""
    unique = []
    _ = [unique.append(item)
         for item in list_tuple if item not in unique]
    return unique


def fair_stats(optp_, data_df, fairscore_names):
    """Compute descriptive statistics of fairness adjustments."""
    prot_names = optp_.var_dict['protected_name']
    score_names = optp_.var_dict['polscore_name']
    txt = '\n\nDescriptive statistics of adjusted and unadjusted scores'
    all_scores = [*score_names, *fairscore_names]
    stats = np.round(data_df[all_scores].describe().T, 3)
    corr = np.round(data_df[all_scores].corr() * 100)
    corr = corr.dropna(how='all', axis=0)
    corr = corr.dropna(how='all', axis=1)
    txt += '\n' + str(stats)
    txt += '\n\nCorrelation of adjusted and unadjusted scores in %'
    txt += '\n' + str(corr)
    correlations_np = np.zeros((len(all_scores), len(prot_names)))
    correlations_df = pd.DataFrame(correlations_np,
                                   index=all_scores, columns=prot_names)
    for score in all_scores:
        for prot in prot_names:
            corr = data_df[[score, prot]].corr()
            correlations_df.loc[score, prot] = round(corr.iloc[1, 0] * 100, 2)
    correlations_df = correlations_df.dropna(how='all', axis=0)
    correlations_df = correlations_df.dropna(how='all', axis=1)
    txt += ('\n' * 2 + 'Bivariate Correlations in %'
            + '\n' + str(correlations_df))

    mcf_ps.print_mcf(optp_.gen_dict, txt, summary=True)
    return txt


def var_to_std(var, constant):
    """Change variance to standard deviation."""
    var_neu = var.copy()
    mask_var = var < constant
    if mask_var.any():
        var_neu = np.where(mask_var, constant, var)
    return np.sqrt(var_neu)


def bound_std(std, bound, no_of_scores):
    """Lower bound for standard deviation to avoid explosion of values."""
    std_neu = std.copy()
    for idx in range(no_of_scores):
        lower = bound * np.mean(std[:, idx])
        mask_var = std[:, idx] < lower
        if mask_var.any():
            std_neu[:, idx] = np.where(mask_var, lower, std[:, idx])
    return std_neu


def data_quantilized(optp_, protected_in_np, material_in_np, seed=12345):
    """Prepare the data (discretize) for fairness quantilization."""
    txt = ''
    # Materially relevant variables if they exist, and if relevant
    disc_methods = optp_.fair_dict['discretization_methods']
    if optp_.fair_dict['material_disc_method'] in disc_methods:
        if material_in_np is None:
            material_np = None
        else:
            if optp_.fair_dict['material_disc_method'] == 'EqualCell':
                material_np = discretize_equalcell(
                    material_in_np, optp_.fair_dict['material_max_groups'])
            elif optp_.fair_dict['material_disc_method'] == 'Kmeans':
                material_np = discretize_kmeans(
                    material_in_np, optp_.fair_dict['material_max_groups'],
                    seed=seed)
            else:
                raise ValueError('Unknown discretization method.')
            txt += ('\nMaterially relevant features discretized: '
                    f'{len(np.unique(material_np))} cells.'
                    )
    else:
        material_np = material_in_np

    # Protected variables, if relevant
    if optp_.fair_dict['protected_disc_method'] in disc_methods:
        if optp_.fair_dict['protected_disc_method'] == 'EqualCell':
            protected_np = discretize_equalcell(
                protected_in_np, optp_.fair_dict['protected_max_groups'])
        elif optp_.fair_dict['protected_disc_method'] == 'Kmeans':
            protected_np = discretize_kmeans(
                protected_in_np, optp_.fair_dict['protected_max_groups'],
                seed=seed+124335)
        else:
            raise ValueError('Unknown discretization method.')
        txt += ('\nProtected features discretized: '
                f'{len(np.unique(protected_np))} cells.'
                )
    else:
        protected_np = protected_in_np

    return protected_np, material_np, txt


def discretize_kmeans(data_in_np, max_groups, seed=12345):
    """Discretize using cells of similar size (for cont. features)."""
    data_np = KMeans(
        n_clusters=max_groups, init='k-means++', n_init='auto', max_iter=1000,
        algorithm='lloyd', random_state=seed, tol=1e-5, verbose=0, copy_x=True
        ).fit_predict(data_in_np.copy())

    return data_np.reshape(-1, 1)


def discretize_equalcell(data_in_np, max_groups):
    """Discretize using cells of similar size (for cont. features)."""
    data_np = data_in_np.copy()

    cols = data_in_np.shape[1]
    for col_idx in range(cols):
        unique_val = np.unique(data_in_np[:, col_idx])
        if len(unique_val) > max_groups:
            data_np[:, col_idx] = pd.qcut(data_in_np[:, col_idx],
                                          q=max_groups, labels=False)
    if cols > 1:
        _, data_np = np.unique(data_np, axis=0, return_inverse=True)  # 2.0.0?
    return data_np.reshape(-1, 1)
