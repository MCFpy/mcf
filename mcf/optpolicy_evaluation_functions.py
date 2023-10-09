
"""
Created on Tue Jul 18 07:58:29 2023.

Provide functions for the evaluation of allocations.

# -*- coding: utf-8 -*-

@author: MLechner
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mcf import mcf_print_stats_functions as ps
from mcf import mcf_variable_importance_functions as vi


def evaluate(optp_, data_df, allocation_df, d_ok, polscore_desc_ok, desc_var):
    """Evalute allocations."""
    var_dic, gen_dic = optp_.var_dict, optp_.gen_dict
    allocations = allocation_df.columns
    results_dic = {}
    population = ['All', 'Switchers'] if d_ok else ['All']
    value, index = value_to_index_dic_fct(allocation_df['random'].to_numpy())
    scores = data_df[var_dic['polscore_name']].to_numpy()
    if polscore_desc_ok:
        scores_add = data_df[var_dic['polscore_desc_name']].to_numpy()
        no_of_vars = np.int16(np.around(scores_add.shape[1]
                                        / gen_dic['no_of_treat']))
        scores_add_idx = np.int16(
            np.reshape(np.arange(scores_add.shape[1]),
                       (no_of_vars, gen_dic['no_of_treat'])))
    else:
        scores_add = None
    titel = True
    for pop in population:
        for alloc_name in allocations:
            alloc_np = np.int16(np.round(allocation_df[alloc_name].to_numpy()))
            if pop == 'Switchers':
                switcher = alloc_np != np.int16(np.round(
                    allocation_df['observed'].to_numpy()))
                alloc_ = alloc_np[switcher]
                scores_ = scores[switcher]
                if polscore_desc_ok:
                    scores_add_ = scores_add[switcher]
            else:
                alloc_, scores_, = alloc_np, scores
                scores_add_ = scores_add
            indices_ = alloc_.copy()
            for idx, val in enumerate(value):
                indices_ = np.where(alloc_ == val, index[idx], indices_)
            score_sel = scores_[np.arange(len(scores_)), indices_]
            score_sel_new_df = pd.DataFrame(
                score_sel, columns=(var_dic['polscore_name'][0]+'alloc',))
            if polscore_desc_ok:
                scores_add_sel = np.zeros((len(scores_add_), no_of_vars))
                score_all_sel_name = []
                for idx, val_idx in enumerate(scores_add_idx):
                    scores_add_k = scores_add_[:, val_idx]
                    scores_add_sel[:, idx] = scores_add_k[
                        np.arange(len(scores_)), indices_]
                    score_all_sel_name.append(
                        var_dic['polscore_desc_name'][val_idx[0]]+'alloc')
                score_add_df = pd.DataFrame(scores_add_sel,
                                            columns=score_all_sel_name)
            else:
                scores_add_sel = score_all_sel_name = score_add_df = None
            desc_new_var_df_ = score_add_df if polscore_desc_ok else None
            results_dic = evaluation_of_alloc(
                optp_, results_dic, pop, alloc_name, score_sel_new_df,
                desc_new_var_df_, alloc_, titel)
            titel = False
    if gen_dic['with_output']:
        ps.print_mcf(gen_dic, '-' * 100, summary=True)
        if desc_var:
            data_new_df = data_df[desc_var].copy()
            for pop in population:
                for alloc_name in allocations:
                    alloc_np = np.int16(
                        np.round(allocation_df[alloc_name].to_numpy()))
                    if pop == 'Switchers':
                        switcher = alloc_np != np.int16(np.round(
                            allocation_df['observed'].to_numpy()))
                        alloc_ = alloc_np[switcher]
                        data_new_df_ = data_new_df[switcher].copy()
                    else:
                        alloc_, data_new_df_ = alloc_np, data_new_df.copy()
                    data_new_df_[alloc_name] = alloc_
                    txt = '\n' + '-' * 100 + '\nDescriptive statistics of '
                    txt += f'features for {alloc_name}'
                    if pop == 'Switchers':
                        txt += ' for switchers'
                    txt += '\n' + '-' * 100
                    ps.print_mcf(gen_dic, txt, summary=True)
                    ps.statistics_by_treatment(
                        gen_dic, data_new_df_, [alloc_name], desc_var,
                        only_next=False, summary=True, median_yes=False,
                        std_yes=False, balancing_yes=False)
    return results_dic


def evaluation_of_alloc(optp_, results_dic, pop, alloc_name, score_sel_new_df,
                        desc_new_var_df, allocation, titel=False):
    """Compute and print results of evaluations."""
    results_new_dic = deepcopy(results_dic)
    local_dic = {}
    if desc_new_var_df is not None:
        desc_var = desc_new_var_df.columns
    txt = ''
    if titel:
        scorevar = score_sel_new_df.columns
        txt += ('\n' + '-' * 100 + '\n' + 'Mean of variables /'
                ' treatment shares' + ' ' * 6 + f'{scorevar[0]:10}')
        if desc_new_var_df is not None:
            for var in desc_var:
                txt += f' {var:10s}'
        txt += ''.join([f'{s:8d}' for s in optp_.gen_dict['d_values']])
        txt += '\n'
    name = pop + ' ' + alloc_name
    mean1 = score_sel_new_df.mean().to_list()
    txt += f'{name:40} {mean1[0]:10.2f}'
    local_dic[name] = mean1
    if desc_new_var_df is not None:
        for var in desc_var:
            mean = desc_new_var_df[var].mean()
            txt += f' {mean:10.2f}'
            local_dic[var] = mean
    treat_shares = get_treat_shares(allocation, optp_.gen_dict['d_values'])
    treat_shares_s = [f'{s:>8.2%}' for s in treat_shares]
    txt += ' ' * 6 + ''.join(treat_shares_s)
    if optp_.gen_dict['with_output']:
        ps.print_mcf(optp_.gen_dict, txt, summary=True)
    results_new_dic[alloc_name] = local_dic.copy()
    return results_new_dic


def get_treat_shares(allocation, d_values):
    """Get treatment shares."""
    values_, treat_counts = np.unique(allocation, return_counts=True)
    treat_shares_ = treat_counts / len(allocation)
    treat_shares, jdx = [0] * len(d_values), 0
    if list(values_):
        for idx, vals in enumerate(d_values):
            if values_[jdx] == vals:
                treat_shares[idx] = treat_shares_[jdx]
                jdx += 1
    return treat_shares


def evaluate_multiple(optp_, allocations_dic, scores_df):
    """Compute and print results for multiple allocation."""
    txt = ('\n' * 2 + '-' * 100 + '\n' + f'{"Allocation":12s}'
           + 'Shares:  '
           + ' '.join([f'{s:7d}' for s in optp_.gen_dict['d_values']])
           + '     Sum / Mean / Variance of Potential Outcomes')
    scores_np = scores_df.to_numpy()
    for alloc in allocations_dic:
        allocation = np.int16(np.round(allocations_dic[alloc].to_numpy()))
        if allocation.shape[1] > 1:
            allocation = allocation[:, 0]
        if (np.min(allocation) < min(optp_.gen_dict['d_values'])
                or np.max(allocation) < max(optp_.gen_dict['d_values'])):
            raise ValueError('Inconistent definition of treatment values')
        treat_shares = get_treat_shares(allocation, optp_.gen_dict['d_values'])
        index = np.int16(np.arange(len(optp_.gen_dict['d_values'])))
        indices_ = allocation.copy()
        idx_row = np.arange(len(scores_np))
        for idx, val in enumerate(np.int16(optp_.gen_dict['d_values'])):
            indices_ = np.where(allocation == val, index[idx], indices_)
        score_sel = scores_np[idx_row, indices_]
        mean, variance = np.mean(score_sel), np.var(score_sel)
        summ = np.sum(score_sel)
        txt += f'\n{alloc:20s}: '
        treat_shares_s = [f'{s:>8.2%}' for s in treat_shares]
        txt += ' ' + ''.join(treat_shares_s)
        txt += f'  {summ:12.2f}  {mean:8.4f}  {variance:8.4f}'
    txt += '\n' + '-' * 100
    ps.print_mcf(optp_.gen_dict, txt, summary=True)


def variable_importance(optp_, data_df, allocation_df, seed=1234567):
    """Compute variable importance measures for various allocations."""
    names_unordered = optp_.var_dict['vi_to_dummy_name']
    for alloc_name in allocation_df.columns:
        if alloc_name in ('random', 'RANDOM'):
            continue
        txt = ('\n' * 2 + '-' * 100 + '\n'
               + f'Variable importance statistic for {alloc_name}\n'
               + '- ' * 50)
        ps.print_mcf(optp_.gen_dict, txt, summary=True)
        if (optp_.var_dict['vi_to_dummy_name'] is None
                or optp_.var_dict['vi_to_dummy_name'] == []):
            x_name = optp_.var_dict['vi_x_name']
        elif (optp_.var_dict['vi_x_name'] is None
                or optp_.var_dict['vi_x_name'] == []):
            x_name = optp_.var_dict['vi_to_dummy_name']
        else:
            x_name = (optp_.var_dict['vi_to_dummy_name']
                      + optp_.var_dict['vi_x_name'])
        if x_name is None or len(x_name) == 1:
            continue
        x_df = data_df[x_name].copy()
        d_df = allocation_df[alloc_name]
        dummy_names = {}
        if names_unordered:
            for idx, name in enumerate(names_unordered):
                x_dummies = pd.get_dummies(x_df[name], columns=[name],
                                           dtype=int)
                x_dummies_names = [name + str(dummy)
                                   for dummy in x_dummies.columns]
                dummy_names[name] = x_dummies.columns = x_dummies_names
                x_dummies_s = x_dummies.reindex(sorted(x_dummies.columns),
                                                axis=1)
                if idx == 0:
                    x_all_dummies = x_dummies_s
                else:
                    x_all_dummies = pd.concat([x_all_dummies, x_dummies_s],
                                              axis=1, copy=True)
            names_wo = [name for name in x_df.columns
                        if name not in names_unordered]
            x_new_df = pd.concat((x_df[names_wo], x_all_dummies), axis=1,
                                 copy=True)
        else:
            x_new_df = x_df
        classif = RandomForestClassifier(
            n_estimators=1000, max_features='sqrt', bootstrap=True,
            oob_score=False, n_jobs=optp_.int_dict['mp_parallel'],
            random_state=seed, verbose=False, min_samples_split=5)
#        classif.fit(x_train, d_train)
        vi.print_variable_importance(
            classif, x_new_df, d_df, x_name, names_unordered, dummy_names,
            optp_.gen_dict, summary=True)


def value_to_index_dic_fct(values_np):
    """Array of indices of values in numpy array."""
    vals = np.int16(np.round(np.unique(values_np)))
    indices = np.int16(np.arange(len(vals)))
    return vals, indices


def get_random_allocation(optp_, no_obs, seed):
    """Obtain random allocation."""
    rng = np.random.default_rng(seed)
    allocation_np = rng.choice(optp_.gen_dict['d_values'], size=no_obs,
                               p=optp_.rnd_dict['shares'])
    return allocation_np
