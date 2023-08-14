
"""
Created on Tue Jul 18 07:58:29 2023.

Provide functions for the evaluation of allocations.

# -*- coding: utf-8 -*-

@author: MLechner
"""
from copy import deepcopy

import numpy as np
import pandas as pd

from mcf import mcf_print_stats_functions as ps


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
        txt += ('\n' + '=' * 100 + '\n' + 'Mean of variables /'
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
    values_, treat_counts = np.unique(allocation, return_counts=True)
    treat_shares_ = treat_counts / len(allocation)
    treat_shares, jdx = [0] * len(optp_.gen_dict['d_values']), 0
    if list(values_):
        for idx, vals in enumerate(optp_.gen_dict['d_values']):
            if values_[jdx] == vals:
                treat_shares[idx] = treat_shares_[jdx]
                jdx += 1
    treat_shares_s = [f'{s:>8.2%}' for s in treat_shares]
    txt += ' ' * 6 + ''.join(treat_shares_s)
    if optp_.gen_dict['with_output']:
        ps.print_mcf(optp_.gen_dict, txt, summary=True)
    results_new_dic[alloc_name] = local_dic.copy()
    return results_new_dic


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
