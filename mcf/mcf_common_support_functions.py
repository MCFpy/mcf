"""
Contains functions for common support adjustments.

Created on Thu May 11 16:30:11 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mcf import mcf_data_functions as mcf_data
from mcf import mcf_general as gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as ps
from mcf import mcf_variable_importance_functions as vi


def common_support(mcf_, tree_df, fill_y_df, train=True):
    """Remove observations from data files that are off-support."""
    gen_dic = mcf_.gen_dict
    lc_dic, var_x_type, cs_dic = mcf_.lc_dict, mcf_.var_x_type, mcf_.cs_dict
    data_train_dic = mcf_.data_train_dict
    d_name, _, no_of_treat = mcf_data.get_treat_info(mcf_)
    x_name, x_type = gp.get_key_values_in_list(var_x_type)
    names_unordered = [x_name[j] for j, val in enumerate(x_type) if val > 0]
    ps.print_mcf(gen_dic, '\n' + '=' * 100 + '\nCommon support analysis',
                 summary=True)
    if train:
        if lc_dic['cs_cv']:   # Crossvalidate ... only tree data is used
            tree_mcf_df, fill_y_mcf_df = tree_df.copy(), fill_y_df.copy()
            d_tree_mcf_np = tree_mcf_df[d_name].to_numpy().ravel()
        else:  # Use lc_dic['cs_share'] of data for common support estim. only
            # Take the same share of obs. from both input samples
            tree_mcf_df, tree_cs_df = train_test_split(
                tree_df, test_size=lc_dic['cs_share'], random_state=42)
            fill_y_mcf_df, fill_y_cs_df = train_test_split(
                fill_y_df, test_size=lc_dic['cs_share'], random_state=42)
            data_cs_df = pd.concat([tree_cs_df, fill_y_cs_df], axis=0)
            x_cs_df, _ = mcf_data.get_x_data(data_cs_df, x_name)
            d_cs_np = data_cs_df[d_name].to_numpy().ravel()
        x_fy_df, _ = mcf_data.get_x_data(fill_y_mcf_df, x_name)
    else:
        tree_mcf_df, fill_y_mcf_df = tree_df, None
    x_mcf_df, obs_mcf = mcf_data.get_x_data(tree_mcf_df, x_name)  # train,adj.
    txt = ''
    if train:
        if names_unordered:  # List is not empty
            x_fy_df, _ = mcf_data.dummies_for_unord(
                x_fy_df, names_unordered, data_train_dict=data_train_dic)
            x_mcf_df, dummy_names = mcf_data.dummies_for_unord(
                x_mcf_df, names_unordered, data_train_dict=data_train_dic)
            if not lc_dic['cs_cv']:
                x_cs_df, _ = mcf_data.dummies_for_unord(
                    x_cs_df, names_unordered, data_train_dict=data_train_dic)
        else:
            dummy_names = None
        if not lc_dic['cs_cv']:
            x_cs_np = x_cs_df.to_numpy()
        x_fy_np, x_mcf_np = x_fy_df.to_numpy(), x_mcf_df.to_numpy()
        if gen_dic['with_output'] and gen_dic['verbose']:
            txt += '\n' + '-' * 100 + '\n'
            txt += 'Computing random forest based common support\n'
            ps.print_mcf(gen_dic, txt, summary=False)
        max_workers = 1 if gen_dic['replication'] else gen_dic['mp_parallel']
        classif = RandomForestClassifier(
            n_estimators=mcf_.cf_dict['boot'], max_features='sqrt',
            bootstrap=True, oob_score=False, n_jobs=max_workers,
            random_state=42, verbose=False, min_samples_split=5)
        if lc_dic['cs_cv']:   # Crossvalidate
            index = np.arange(obs_mcf)       # indices
            rng = np.random.default_rng(seed=9324561)
            rng.shuffle(index)
            index_folds = np.array_split(index, lc_dic['cs_cv_k'])
            pred_mcf_np = np.empty((len(index), no_of_treat))
            pred_fy_np_fold = np.zeros((len(x_fy_np), no_of_treat))
            forests = []
            for fold_pred in range(lc_dic['cs_cv_k']):
                fold_train = [x for idx, x in enumerate(index_folds)
                              if idx != fold_pred]
                index_train = np.hstack(fold_train)
                index_pred = index_folds[fold_pred]
                x_pred, x_train = x_mcf_np[index_pred], x_mcf_np[index_train]
                d_train = d_tree_mcf_np[index_train]
                classif.fit(x_train, d_train)
                forests.append(deepcopy(classif))
                pred_mcf_np[index_pred, :] = classif.predict_proba(x_pred)
                pred_fy_np_fold += classif.predict_proba(x_fy_np)
            pred_cs_np, d_cs_np = pred_mcf_np, d_tree_mcf_np  # To get cut-offs
            pred_fy_np = pred_fy_np_fold / lc_dic['cs_cv_k']
        else:
            x_train, x_test, d_train, d_test = train_test_split(
                x_cs_np, d_cs_np, test_size=0.25, random_state=42)
            classif.fit(x_train, d_train)
            pred_cs_np = classif.predict_proba(x_test)  # -> determine cut-offs
            d_cs_np = d_test
            pred_mcf_np = classif.predict_proba(x_mcf_np)   # cut and return
            pred_fy_np = classif.predict_proba(x_fy_np)     # cut and return
            forests = [classif]
        cs_dic['forests'] = forests
        if gen_dic['with_output']:
            vi.print_variable_importance(
                deepcopy(classif), x_mcf_df, tree_mcf_df[d_name], x_name,
                names_unordered, dummy_names, gen_dic, summary=False)
        # Normalize estimated probabilities to add up to 1
        pred_cs_np_sum = pred_cs_np.sum(axis=1, keepdims=True)
        pred_mcf_np_sum = pred_mcf_np.sum(axis=1, keepdims=True)
        pred_fy_np_sum = pred_fy_np.sum(axis=1, keepdims=True)
        pred_cs_np /= pred_cs_np_sum
        pred_mcf_np /= pred_mcf_np_sum
        pred_fy_np /= pred_fy_np_sum
        # Determine cut-offs nased on pred_cs_np
        cs_dic['cut_offs'] = get_cut_off_probs(mcf_, pred_cs_np, d_cs_np)
        mcf_.cs_dict = cs_dic   # Update instance with cut-off prob's
        # Descriptive stats
        if gen_dic['with_output']:
            plot_support(mcf_, pred_cs_np, d_cs_np)
            descriptive_stats_on_off_support(mcf_, pred_fy_np, fill_y_mcf_df,
                                             'Training - fill mcf with y data')
        # Reduce samples
        fill_y_mcf_df, _ = on_off_support_df(mcf_, pred_fy_np, fill_y_mcf_df)
    else:  # Reduce prediction sample
        # Predict treatment probabilities
        if names_unordered:  # List is not empty
            x_mcf_df, _ = mcf_data.dummies_for_unord(
                x_mcf_df, names_unordered, data_train_dict=data_train_dic)
        pred_mcf_np = np.zeros((len(x_mcf_df), no_of_treat))
        # If cross-validation, take average of forests in folds
        for forest in cs_dic['forests']:
            pred_mcf_np += forest.predict_proba(x_mcf_df.to_numpy())
        pred_mcf_np /= len(cs_dic['forests'])
    # Normalize estimated probabilities to add up to 1
    pred_mcf_np /= pred_mcf_np.sum(axis=1, keepdims=True)
    # Delete observation off support
    if gen_dic['with_output']:
        titel = 'Training - build mcf data' if train else 'Prediction data'
        descriptive_stats_on_off_support(mcf_, pred_mcf_np, tree_mcf_df, titel)
    tree_mcf_df, _ = on_off_support_df(mcf_, pred_mcf_np, tree_mcf_df)
    return tree_mcf_df, fill_y_mcf_df


def check_if_too_many_deleted(mcf_, obs_keep, obs_del):
    """Check if too many obs are deleted and raise Exception if so."""
    max_del_train = mcf_.cs_dict['max_del_train']
    share_del = obs_del / (obs_keep + obs_del)
    if share_del > max_del_train:
        err_str = (
            f'{share_del:3.1%} observation deleted in common support, but only'
            f' {max_del_train:3.1%} observations of training data are allowed'
            ' to be deleted in support check. Programme is terminated. Improve'
            ' balance of input data or change share allowed to be deleted.')
        raise ValueError(err_str)


def descriptive_stats_on_off_support(mcf_, probs_np, data_df, titel=''):
    """Compute descriptive stats for deleted and retained observations."""
    keep_df, delete_df = on_off_support_df(mcf_, probs_np, data_df)
    gen_dic, var_dic = mcf_.gen_dict, mcf_.var_dict
    if delete_df.empty:
        txt = (f'\n\nData investigated for common support: {titel}\n'
               + '-' * 100)
        ps.print_mcf(gen_dic, '\nNo observations deleted in common support '
                     'check', summary=True)
    else:
        d_name, _, _ = mcf_data.get_treat_info(mcf_)
        x_name = var_dic['x_name']
        obs_del, obs_keep = len(delete_df), len(keep_df)
        obs = obs_del + obs_keep
        txt = '\n' + '-' * 100
        txt += f'\nData investigated for common support: {titel}\n' + '-' * 100
        txt += f'\nObservations deleted: {obs_del:4} ({obs_del/obs:.2%})'
        txt += '\n' + '-' * 100
        ps.print_mcf(gen_dic, txt, summary=True)
        txt = ''
        with pd.option_context(
                'display.max_rows', 500, 'display.max_columns', 500,
                'display.expand_frame_repr', True, 'display.width', 150,
                'chop_threshold', 1e-13):
            all_var_names = [name.upper() for name in data_df.columns]
            if d_name[0].upper() in all_var_names:
                d_keep = keep_df[d_name]
                d_delete = delete_df[d_name]
                d_keep_count = d_keep.value_counts(sort=False)
                d_delete_count = d_delete.value_counts(sort=False)
                d_keep_count = pd.concat(
                    [d_keep_count, np.round(d_keep_count / obs_keep * 100, 2)],
                    axis=1)
                d_delete_count = pd.concat(
                    [d_delete_count,
                     np.round(d_delete_count / obs_del * 100, 2)], axis=1)
                d_keep_count.columns = ['Obs.', 'Share in %']
                d_delete_count.columns = ['Obs.', 'Share in %']
                if gen_dic['panel_data']:
                    cluster_id = data_df[var_dic['cluster_name']].squeeze()
                    cluster_keep = keep_df[var_dic['cluster_name']].squeeze()
                    cluster_delete = delete_df[var_dic['cluster_name']
                                               ].squeeze()
                k_str = '\nObservations kept, by treatment\n    '
                d_str = '\nObservations deleted, by treatment\n '
                k_str += d_keep_count.to_string()
                d_str += d_delete_count.to_string()
                txt += k_str + '\n' + '-   ' * 20 + d_str
                if gen_dic['panel_data']:
                    txt += '-   ' * 20
                    txt += '\nTotal number of panel units:'
                    txt += f'{len(cluster_id.unique())}'
                    txt += '\nObservations belonging to '
                    txt += f'{len(cluster_keep.unique())} panel units that are'
                    txt += '  ON support\nObservations belonging to '
                    txt += f'{len(cluster_delete.unique())} panel units are'
                    txt += ' OFF support'
                ps.print_mcf(gen_dic, txt, summary=True)
            else:
                txt = f'\nData investigated for common support: {titel}\n'
                txt += '-' * 100 + '\nTreatment not in prediction data.\n'
                txt += '-' * 100
                ps.print_mcf(gen_dic, txt, summary=False)
            txt = '\n' + '-' * 100
            txt += '\nFull sample (Data ON and OFF support)' + '\n' + '-' * 100
            ps.print_mcf(gen_dic, txt, summary=False)
            ps.print_mcf(gen_dic, data_df[x_name].describe().transpose(),
                         summary=False)
            if d_name[0].upper() in all_var_names:
                mean_by_treatment(data_df[d_name], data_df[x_name], gen_dic,
                                  summary=False)
            txt = '\n' + '-' * 100 + '\nData ON support' + '\n' + '-' * 100
            ps.print_mcf(gen_dic, txt, summary=False)
            ps.print_mcf(gen_dic, keep_df[x_name].describe().transpose(),
                         summary=False)
            if d_name[0].upper() in all_var_names:
                mean_by_treatment(keep_df[d_name], keep_df[x_name], gen_dic,
                                  summary=False)
            txt = '\n' + '-' * 100 + '\nData OFF support' + '\n' + '-' * 100
            ps.print_mcf(gen_dic, txt, summary=False)
            ps.print_mcf(gen_dic, delete_df[x_name].describe().transpose(),
                         summary=False)
            if d_name[0].upper() in all_var_names:
                mean_by_treatment(delete_df[d_name], delete_df[x_name],
                                  gen_dic, summary=False)
        check_if_too_many_deleted(mcf_, obs_keep, obs_del)


def mean_by_treatment(treat_df, data_df, gen_dic, summary=False):
    """Compute mean by treatment status."""
    treat_df = treat_df.squeeze()
    treat_vals = pd.unique(treat_df)
    txt = '\n------------------ Mean by treatment status ---------------------'
    ps.print_mcf(gen_dic, txt, summary=summary)
    if len(treat_vals) > 0:
        mean = data_df.groupby(treat_df).mean(numeric_only=True)
        ps.print_mcf(gen_dic, mean.transpose(), summary=summary)
    else:
        txt = f'\nAll obs have same treatment: {treat_vals}'
        ps.print_mcf(gen_dic, txt, summary=summary)


def on_off_support_df(mcf_, probs_np, data_df):
    """Split DataFrame into retained and deleted part."""
    cs_dic = mcf_.cs_dict
    _, _, no_of_treat = mcf_data.get_treat_info(mcf_)
    lower, upper = cs_dic['cut_offs']['lower'], cs_dic['cut_offs']['upper']
    obs = len(probs_np)
    off_support = np.empty(obs, dtype=bool)
    # off_upper = np.empty(no_of_treat, dtype=bool)
    # off_lower = np.empty_like(off_upper)
    for i in range(obs):
        off_upper = np.any(probs_np[i, :] > upper)
        off_lower = np.any(probs_np[i, :] < lower)
        off_support[i] = off_upper or off_lower
    data_on_df = data_df[~off_support].copy()
    data_off_df = data_df[off_support].copy()
    return data_on_df, data_off_df


def plot_support(mcf_, probs_np, d_np):
    """Histogrammes for distribution of treatment probabilities for overlap."""
    cs_dic, int_dic = mcf_.cs_dict, mcf_.int_dict
    lower, upper = cs_dic['cut_offs']['lower'], cs_dic['cut_offs']['upper']
    _, d_values, _ = mcf_data.get_treat_info(mcf_)
    color_list = ['red', 'blue', 'green', 'violet', 'magenta', 'crimson',
                  'yellow', 'darkorange', 'khaki', 'skyblue', 'darkgreen',
                  'olive', 'greenyellow',  'aguamarine', 'deeppink',
                  'royalblue', 'navy', 'blueviolet', 'purple']
    if len(color_list) < len(d_values):
        color_list = color_list * len(d_values)
    color_list = color_list[:len(d_values)]
    for idx_p, ival_p in enumerate(d_values):  # iterate treatment probs
        treat_prob = probs_np[:, idx_p]
        titel = f'Probability of treatment {ival_p} in different subsamples'
        f_titel = f'common_support_pr_treat{ival_p}'
        file_name_jpeg = (cs_dic['common_support_fig_pfad_jpeg']
                          + '/' + f_titel + '.jpeg')
        file_name_pdf = (cs_dic['common_support_fig_pfad_pdf']
                         + '/' + f_titel + '.pdf')
        file_name_jpeg_d = (cs_dic['common_support_fig_pfad_jpeg']
                            + '/' + f_titel + '_d.jpeg')
        file_name_pdf_d = (cs_dic['common_support_fig_pfad_pdf']
                           + '/' + f_titel + '_d.pdf')
        data_hist = [treat_prob[d_np == val] for val in d_values]
        fig, axs = plt.subplots()
        fig_d, axs_d = plt.subplots()
        labels = ['Treat ' + str(d) for d in d_values]
        for idx, dat in enumerate(data_hist):
            axs.hist(dat, bins='auto', histtype='bar', label=labels[idx],
                     color=color_list[idx], alpha=0.5, density=False)
            _, bins, _ = axs_d.hist(dat, bins='auto', histtype='bar',
                                    label=labels[idx], color=color_list[idx],
                                    alpha=0.5, density=True)
            sigma = np.std(dat)
            fit_line = ((1 / (np.sqrt(2 * np.pi) * sigma))
                        * np.exp(-0.5 * (1 / sigma
                                         * (bins - np.mean(dat)))**2))
            axs_d.plot(bins, fit_line, '--', color=color_list[idx],
                       label='Smoothed ' + labels[idx])
        axs.set_title(titel)
        axs.set_xlabel('Treatment probability')
        axs.set_ylabel('Observations')
        axs.set_xlim([0, 1])
        axs.axvline(lower[idx_p], color='blue', linewidth=0.7,
                    linestyle="--", label='min')
        axs.axvline(upper[idx_p], color='black', linewidth=0.7,
                    linestyle="--", label='max')
        axs.legend(loc=int_dic['legend_loc'], shadow=True,
                   fontsize=int_dic['fontsize'])
        mcf_sys.delete_file_if_exists(file_name_jpeg)
        mcf_sys.delete_file_if_exists(file_name_pdf)
        fig.savefig(file_name_jpeg, dpi=int_dic['dpi'])
        fig.savefig(file_name_pdf, dpi=int_dic['dpi'])
        axs_d.set_title(titel)
        axs_d.set_xlabel('Treatment probability')
        axs_d.set_ylabel('Density')
        axs_d.set_xlim([0, 1])
        axs_d.axvline(lower[idx_p], color='blue', linewidth=0.7,
                      linestyle="--", label='min')
        axs_d.axvline(upper[idx_p], color='black', linewidth=0.7,
                      linestyle="--", label='max')
        axs_d.legend(loc=int_dic['legend_loc'], shadow=True,
                     fontsize=int_dic['fontsize'])
        mcf_sys.delete_file_if_exists(file_name_jpeg_d)
        mcf_sys.delete_file_if_exists(file_name_pdf_d)
        fig_d.savefig(file_name_jpeg_d, dpi=int_dic['dpi'])
        fig_d.savefig(file_name_pdf_d, dpi=int_dic['dpi'])
        if int_dic['show_plots']:
            plt.show()
        else:
            plt.close()


def get_cut_off_probs(mcf_, probs_np, d_np):
    """Compute the cut-offs for common support for training only."""
    cs_dic, gen_dic = mcf_.cs_dict, mcf_.gen_dict
    _, d_values, no_of_treat = mcf_data.get_treat_info(mcf_)
    if cs_dic['type'] == 1:
        q_s = cs_dic['quantil']
        upper_limit = np.empty((no_of_treat, no_of_treat))
        lower_limit = np.empty_like(upper_limit)
        for idx, ival in enumerate(d_values):
            probs = probs_np[d_np == ival]
            if q_s == 1:
                upper_limit[idx, :] = np.max(probs, axis=0)
                lower_limit[idx, :] = np.min(probs, axis=0)
            else:
                upper_limit[idx, :] = np.quantile(probs, q_s, axis=0)
                lower_limit[idx, :] = np.quantile(probs, 1 - q_s, axis=0)
            upper_limit[idx, 0] = 1
            lower_limit[idx, 0] = 0
        txt = ''
        if cs_dic['adjust_limits'] != 0:
            upper_limit *= 1 + cs_dic['adjust_limits']
            lower_limit *= 1 - cs_dic['adjust_limits']
            lower_limit = np.clip(lower_limit, a_min=0, a_max=1)
            upper_limit = np.clip(upper_limit, a_min=0, a_max=1)
            if gen_dic['with_output']:
                txt += '\n' + '-' * 100 + '\nCommon support bounds adjusted by'
                txt += f' {cs_dic["adjust_limits"]:5.2%}-points\n' + '-' * 100
        if gen_dic['with_output']:
            txt += '\nTreatment sample     Treatment probabilities in %'
            txt += '\n--------------------- Upper limits ----------------'
            for idx, ival in enumerate(d_values):
                txt += f'\nD = {ival:9}          '
                for jdx in range(no_of_treat):
                    txt += f'{upper_limit[idx, jdx]:>7.2%}  '
            txt += '\n--------------------- Lower limits ----------------'
            for idx, ival in enumerate(d_values):
                txt += f'\nD = {ival:9}          '
                for jdx in range(no_of_treat):
                    txt += f'{lower_limit[idx, jdx]:>7.2%}  '
            txt += '\n' + 100 * '-'
            txt += '\nFirst treatment is set to 1 and 0 (ignored) due to'
            txt += ' additivity.' + '\n' + 100 * '-'
        upper, lower = np.min(upper_limit, axis=0), np.max(lower_limit, axis=0)
        if gen_dic['with_output']:
            upper_str = [f'{x:>7.2%}' for x in upper]
            lower_str = [f'{x:>7.2%}' for x in lower]
            txt += '\nUpper limits used: ' + ' '.join(upper_str)
            txt += '\nLower limits used: ' + ' '.join(lower_str)
            txt += '\n' + 100 * '-'
            ps.print_mcf(gen_dic, txt, summary=True)
    else:
        # Normalize such that probabilities add up to 1
        upper = np.ones(no_of_treat) * (1 - cs_dic['min_p'])
        lower = np.ones(no_of_treat) * cs_dic['min_p']
    cut_offs = {'upper': upper, 'lower': lower}
    return cut_offs
