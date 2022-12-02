"""
Procedures needed for Common support estimation.

Created on Thu Dec  8 15:48:57 2020.

@author: MLechner

# -*- coding: utf-8 -*-
"""
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mcf import mcf_data_functions as mcf_data
from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est


def common_support(predict_file, tree_file, fill_y_file, fs_file, var_x_type,
                   v_dict, c_dict, cs_list=None, prime_values_dict=None,
                   pred_tr_np=None, d_tr_np=None):
    """
    Remove observations from data files that are off-support.

    Parameters
    ----------
    predict_file : String of csv-file. Data to predict the RF.
    train_file : String of csv-file. Data to train the RF.
    fill_y_file : String of csv-file. Data with y to be used by RF.
    fs_file : String of csv-file. Data with y to be used by RF.
    var_x_type : Dict. Features.
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.
    cs_list: Tuple. Contains the information from estimated propensity score
                    needed to predict for other data. Default is None.
    prime_values_dict: Dict. List of unique values for variables to dummy.
                    Default is None.
    pred_t: Numpy array. Predicted treatment probabilities in training data.
                         Needed to define cut-offs.
    d_train: Numpy series. Observed treatment in training data (tree_file).

    Returns
    -------
    predict_file_new : String of csv-file. Adjusted data.
    cs_list: Tuple. Contains the information from estimated propensity score
                    needed to predict for other data.
    pred_t: Numpy array. Predicted treatment probabilities in training data.
    d_train_tree: estimated tree by sklearn.

    """
    def r2_obb(c_dict, idx, oob_best):
        d_values = (c_dict['ct_grid_nn_val']
                    if c_dict['d_type'] == 'continuous'
                    else c_dict['d_values'])
        if c_dict['with_output']:
            print('\n')
            print('-' * 80)
            print(f'Treatment: {d_values[idx]:2}',
                  f'OOB Score (R2 in %): {oob_best * 100:6.3f}')
            print('-' * 80)

    def get_data(file_name, x_name):
        data = pd.read_csv(file_name)
        x_all = data[x_name]    # deep copies
        obs = len(x_all.index)
        return data, x_all, obs

    def check_cols(x_1, x_2, name1, name2):
        var1 = set(x_1.columns)
        var2 = set(x_2.columns)
        if var1 != var2:
            if len(var1-var2) > 0:
                print('Variables in ', name1, 'not contained in ', name2,
                      *(var1-var2))
            if len(var2-var1) > 0:
                print('Variables in ', name2, 'not contained in ', name1,
                      *(var2-var1))
            raise Exception(name1 + ' data and ' + name2 + ' data contain' +
                            ' differnt variables. Programm stopped.')

    def mean_by_treatment(treat_pd, data_pd):
        treat_pd = treat_pd.squeeze()
        treat_vals = pd.unique(treat_pd)
        print('--------------- Mean by treatment status ------------------')
        if len(treat_vals) > 0:
            mean = data_pd.groupby(treat_pd).mean(numeric_only=True)
            print(mean.transpose())
        else:
            print('All obs have same treatment:', treat_vals)

    def on_support_data_and_stats(obs_to_del_np, data_pd, x_data_pd, out_file,
                                  upper_l, lower_l, c_dict, header=False,
                                  d_name=None):
        obs_to_keep = np.invert(obs_to_del_np)
        data_keep = data_pd[obs_to_keep]
        gp.delete_file_if_exists(out_file)
        data_keep.to_csv(out_file, index=False)
        if c_dict['with_output']:
            x_keep, x_delete = x_data_pd[obs_to_keep], x_data_pd[obs_to_del_np]
            if header:
                u_str, l_str = '', ''
                for upper, lower in zip(upper_l, lower_l):
                    u_str += f'{upper:10.5%} '
                    l_str += f'{lower:10.5%} '
                print_str = (
                    '\n' * 2 + '=' * 80 + '\nCommon support check' + '\n' +
                    '-' * 80 + '\n'
                    + f'Upper limits on treatment probabilities: {u_str}'
                    + '\n'
                    + f'Lower limits on treatment probabilities: {l_str}')
                print(print_str)
                gp.print_f(c_dict['outfilesummary'], print_str)
            print_str = ('-' * 80 + '\nData investigated and saved:'
                         + f'{out_file}' + '\n' + '-' * 80 + '\n'
                         + f'Observations deleted: {np.sum(obs_to_del_np):4}'
                         + f' ({np.mean(obs_to_del_np):.3%})'
                         + '\n' + '-' * 80)
            print(print_str)
            gp.print_f(c_dict['outfilesummary'], print_str)
            with pd.option_context(
                    'display.max_rows', 500, 'display.max_columns', 500,
                    'display.expand_frame_repr', True, 'display.width', 150,
                    'chop_threshold', 1e-13):
                all_var_names = [name.upper() for name in data_pd.columns]
                if d_name[0].upper() in all_var_names:
                    d_keep = data_keep[d_name]
                    d_delete = data_pd[d_name]
                    d_delete = d_delete[obs_to_del_np]
                    d_keep_count = d_keep.value_counts(sort=False)
                    d_delete_count = d_delete.value_counts(sort=False)
                    d_keep_count = pd.concat(
                        [d_keep_count,
                         d_keep_count / np.sum(obs_to_keep) * 100], axis=1)
                    d_delete_count = pd.concat(
                        [d_delete_count,
                         d_delete_count / np.sum(obs_to_del_np) * 100], axis=1)
                    d_keep_count.columns = ['Obs.', 'Share in %']
                    d_delete_count.columns = ['Obs.', 'Share in %']
                    if c_dict['panel_data']:
                        cluster_id = data_pd[v_dict['cluster_name']].squeeze()
                        cluster_keep = cluster_id[obs_to_keep].squeeze()
                        cluster_delete = cluster_id[obs_to_del_np].squeeze()
                    k_str = 'Observations kept by treatment\n    '
                    d_str = '\nObservations deleted by treatment\n '
                    k_str += d_keep_count.to_string()
                    d_str += d_delete_count.to_string()
                    print_str = k_str + '\n' + '-   ' * 20 + d_str
                    print(print_str)
                    gp.print_f(c_dict['outfilesummary'], print_str)
                    if c_dict['panel_data']:
                        print('-   ' * 20)
                        print('Total number of panel unit:',
                              len(cluster_id.unique()))
                        print('Observations belonging to ',
                              len(cluster_keep.unique()),
                              'panel units are ON support')
                        print('Observations belonging to ',
                              len(cluster_delete.unique()),
                              'panel units are OFF support')
                else:
                    gp.print_f(c_dict['outfilesummary'],
                               'Treatment not in prediction file.\n'
                               + '-' * 80)
                if d_name[0].upper() in all_var_names:
                    print('\nFull sample (ON and OFF support observations)')
                    mean_by_treatment(data_pd[d_name], x_data_pd)
                print('-' * 80)
                print('Data ON support')
                print('-' * 80)
                print(x_keep.describe().transpose())
                if d_name[0].upper() in all_var_names:
                    print()
                    mean_by_treatment(d_keep, x_keep)
                print('-' * 80)
                print('Data OFF support')
                print('-' * 80)
                print(x_delete.describe().transpose())
                if d_name[0].upper() in all_var_names:
                    print()
                    if np.sum(obs_to_del_np) > 1:
                        mean_by_treatment(d_delete, x_delete)
                    else:
                        print('Only single observation deleted.')
            assert np.mean(obs_to_del_np) <= c_dict['support_max_del_train'], (
                'Less than {100-c_dict["support_max_del_train"]*100):3}% obs.'
                + ' left after common support check of training data.'
                + ' Programme terminated. Improve balance of input data before'
                + ' forest building.')
    if c_dict['d_type'] == 'continuous':
        d_name = v_dict['d_grid_nn_name']
        no_of_treat = len(c_dict['ct_grid_nn_val'])
    else:
        d_name, no_of_treat = v_dict['d_name'], c_dict['no_of_treat']
    x_name, x_type = gp.get_key_values_in_list(var_x_type)
    names_unordered = [x_name[j] for j, val in enumerate(x_type) if val > 0]
    fs_adjust, obs_fs = False, 0
    if c_dict['train_mcf']:
        data_tr, x_tr, obs_tr = get_data(tree_file, x_name)  # train,adj.
        data_fy, x_fy, obs_fy = get_data(fill_y_file, x_name)  # adj.
        if c_dict['fs_yes']:
            if fs_file not in (tree_file, fill_y_file):
                data_fs, x_fs, obs_fs = get_data(fs_file, x_name)  # adj.
                fs_adjust = True
    if c_dict['pred_mcf']:
        data_pr, x_pr, obs_pr = get_data(predict_file, x_name)
    else:
        obs_pr = 0
    if names_unordered:  # List is not empty
        if c_dict['train_mcf'] and c_dict['pred_mcf']:
            x_total = pd.concat([x_tr, x_fy, x_pr], axis=0)
            if fs_adjust:
                x_total = pd.concat([x_total, x_fs], axis=0)
            x_dummies = pd.get_dummies(x_total, columns=names_unordered)
            x_total = pd.concat([x_total[names_unordered],
                                 x_dummies], axis=1)
            x_tr, x_fy = x_total[:obs_tr], x_total[obs_tr:obs_tr+obs_fy]
            x_pr = x_total[obs_tr+obs_fy:obs_tr+obs_fy+obs_pr]
            if fs_adjust:
                x_fs = x_total[obs_tr+obs_fy+obs_pr:]
        elif c_dict['train_mcf'] and not c_dict['pred_mcf']:
            x_total = pd.concat([x_tr, x_fy], axis=0)
            if fs_adjust:
                x_total = pd.concat([x_total, x_fs], axis=0)
            x_dummies = pd.get_dummies(x_total, columns=names_unordered)
            x_total = pd.concat([x_total[names_unordered],
                                 x_dummies], axis=1)
            x_tr, x_fy = x_total[:obs_tr], x_total[obs_tr:obs_tr+obs_fy]
            if fs_adjust:
                x_fs = x_total[obs_tr+obs_fy:]
        else:
            x_add_tmp = check_if_obs_needed(names_unordered, x_pr,
                                            prime_values_dict)
            x_total = (pd.concat([x_pr, x_add_tmp], axis=0)
                       if x_add_tmp is not None else x_pr)
            x_dummies = pd.get_dummies(x_total, columns=names_unordered)
            x_pr = pd.concat([x_total[names_unordered], x_dummies], axis=1)
            if x_add_tmp is not None:  # remove add_temp
                x_pr = x_pr[:obs_pr]
    x_name_all = (x_tr.columns.values.tolist()
                  if c_dict['train_mcf'] else x_pr.columns.values.tolist())
    if c_dict['train_mcf']:
        x_tr_np = x_tr.to_numpy(copy=True)
        d_all_in = pd.get_dummies(data_tr[d_name], columns=d_name)
        d_tr_np = d_all_in.to_numpy(copy=True)
        pred_tr_np = np.empty((np.shape(x_tr_np)[0], no_of_treat))
    if c_dict['train_mcf']:
        x_pred_all = x_fy.copy()
        if c_dict['pred_mcf']:
            x_pred_all = pd.concat([x_pred_all, x_pr], axis=0)
        if fs_adjust:
            x_pred_all = pd.concat([x_pred_all, x_fs], axis=0)
    else:
        obs_fy = 0
        x_pred_all = x_pr.copy()
    x_pred_all_np = x_pred_all.to_numpy(copy=True)
    pred_all_np = np.empty((obs_fy+obs_fs+obs_pr, no_of_treat))
    workers_mp = (copy.copy(c_dict['no_parallel'])
                  if c_dict['no_parallel'] > 1 else None)
    if c_dict['train_mcf']:
        check_cols(x_tr, x_fy, 'Tree', 'Fill_y')
        if fs_adjust:
            check_cols(x_tr, x_fs, 'Tree', 'Feature selection')
    if c_dict['train_mcf'] and c_dict['pred_mcf']:
        check_cols(x_tr, x_pr, 'Tree', 'Prediction')
    if c_dict['train_mcf']:
        if c_dict['with_output'] and c_dict['verbose']:
            print('\n')
            print('-' * 80)
            print('Computing random forest based common support')
    if c_dict['train_mcf']:
        cs_list = []
        c_dict_new = mcf_data.m_n_grid(copy.deepcopy(c_dict),    # dict only
                                       len(x_pred_all.columns))  # used here
        for idx in range(no_of_treat):
            return_forest = bool(c_dict['save_forest'])
            ret_rf = gp_est.random_forest_scikit(
                x_tr_np, d_tr_np[:, idx], x_pred_all_np, boot=c_dict['boot'],
                n_min=c_dict_new['grid_n_min']/2,
                no_features='sqrt', workers=workers_mp,
                pred_p_flag=True, pred_t_flag=True, pred_oob_flag=True,
                with_output=False, variable_importance=c_dict['verbose'],
                x_name=x_name_all, var_im_with_output=c_dict['with_output'],
                return_forest_object=return_forest)
            pred_all_np[:, idx] = np.copy(ret_rf[0])
            pred_tr_np[:, idx] = np.copy(ret_rf[1])
            oob_best = np.copy(ret_rf[2])
            if c_dict['save_forest']:
                cs_list.append(ret_rf[6])
            r2_obb(c_dict, idx, oob_best)
            if no_of_treat == 2:
                pred_all_np[:, idx+1] = 1 - pred_all_np[:, idx]
                pred_tr_np[:, idx+1] = 1 - pred_tr_np[:, idx]
                break
    else:
        for idx in range(no_of_treat):
            pred_all_np[:, idx] = cs_list[idx].predict(x_pred_all_np)
            if no_of_treat == 2:
                pred_all_np[:, idx+1] = 1 - pred_all_np[:, idx]
                break
    obs_to_del_all, obs_to_del_tr, upper_l, lower_l = off_support_and_plot(
        pred_tr_np, pred_all_np, d_tr_np, c_dict)
    # split obs_to_del_all into its parts
    obs_to_del_fs, obs_to_del_fy, obs_to_del_pr = False, False, False
    predict_file_new = None
    if c_dict['train_mcf']:
        obs_to_del_fy = obs_to_del_all[:obs_fy]
        if c_dict['pred_mcf']:
            obs_to_del_pr = obs_to_del_all[obs_fy:obs_fy+obs_pr]
        if fs_adjust:
            obs_to_del_fs = obs_to_del_all[obs_fy+obs_pr:]
    else:
        obs_to_del_pr = obs_to_del_all
    if c_dict['train_mcf']:
        if np.any(obs_to_del_tr):
            on_support_data_and_stats(obs_to_del_tr, data_tr, x_tr, tree_file,
                                      upper_l, lower_l, c_dict, header=True,
                                      d_name=d_name)
        if np.any(obs_to_del_fs):
            on_support_data_and_stats(obs_to_del_fs, data_fs, x_fs, fs_file,
                                      upper_l, lower_l, c_dict, d_name=d_name)
        if np.any(obs_to_del_fy):
            on_support_data_and_stats(obs_to_del_fy, data_fy, x_fy,
                                      fill_y_file, upper_l, lower_l, c_dict,
                                      d_name=d_name)
    if c_dict['pred_mcf']:
        if np.any(obs_to_del_pr):
            on_support_data_and_stats(
                obs_to_del_pr, data_pr, x_pr, c_dict['preddata3_temp'],
                upper_l, lower_l, c_dict, d_name=d_name)
            predict_file_new = c_dict['preddata3_temp']
        else:
            predict_file_new = predict_file
    else:
        predict_file_new = None
    return predict_file_new, cs_list, pred_tr_np, d_tr_np


def check_if_obs_needed(names_unordered, x_all_p, prime_values_dict):
    """Generate new rows -> all values of unordered variables are in data."""
    no_change, max_length = True, 1
    for name in names_unordered:
        length = len(prime_values_dict[name])
        if length > max_length:
            max_length = length
    x_add_tmp = x_all_p[:max_length].copy()
    for name in names_unordered:
        unique_vals_p = np.sort(x_all_p[name].unique())
        unique_vals_t = np.sort(prime_values_dict[name])
        if len(unique_vals_p) > len(unique_vals_t) or (
               (len(unique_vals_p) == len(unique_vals_t))
               and not np.all(unique_vals_p == unique_vals_t)):
            print(name, 'Training values: ', unique_vals_t,
                  'Prediction values:', unique_vals_p)
            raise Exception('Common support variable value error')
        add_vals_in_train = np.setdiff1d(unique_vals_t, unique_vals_p)
        if add_vals_in_train.size > 0:
            for i, val in enumerate(add_vals_in_train):
                x_add_tmp[i, name] = val
            no_change = False
    if no_change:
        return None
    return x_add_tmp


def off_support_and_plot(pred_t, pred_p, d_t, c_dict):
    """
    Plot histogrammes and indicate which observations are off support.

    Parameters
    ----------
    pred_t : N x no of treat Numpy array. Predictions of treat probs in train.
    pred_p : N x no of treat Numpy array. Predictions of treat probs in pred.
    d_t: N x no of treat Numpy array. Treatment dummies.
    c_dict : Dict. Parameters.

    Returns
    -------
    off_support_p : N x 1 Numpy array of boolean. True if obs is off support.
    off_support_t : N x 1 Numpy array of boolean. True if obs is off support.
    upper : No of treatment x 1 Numpy array of float. Upper limits.
    lower : No of treatment x 1 Numpy array of float. Lower limits.

    """
    # Normalize such that probabilities add up to 1
    if c_dict['d_type'] == 'continuous':
        d_values = c_dict['ct_grid_nn_val']
        no_of_treat = len(d_values)
    else:
        d_values, no_of_treat = c_dict['d_values'], c_dict['no_of_treat']
    pred_t = pred_t / pred_t.sum(axis=1, keepdims=True)
    pred_p = pred_p / pred_p.sum(axis=1, keepdims=True)
    n_p, n_t = np.shape(pred_p)[0], np.shape(pred_t)[0]
    q_s = c_dict['support_quantil']
    if c_dict['common_support'] == 1:
        upper_limit = np.empty((no_of_treat, no_of_treat))
        lower_limit = np.empty_like(upper_limit)
        for idx in range(no_of_treat):
            if q_s == 1:
                upper_limit[idx, :] = np.max(pred_t[d_t[:, idx] == 1], axis=0)
                lower_limit[idx, :] = np.min(pred_t[d_t[:, idx] == 1], axis=0)
            else:
                upper_limit[idx, :] = np.quantile(pred_t[d_t[:, idx] == 1],
                                                  q_s, axis=0)
                lower_limit[idx, :] = np.quantile(pred_t[d_t[:, idx] == 1],
                                                  1-q_s, axis=0)
        if c_dict['support_adjust_limits'] != 0:
            upper_limit *= 1 + c_dict['support_adjust_limits']
            lower_limit *= 1 - c_dict['support_adjust_limits']
            if c_dict['with_output']:
                print('-' * 80)
                print('Common support bounds adjusted by',
                      f'{c_dict["support_adjust_limits"]*100} % ')
                print('-' * 80)
        if c_dict['with_output']:
            print('Treatment sample     Treatment probabilities in %')
            print('--------------------- Upper limits ----------------')
            for idx, ival in enumerate(d_values):
                print(f'D = {ival:9}', end='              ')
                for jdx in range(no_of_treat):
                    print(f'{upper_limit[idx, jdx]:7.4f} ', end=' ')
                print(' ')
            print('--------------------- Lower limits ----------------')
            for idx, ival in enumerate(d_values):
                print(f'D = {ival:9}', end='              ')
                for jdx in range(no_of_treat):
                    print(f'{lower_limit[idx, jdx]:7.4f} ', end=' ')
                print(' ')
        upper, lower = np.min(upper_limit, axis=0), np.max(lower_limit, axis=0)
    else:
        # Normalize such that probabilities add up to 1
        upper = np.ones(no_of_treat) * (1 - c_dict['support_min_p'])
        lower = np.ones(no_of_treat) * c_dict['support_min_p']
    off_support_p = np.empty(n_p, dtype=bool)
    off_support_t = np.empty(n_t, dtype=bool)
    off_upper = np.empty(no_of_treat, dtype=bool)
    off_lower = np.empty_like(off_upper)
    for i in range(n_p):
        off_upper = np.any(pred_p[i, :] > upper)
        off_lower = np.any(pred_p[i, :] < lower)
        off_support_p[i] = off_upper or off_lower
    for i in range(n_t):
        off_upper = np.any(pred_t[i, :] > upper)
        off_lower = np.any(pred_t[i, :] < lower)
        off_support_t[i] = off_upper or off_lower
    if c_dict['with_output']:
        color_list = ['red', 'blue', 'green', 'violet', 'magenta', 'crimson',
                      'yellow', 'darkorange', 'khaki', 'skyblue', 'darkgreen',
                      'olive', 'greenyellow',  'aguamarine', 'deeppink',
                      'royalblue', 'navy', 'blueviolet', 'purple']
        if len(color_list) < len(d_values):
            color_list = color_list * len(d_values)
        color_list = color_list[:len(d_values)]
        for idx_p, ival_p in enumerate(d_values):  # iterate treatment probs
            treat_prob = pred_t[:, idx_p]
            titel = (f'Probability of treatment {ival_p} in different '
                     + 'subsamples')
            f_titel = f'common_support_pr_treat{ival_p}'
            file_name_jpeg = (c_dict['common_support_fig_pfad_jpeg']
                              + '/' + f_titel + '.jpeg')
            file_name_pdf = (c_dict['common_support_fig_pfad_pdf']
                             + '/' + f_titel + '.pdf')
            file_name_jpeg_d = (c_dict['common_support_fig_pfad_jpeg']
                                + '/' + f_titel + '_d.jpeg')
            file_name_pdf_d = (c_dict['common_support_fig_pfad_pdf']
                               + '/' + f_titel + '_d.pdf')
            # data_hist = []
            # for idx_sa, _ in enumerate(d_values):  # iterate treat.sample
            #     data_hist.append(treat_prob[d_t[:, idx_sa] == 1])
            data_hist = [treat_prob[d_t[:, idx_sa] == 1]
                         for idx_sa, _ in enumerate(d_values)]
            fig, axs = plt.subplots()
            fig_d, axs_d = plt.subplots()
            labels = ['Treat ' + str(d) for d in d_values]
            for idx, data in enumerate(data_hist):
                axs.hist(data, bins='auto', histtype='bar',
                         label=labels[idx], color=color_list[idx],
                         alpha=0.5, density=False)
                _, bins, _ = axs_d.hist(data, bins='auto', histtype='bar',
                                        label=labels[idx],
                                        color=color_list[idx],
                                        alpha=0.5, density=True)
                sigma = np.std(data)
                fit_line = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
                    -0.5 * (1 / sigma * (bins - np.mean(data)))**2))
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
            axs.legend(loc=c_dict['fig_legend_loc'], shadow=True,
                       fontsize=c_dict['fig_fontsize'])
            gp.delete_file_if_exists(file_name_jpeg)
            gp.delete_file_if_exists(file_name_pdf)
            fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
            fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
            axs_d.set_title(titel)
            axs_d.set_xlabel('Treatment probability')
            axs_d.set_ylabel('Density')
            axs_d.set_xlim([0, 1])
            axs_d.axvline(lower[idx_p], color='blue', linewidth=0.7,
                          linestyle="--", label='min')
            axs_d.axvline(upper[idx_p], color='black', linewidth=0.7,
                          linestyle="--", label='max')
            axs_d.legend(loc=c_dict['fig_legend_loc'], shadow=True,
                         fontsize=c_dict['fig_fontsize'])
            gp.delete_file_if_exists(file_name_jpeg_d)
            gp.delete_file_if_exists(file_name_pdf_d)
            fig_d.savefig(file_name_jpeg_d, dpi=c_dict['fig_dpi'])
            fig_d.savefig(file_name_pdf_d, dpi=c_dict['fig_dpi'])
            if c_dict['show_plots']:
                plt.show()
            else:
                plt.close()
    return off_support_p, off_support_t, upper, lower
