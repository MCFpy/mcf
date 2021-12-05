"""Created on Sat Sep 12 12:59:15 2020.

Contains the functions needed for the running all parts of the programme

@author: MLechner
# -*- coding: utf-8 -*-
"""
from concurrent import futures
import numpy as np
import pandas as pd
from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est
from mcf import general_purpose_mcf as gp_mcf


def predict_hf(weights, data_file, y_data, cl_data, w_data, v_dict, c_dict,
               print_predictions=True):
    """Compute predictions & their std.errors and save to file (hon forest).

    Parameters
    ----------
    weights : List of lists. For every obs, positive weights are saved.
    pred_data : String. csv-file with data to make predictions for.
    y_data : Numpy array. All outcome variables.
    cl_data : Numpy array. Cluster variable.
    w_data : Numpy array. Sampling weights.
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.

    Returns
    -------
    post_estimation_file : String. Name of files with predictions.
    y_pred : Numpy array. Predictions.
    y_pred_se: Numpy array. Standard errors of predictions.
    names_y_pred: List of strings: All names of Predictions in file.
    names_y_pred_se: List of strings: All names of SE of predictions in file.

    """
    if c_dict['with_output'] and not c_dict['print_to_file']:
        print('\nComputing predictions')
    n_x = len(weights)
    n_y = np.size(y_data, axis=0)
    no_of_out = len(v_dict['y_name'])
    larger_0 = equal_0 = mean_pos = std_pos = gini_all = gini_pos = 0
    share_censored = 0
    share_largest_q = np.zeros(3)
    sum_larger = np.zeros(len(c_dict['q_w']))
    obs_larger = np.zeros_like(sum_larger)
    pred_y = np.empty((n_x, no_of_out))
    pred_y_se = np.empty_like(pred_y)
    if not c_dict['w_yes']:
        w_data = None
    else:
        if np.size(y_data, axis=0) != np.size(w_data, axis=0):
            raise Exception('Output variable has different no of observations',
                            'than weights')
    if c_dict['cluster_std']:
        no_of_cluster = np.size(np.unique(cl_data))
    else:
        no_of_cluster = None
    l1_to_9 = [None] * n_x
    if c_dict['no_parallel'] < 1.5:
        maxworkers = 1
    else:
        if c_dict['mp_automatic']:
            maxworkers = gp_mcf.find_no_of_workers(c_dict['no_parallel'],
                                                   c_dict['sys_share'])
        else:
            maxworkers = c_dict['no_parallel']
    if c_dict['with_output']:
        print('Number of parallel processes: ', maxworkers)
    if maxworkers == 1:
        for idx in range(n_x):
            ret_all_i = pred_func1_for_mp(idx, weights[idx][0], cl_data,
                                          no_of_cluster, w_data, y_data,
                                          no_of_out, n_y, c_dict)
            pred_y[idx, :] = ret_all_i[1]
            pred_y_se[idx, :] = ret_all_i[2]
            l1_to_9[idx] = ret_all_i[3]
            share_censored += ret_all_i[4]/n_x
            if c_dict['with_output'] and not c_dict['print_to_file']:
                gp.share_completed(idx+1, n_x)
    else:
        with futures.ProcessPoolExecutor(max_workers=maxworkers) as fpp:
            ret_fut = {fpp.submit(pred_func1_for_mp, idx, weights[idx][0],
                                  cl_data, no_of_cluster, w_data, y_data,
                                  no_of_out, n_y, c_dict):
                       idx for idx in range(n_x)}
            for jdx, fer in enumerate(futures.as_completed(ret_fut)):
                ret_all_i = fer.result()
                del ret_fut[fer]
                iix = ret_all_i[0]
                pred_y[iix, :] = ret_all_i[1]
                pred_y_se[iix, :] = ret_all_i[2]
                l1_to_9[iix] = ret_all_i[3]
                share_censored += ret_all_i[4]/n_x
                if c_dict['with_output'] and not c_dict['print_to_file']:
                    gp.share_completed(jdx+1, n_x)
    for idx in range(n_x):
        larger_0 += l1_to_9[idx][0]
        equal_0 += l1_to_9[idx][1]
        mean_pos += l1_to_9[idx][2]
        std_pos += l1_to_9[idx][3]
        gini_all += l1_to_9[idx][4]
        gini_pos += l1_to_9[idx][5]
        share_largest_q += l1_to_9[idx][6]
        sum_larger += l1_to_9[idx][7]
        obs_larger += l1_to_9[idx][8]
    if c_dict['with_output'] and print_predictions:
        print('\n')
        print('=' * 80)
        print('Analysis of weights (normalised to add to 1): ', 'Predicted',
              '(stats are averaged over all predictions)')
        print_weight_stat_pred(larger_0 / n_x, equal_0 / n_x, mean_pos / n_x,
                               std_pos / n_x, gini_all / n_x, gini_pos / n_x,
                               share_largest_q / n_x, sum_larger / n_x,
                               obs_larger / n_x, c_dict, share_censored)
        print_pred(y_data, pred_y, pred_y_se, w_data, c_dict, v_dict)
    # Add results to data file
    name_pred = []
    idx = 0
    for o_name in v_dict['y_name']:
        name_pred += [o_name]
    name_pred_y = [s + '_pred' for s in name_pred]
    name_pred_y_se = [s + '_pred_se' for s in name_pred]
    pred_y_df = pd.DataFrame(data=pred_y, columns=name_pred_y)
    pred_y_se_df = pd.DataFrame(data=pred_y_se, columns=name_pred_y_se)
    data_df = pd.read_csv(data_file)
    df_list = [data_df, pred_y_df, pred_y_se_df]
    data_file_new = pd.concat(df_list, axis=1)
    gp.delete_file_if_exists(c_dict['pred_sample_with_pred'])
    data_file_new.to_csv(c_dict['pred_sample_with_pred'], index=False)
    if c_dict['with_output'] and print_predictions:
        gp.print_descriptive_stats_file(
            c_dict['pred_sample_with_pred'], 'all', c_dict['print_to_file'])
    return (c_dict['pred_sample_with_pred'], pred_y, pred_y_se, name_pred_y,
            name_pred_y_se)


def pred_func1_for_mp(idx, weights_idx, cl_data, no_of_cluster, w_data, y_data,
                      no_of_out, n_y, c_dict):
    """
    Compute function to be looped over observations for Multiprocessing.

    Parameters
    ----------
    idx : Int. Counter.
    weights_i : List of int. Indices of non-zero weights.
    cl_data : Numpy vector. Cluster variable.
    no_of_cluster : Int. Number of clusters.
    w_data : Numpy vector. Sampling weights.
    y : Numpy array. Outcome variable.
    no_of_out : Int. Number of outcomes.
    n_y : Int. Length of outcome data.
    c_dict : Dict. Parameters.

    Returns
    -------
    i: Int. Counter.
    pred_y_i: Numpy array.
    pred_y_se_i: Numpy array.
    l1_to_9: Tuple of lists.
    """
    pred_y_idx = np.empty(no_of_out)
    pred_y_se_idx = pred_y_idx.copy()
    if c_dict['cluster_std']:
        w_add = np.zeros(no_of_cluster)
    else:
        w_add = np.zeros(n_y)
    w_index = weights_idx[0]  # Indices of non-zero weights
    w_i = np.array(weights_idx[1], copy=True)
    if c_dict['w_yes']:
        # w_t = w_data[w_index].flatten()
        w_t = w_data[w_index].reshape(-1)
        w_i = w_i * w_t
    else:
        w_t = None
    w_i = w_i / np.sum(w_i)
    if c_dict['max_weight_share'] < 1:
        w_i, _, share_i = gp_mcf.bound_norm_weights(
            w_i, c_dict['max_weight_share'])
    if c_dict['cluster_std']:
        cl_i = np.copy(cl_data[w_index])
        w_all_i = np.zeros(n_y)
        w_all_i[w_index] = w_i.flatten()
    else:
        cl_i = 0
    for odx in range(no_of_out):
        ret = gp_est.weight_var(w_i, y_data[w_index, odx], cl_i, c_dict,
                                weights=w_t)
        # pred_y_idx[odx] = np.copy(ret[0])
        pred_y_idx[odx] = ret[0]
        pred_y_se_idx[odx] = np.sqrt(ret[1])
        if c_dict['cluster_std']:
            ret2 = gp_est.aggregate_cluster_pos_w(
                cl_data, w_all_i, y_data[:, odx], sweights=w_data)
            if odx == 0:
                # w_add = np.copy(ret2[0])
                w_add = ret2[0]
        else:
            if odx == 0:
                # w_add[w_index] = np.copy(ret[2])
                w_add[w_index] = ret[2]
    l1_to_9 = analyse_weights_pred(w_add, c_dict)
    return idx, pred_y_idx, pred_y_se_idx, l1_to_9, share_i


def analyse_weights_pred(weights, c_dict):
    """Describe the weights.

    Parameters
    ----------
    weights : Numyp array. Weights.
    title : String. Title for output.
    c : Dict. Parameters.

    Returns
    -------
    larger_0 : Numpy array.
    equal_0 : Numpy array.
    mean_pos : Numpy array.
    std_pos : Numpy array.
    gini_all : Numpy array.
    gini_pos : Numpy array.
    share_largest_q : Numpy array.
    sum_larger : Numpy array.
    obs_larger : Numpy array.

    """
    w_j = weights.flatten()
    w_j = w_j / np.sum(w_j)
    w_pos = w_j[w_j > 1e-15]
    # n_pos = np.size(w_pos)
    n_pos = len(w_pos)
    larger_0 = (w_j > 1e-15).sum()
    equal_0 = (w_j <= 1e-15).sum()
    mean_pos = np.mean(w_pos)
    std_pos = np.std(w_pos)
    # gini_all = gp_est.gini_coefficient(w_j) * 100
    # gini_pos = gp_est.gini_coefficient(w_pos) * 100
    gini_all = gp_est.gini_coeff_pos(w_j, len(w_j)) * 100
    gini_pos = gp_est.gini_coeff_pos(w_pos, n_pos) * 100
    qqq = np.quantile(w_pos, (0.99, 0.95, 0.9))
    share_largest_q = np.empty(3)
    sum_larger = np.empty(len(c_dict['q_w']))
    obs_larger = np.empty(len(c_dict['q_w']))
    for idx in range(3):
        share_largest_q[idx] = np.sum(w_pos[w_pos >= qqq[idx]]) * 100
    for iix, val in enumerate(c_dict['q_w']):
        sum_larger[iix] = np.sum(w_pos[w_pos >= (val - 1e-15)]) * 100
        obs_larger[iix] = np.size(w_pos[w_pos >= (val - 1e-15)]) / n_pos * 100
    return (larger_0, equal_0, mean_pos, std_pos, gini_all, gini_pos,
            share_largest_q, sum_larger, obs_larger)


def print_weight_stat_pred(larger_0, equal_0, mean_pos, std_pos, gini_all,
                           gini_pos, share_largest_q, sum_larger, obs_larger,
                           c_dict, share_censored=0):
    """Print the weight statistics.

    Parameters
    ----------
    larger_0 : Numpy array.
    equal_0 : Numpy array.
    mean_pos : Numpy array.
    std_pos : Numpy array.
    gini_all : Numpy array.
    gini_pos : Numpy array.
    share_largest_q :Numpy array.
    sum_larger : Numpy array.
    obs_larger : Numpy array.
    share_censored: Numpy array.
    q : List.
    c : Dict. Parameters.

    Returns
    -------
    None.

    """
    print('# of weights > 0: {0:<6}'.format(larger_0), ', ',
          '# of weights = 0: {0:<6}'.format(equal_0), ', ',
          'Mean of positive weights: {:7.4f}'.format(mean_pos), ', ',
          'Std of positive weights: {:7.4f}'.format(std_pos))
    print('Gini coefficient (incl. weights=0):                        ',
          '{:7.4f}%'.format(gini_all))
    print('Gini coefficient (weights > 0):                            ',
          '{:7.4f}%'.format(gini_pos))
    print('Share of 1% / 5% / 10% largest weights of all weights > 0: ',
          '{:7.4f}% {:7.4f}% {:7.4f}%'.format(share_largest_q[0],
                                              share_largest_q[1],
                                              share_largest_q[2]))
    print('Share of weights > 0.5,0.25,0.1,0.05,...,0.01 (among w>0): ',
          end=' ')
    for idx in range(len(c_dict['q_w'])):
        print('{:7.4}%'.format(sum_larger[idx]), end=' ')
    print('\nShare of obs. with weights > 0.5, ..., 0.01   (among w>0): ',
          end=' ')
    for idx in range(len(c_dict['q_w'])):
        print('{:7.4}%'.format(obs_larger[idx]), end=' ')
    print('\n')
    if share_censored > 0:
        print('Share of weights censored at {:8.3f}%: '.format(
            c_dict['max_weight_share']*100), '{:8.3f}% '.format(
            share_censored*100))
    print('=' * 80)


def print_pred(y_data, y_pred, y_pred_se, w_data, c_dict, v_dict):
    """Print statistics for predictions.

    Parameters
    ----------
    y_data : 2D Numpy array. Effects. (obs x outcomes)
    y_pred_se : 2D Numpy array. Standard errors.
    effect_list : List. Names of effects.
    v_dict : Dict. Variables.

    Returns
    -------
    None.

    """
    no_outcomes = np.size(y_pred, axis=1)
    if not c_dict['w_yes']:
        w_data = 0
    print('\n')
    print('=' * 80, '\nDescriptives for prediction', '\n', '-' * 80)
    for odx in range(no_outcomes):
        if c_dict['orf']:
            print('\nProbability of: ', v_dict['orf_y_name'][odx])
        else:
            print('\nOutcome variable: ', v_dict['y_name'][odx])
        print('- ' * 40)
        print('     Mean      Median      Std       mean(SE)  R2(%)')
        # est = y_pred[:, odx].flatten()
        est = y_pred[:, odx].reshape(-1)
        # est_se = y_pred_se[:, odx].flatten()
        est_se = y_pred_se[:, odx].reshape(-1)
        if (not c_dict['orf']) and (np.shape(y_data)[0] ==
                                    np.shape(y_pred)[0]):
            r_2 = gp_est.r_squared(y_data[:, odx],
                                   y_pred[:, odx], w_data) * 100
        else:
            r_2 = None
        print('{:10.5f} {:10.5f} {:10.5f}'.format(
            np.mean(est), np.median(est), np.std(est)), end=' ')
        print(' {:10.5f} '.format(np.mean(est_se)), end=' ')
        if r_2 is None:
            print('Not computed.')
        else:
            print(' {:6.2f}% '.format(r_2))
        print('-' * 80, '\n')
