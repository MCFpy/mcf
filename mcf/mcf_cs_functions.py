"""
Procedures needed for Common support estimation.

Created on Thu Dec  8 15:48:57 2020.

@author: MLechner

# -*- coding: utf-8 -*-
"""
import copy
import numpy as np
import pandas as pd
import mcf.general_purpose as gp


def common_support(train_file, predict_file, var_x_type, v_dict, c_dict,
                   cs_list=None, prime_values_dict=None, pred_t=None,
                   d_train=None):
    """
    Remove observations from prediction file that are off-support.

    Parameters
    ----------
    train_file : String of csv-file. Data to train the RF.
    predict_file : String of csv-file. Data to predict the RF.
    var_x_type : Dict. Features.
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.
    cs_list: Tuple. Contains the information from estimated propensity score
                    needed to predict for other data. Default is None.
    prime_values_dict: Dict. List of unique values for variables to dummy.
                    Default is None.
    pred_t: Numpy array. Predicted treatment probabilities in training data.
    d_train: Numpy series. Observed treatment in training data.

    Returns
    -------
    predict_file_new : String of csv-file. Adjusted data.
    cs_list: Tuple. Contains the information from estimated propensity score
                    needed to predict for other data.
    pred_t: Numpy array. Predicted treatment probabilities in training data.

    """
    def r2_obb(c_dict, idx, oob_best):
        if c_dict['with_output']:
            print('\n')
            print('-' * 80)
            print('Treatment: {:2}'.format(c_dict['d_values'][idx]),
                  'OOB Score (R2 in %): {:6.3f}'.format(oob_best * 100))
            print('-' * 80)

    x_name, x_type = gp.get_key_values_in_list(var_x_type)
    names_unordered = []  # Split ordered variables into dummies
    for j, val in enumerate(x_type):
        if val > 0:
            names_unordered.append(x_name[j])
    if c_dict['train_mcf']:
        indata = pd.read_csv(train_file)
        x_all_t = indata[x_name]      # deep copies
    if c_dict['pred_mcf']:
        preddata = pd.read_csv(predict_file)
        x_all_p = preddata[x_name]
    if names_unordered:  # List is not empty
        if c_dict['train_mcf'] and c_dict['pred_mcf']:
            n_obs_t = len(x_all_t.index)
            x_total = pd.concat([x_all_t, x_all_p], axis=0)
            x_dummies = pd.get_dummies(x_total[names_unordered],
                                       columns=names_unordered)
            x_total = pd.concat([x_total, x_dummies], axis=1)
            x_all_t = x_total[:n_obs_t]
            x_all_p = x_total[n_obs_t:]
        elif c_dict['train_mcf'] and not c_dict['pred_mcf']:
            x_dummies = pd.get_dummies(x_all_t[names_unordered],
                                       columns=names_unordered)
            x_all_t = pd.concat([x_all_t, x_dummies], axis=1)
        else:
            x_add_tmp = check_if_obs_needed(names_unordered, x_all_p,
                                            prime_values_dict)
            if x_add_tmp is not None:
                n_obs_add = len(x_add_tmp.index)
                x_total = pd.concat([x_all_p, x_add_tmp], axis=0)
            else:
                x_total = x_all_p
            x_dummies = pd.get_dummies(x_total[names_unordered],
                                       columns=names_unordered)
            x_all_p = pd.concat([x_total, x_dummies], axis=1)
            if x_add_tmp is not None:
                x_all_p = x_all_p[n_obs_add:]
    if c_dict['train_mcf']:
        x_name_all = x_all_t.columns.values.tolist()
    else:
        x_name_all = x_all_p.columns.values.tolist()
    if c_dict['train_mcf']:
        x_train = x_all_t.to_numpy(copy=True)
        d_all_in = pd.get_dummies(indata[v_dict['d_name']],
                                  columns=v_dict['d_name'])
        d_train = d_all_in.to_numpy(copy=True)
        pred_t = np.empty(np.shape(d_train))
    if c_dict['pred_mcf']:
        x_pred = x_all_p.to_numpy(copy=True)
        pred_p = np.empty((np.shape(x_pred)[0], c_dict['no_of_treat']))
    if c_dict['no_parallel'] > 1:
        workers_mp = copy.copy(c_dict['no_parallel'])
    else:
        workers_mp = None
    if c_dict['train_mcf'] and c_dict['pred_mcf']:
        if np.shape(x_train)[1] != np.shape(x_pred)[1]:
            raise Exception('Prediction data and input data have differnt' +
                            'number of columns')
    if c_dict['train_mcf']:
        if c_dict['with_output'] and c_dict['verbose']:
            print('\n')
            print('-' * 80)
            print('Computing random forest based common support')
    if c_dict['train_mcf'] and c_dict['pred_mcf']:
        cs_list = []
        for idx in range(c_dict['no_of_treat']):
            return_forest = bool(c_dict['save_forest'])
            # if c_dict['save_forest']:
            #     return_forest = True
            # else:
            #     return_forest = False
            ret_rf = gp.RandomForest_scikit(
                x_train, d_train[:, idx], x_pred, boot=c_dict['boot'],
                n_min=c_dict['grid_n_min'], no_features=c_dict['grid_m'],
                workers=workers_mp, pred_p_flag=True, pred_t_flag=True,
                pred_oob_flag=True, with_output=False,
                variable_importance=True, x_name=x_name_all,
                var_im_with_output=c_dict['with_output'],
                return_forest_object=return_forest)
            pred_p[:, idx] = np.copy(ret_rf[0])
            pred_t[:, idx] = np.copy(ret_rf[1])
            oob_best = np.copy(ret_rf[2])
            if c_dict['save_forest']:
                cs_list.append(ret_rf[6])
            r2_obb(c_dict, idx, oob_best)
            if c_dict['no_of_treat'] == 2:
                pred_p[:, idx+1] = 1 - pred_p[:, idx]
                pred_t[:, idx+1] = 1 - pred_t[:, idx]
                break
    elif c_dict['train_mcf'] and not c_dict['pred_mcf']:
        cs_list = []
        for idx in range(c_dict['no_of_treat']):
            ret_rf = gp.RandomForest_scikit(
                x_train, d_train[:, idx], None, boot=c_dict['boot'],
                n_min=c_dict['grid_n_min'], no_features=c_dict['grid_m'],
                workers=workers_mp, pred_p_flag=False, pred_t_flag=True,
                pred_oob_flag=True, with_output=False,
                variable_importance=True, x_name=x_name_all,
                var_im_with_output=c_dict['with_output'],
                return_forest_object=True)
            pred_t[:, idx] = np.copy(ret_rf[1])
            oob_best = np.copy(ret_rf[2])
            if c_dict['save_forest']:
                cs_list.append(ret_rf[6])
            r2_obb(c_dict, idx, oob_best)
            if c_dict['no_of_treat'] == 2:
                pred_t[:, idx+1] = 1 - pred_t[:, idx]
                break
    elif c_dict['pred_mcf'] and not c_dict['train_mcf']:
        for idx in range(c_dict['no_of_treat']):
            pred_p[:, idx] = cs_list[idx].predict(x_pred)
            if c_dict['no_of_treat'] == 2:
                pred_p[:, idx+1] = 1 - pred_p[:, idx]
                break
    if c_dict['pred_mcf']:
        obs_to_del, upper, lower = indicate_off_support(pred_t, pred_p,
                                                        d_train, c_dict)
        if np.any(obs_to_del):
            obs_to_keep = np.invert(obs_to_del)
            data_keep = pd.DataFrame(preddata[obs_to_keep])
            data_delete = pd.DataFrame(preddata[obs_to_del])
            predict_file_new = c_dict['preddata3_temp']
            gp.delete_file_if_exists(predict_file_new)
            gp.delete_file_if_exists(c_dict['off_support_temp'])
            data_keep.to_csv(predict_file_new, index=False)
            data_delete.to_csv(c_dict['off_support_temp'], index=False)
        else:
            predict_file_new = predict_file
        if c_dict['with_output']:
            print('\n')
            print('=' * 80)
            print('Common support check')
            print('-' * 80)
            print('Upper limits on treatment probabilities: ', upper)
            print('Lower limits on treatment probabilities: ', lower)
            if np.any(obs_to_del):
                print('Observations deleted: {:4}'.format(np.sum(obs_to_del)),
                      ' ({:6.3f}%)'.format(np.mean(obs_to_del)*100))
                # if np.sum(obs_to_del) == np.size(obs_to_del):
                if np.sum(obs_to_del) == len(obs_to_del):
                    raise Exception('No observations left after common' +
                                    'support adjustment. Programme terminated.'
                                    )
                gp.print_descriptive_stats_file(predict_file_new, x_name,
                                                c_dict['print_to_file'])
                gp.print_descriptive_stats_file(c_dict['off_support_temp'],
                                                x_name,
                                                c_dict['print_to_file'])
            else:
                print("No observations deleted.")
            print('-' * 80)
    else:
        predict_file_new = None
    return predict_file_new, cs_list, pred_t, d_train


def check_if_obs_needed(names_unordered, x_all_p, prime_values_dict):
    """Generate new rows -> all values of unordered variables are in data."""
    no_change = True
    max_length = 1
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


def indicate_off_support(pred_t, pred_p, d_t, c_dict):
    """
    Indicate which observation is off support.

    Parameters
    ----------
    pred_t : N x no of treat Numpy array. Predictions of treat probs in train.
    pred_p : N x no of treat Numpy array. Predictions of treat probs in pred.
    d_t: N x no of treat Numpy array. Treatment dummies.
    c_dict : Dict. Parameters.

    Returns
    -------
    off_support : NN x 1 Numpy array of boolean. True if obs is off support.

    """
    # Normalize such that probabilities add up to 1
    pred_t = pred_t / pred_t.sum(axis=1, keepdims=True)
    pred_p = pred_p / pred_p.sum(axis=1, keepdims=True)
    n_p = np.shape(pred_p)[0]
    q_s = c_dict['support_quantil']
    if c_dict['common_support'] == 1:
        upper_limit = np.empty((c_dict['no_of_treat'],
                                c_dict['no_of_treat']))
        lower_limit = np.empty((c_dict['no_of_treat'],
                                c_dict['no_of_treat']))
        for idx in range(c_dict['no_of_treat']):
            if q_s == 1:
                upper_limit[idx, :] = np.max(pred_t[d_t[:, idx] == 1], axis=0)
                lower_limit[idx, :] = np.min(pred_t[d_t[:, idx] == 1], axis=0)
            else:
                upper_limit[idx, :] = np.quantile(pred_t[d_t[:, idx] == 1],
                                                  q_s, axis=0)
                lower_limit[idx, :] = np.quantile(pred_t[d_t[:, idx] == 1],
                                                  1-q_s, axis=0)
        if c_dict['with_output']:
            print('Treatment sample     Treatment probabilities in %')
            print('--------------------- Upper limits ----------------')
            for idx, ival in enumerate(c_dict['d_values']):
                print('D = {:2}'.format(ival), end='              ')
                for jdx in range(c_dict['no_of_treat']):
                    print('{:7.4f} '.format(upper_limit[idx, jdx]), end=' ')
                print(' ')
            print('--------------------- Lower limits ----------------')
            for idx, ival in enumerate(c_dict['d_values']):
                print('D = {:2}'.format(ival), end='              ')
                for jdx in range(c_dict['no_of_treat']):
                    print('{:7.4f} '.format(lower_limit[idx, jdx]), end=' ')
                print(' ')
        upper = np.min(upper_limit, axis=0)
        lower = np.max(lower_limit, axis=0)
    else:
        # Normalize such that probabilities add up to 1
        upper = np.ones(
            c_dict['no_of_treat']) * (1 - c_dict['support_min_p'])
        lower = np.ones(c_dict['no_of_treat']) * c_dict['support_min_p']
    off_support = np.empty(n_p, dtype=bool)
    off_upper = np.empty(c_dict['no_of_treat'], dtype=bool)
    off_lower = np.empty(c_dict['no_of_treat'], dtype=bool)
    for i in range(n_p):
        off_upper = np.any(pred_p[i, :] > upper)
        off_lower = np.any(pred_p[i, :] < lower)
        off_support[i] = off_upper or off_lower
    return off_support, upper, lower
