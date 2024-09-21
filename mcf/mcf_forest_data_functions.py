"""Created on Sat Jun  3 07:51:00 2023.

Contains the data related functions needed for building the forst.

@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy
from math import ceil
from numba import njit, prange

import numpy as np
import pandas as pd
import ray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch

from mcf import mcf_cuda_functions as mcf_c
from mcf import mcf_data_functions as data
from mcf import mcf_general as gp
from mcf import mcf_general_sys as gp_sys
from mcf import mcf_print_stats_functions as ps


def prepare_data_for_forest(mcf_, data_df, no_y_nn=False):
    """Prepare data for Forest and variable importance estimation.

    Parameters
    ----------
    data_df : DataFrame. Data.
    no_y_nn : Bool. Default is False.

    Returns
    -------
    x_name : Dict.
    x_type :Dict.
    x_values : Dict.
    cf_dict : Dict. Parameters (updated).
    pen_mult : INT. Multiplier for penalty.
    data_np : Numpy array. Data for estimation.
    y_i : INT. Index of position of y in data_np.
    y_nn_i : Numpy array of INT.
    x_i : Numpy array of INT.
    x_ind : Numpy array of INT.
    x_ai_ind : Numpy array of INT.
    d_i : INT.
    w_i : INT.
    cl_i : INT.

    """
    cf_dic, var_dic = mcf_.cf_dict, mcf_.var_dict
    x_name, x_type = gp.get_key_values_in_list(mcf_.var_x_type)
    x_type = np.array(x_type)
    x_name2, x_values = gp.get_key_values_in_list(mcf_.var_x_values)
    if x_name != x_name2:
        raise RuntimeError('Wrong order of names', x_name, x_name2)
    p_x = len(x_name)     # Number of variables
    cf_dic = m_n_grid(cf_dic, p_x)  # Grid for # of var's used for split
    x_ind = np.array(range(p_x))    # Indices instead of names of variable
    if not var_dic['x_name_always_in'] == []:
        x_ai_ind = np.array([i for i in range(p_x)
                             if x_name[i] in var_dic['x_name_always_in']])
    else:
        x_ai_ind = []              # Indices of variables used for all splits
    if len(data_df.columns) > len(set(data_df.columns)):
        duplicates = [x for x in list(data_df.columns)
                      if list(data_df.columns).count(x) > 1]
        raise RuntimeError(f'Duplicate variable names.{duplicates}'
                           ' Names are NOT case sensitive')
    y_dat = data_df[var_dic['y_tree_name']].to_numpy()
    if cf_dic['mtot'] in (1, 4):
        pen_mult = cf_dic['p_diff_penalty'] * np.var(y_dat)
    else:
        pen_mult = 0
    d_dat, d_i = data_df[var_dic['d_name']].to_numpy(), [1]
    y_nn = (np.zeros((len(d_dat), len(var_dic['y_match_name'])))
            if no_y_nn else data_df[var_dic['y_match_name']].to_numpy())
    y_nn_i = np.arange(2, 2 + len(var_dic['y_match_name']))
    x_dat = data_df[x_name].to_numpy()
    for col_indx in range(np.shape(x_dat)[1]):
        if x_type[col_indx] > 0:
            x_dat[:, col_indx] = np.around(x_dat[:, col_indx])
    x_i = np.arange(
        2 + len(var_dic['y_match_name']),
        2 + len(var_dic['y_match_name']) + len(var_dic['x_name']))
    data_np = np.concatenate((y_dat, d_dat, y_nn, x_dat), axis=1)
    if mcf_.gen_dict['weighted']:
        w_dat = data_df[var_dic['w_name']].to_numpy()
        data_np = np.concatenate((data_np, w_dat), axis=1)
        w_i = data_np.shape[1] - 1
    else:
        w_i = None
    if mcf_.gen_dict['panel_in_rf']:
        cl_dat = data_df[var_dic['cluster_name']].to_numpy()
        data_np = np.concatenate((data_np, cl_dat), axis=1)
        cl_i = data_np.shape[1] - 1
    else:
        cl_i = None
    if mcf_.gen_dict['d_type'] == 'continuous':
        d_grid_dat = data_df[var_dic['grid_nn_name']].to_numpy()
        data_np = np.concatenate((data_np, d_grid_dat), axis=1)
        d_grid_i = data_np.shape[1] - 1
    else:
        d_grid_i = None
    y_i = [0]
    return (x_name, x_type, x_values, cf_dic, pen_mult, data_np,
            y_i, y_nn_i, x_i, x_ind, x_ai_ind, d_i, w_i, cl_i, d_grid_i)


def m_n_grid(cf_dic, no_vars):
    """Generate the grid for the # of coefficients (log-scale).Sort n_min grid.

    Parameters
    ----------
    cf_dic : Dict. Parameters of MCF estimation
    no_vars : INT. Number of x-variables used for tree building

    Returns
    -------
    cf_dic : Dict. Updated (globally) dictionary with parameters.

    """
    m_min = max(round(cf_dic['m_share_min'] * no_vars), 1)
    m_max = round(cf_dic['m_share_max'] * no_vars)
    if m_min == m_max:
        cf_dic['m_grid'] = 1
        grid_m = m_min
    else:
        if cf_dic['m_grid'] == 1:
            grid_m = round((m_min + m_max) / 2)
        else:
            grid_m = gp.grid_log_scale(m_max, m_min, cf_dic['m_grid'])
            grid_m = [int(idx) for idx in grid_m]
    if (isinstance(cf_dic['n_min_values'], (list, tuple, np.ndarray))
            and len(cf_dic['n_min_values']) > 1):
        cf_dic['n_min_values'] = sorted(cf_dic['n_min_values'], reverse=True)
    cf_dic['m_values'] = grid_m   # changes this dictionary globally
    return cf_dic


def nn_matched_outcomes(mcf_, data_df, print_out=True):
    """Find nearest neighbours."""
    var_dic, gen_dic, cf_dic = mcf_.var_dict, mcf_.gen_dict, mcf_.cf_dict
    lc_dic, var_x_type = mcf_.lc_dict, mcf_.var_x_type
    int_dic = mcf_.int_dict
    if gen_dic['x_type_1'] or gen_dic['x_type_2']:  # Expand cat var to dummy
        var_names_unordered = gp.dic_get_list_of_key_by_item(var_x_type,
                                                             [1, 2])
        x_dummies = data_df[var_names_unordered].astype('category')
        x_dummies = pd.get_dummies(x_dummies, dtype=int)
    if gen_dic['x_type_0']:
        var_names_ordered = gp.dic_get_list_of_key_by_item(var_x_type, [0])
        x_ord = data_df[var_names_ordered]
    if gen_dic['x_type_0'] and (gen_dic['x_type_1'] or gen_dic['x_type_2']):
        x_df = pd.concat([x_dummies, x_ord], axis=1)
    elif gen_dic['x_type_0'] and not (gen_dic['x_type_1']
                                      or gen_dic['x_type_2']):
        x_df = x_ord
    else:
        x_df = x_dummies
    x_dat = x_df.to_numpy()
    y_dat = data_df[var_dic['y_tree_name']].to_numpy()
    if cf_dic['match_nn_prog_score']:
        if lc_dic['yes']:
            y_dat_match = data_df[var_dic['y_tree_name_unc']].to_numpy()
        else:
            y_dat_match = y_dat
    d_name, d_values, no_of_treat = data.get_treat_info(mcf_)
    d_dat = data_df[d_name].to_numpy()
    obs = len(x_dat)                        # Reduce x_dat to prognostic scores
    if (cf_dic['match_nn_prog_score']
        and ((cf_dic['mtot'] == 1) or (cf_dic['mtot'] == 4))
            and (len(x_df.columns) >= 2 * no_of_treat)):
        if gen_dic['with_output'] and gen_dic['verbose']:
            ps.print_mcf(gen_dic, '\nComputing prognostic score for matching',
                         summary=False)
        cf_dic['nn_main_diag_only'] = False   # Use full Mahalanobis matrix
        x_dat_neu = np.empty((obs, no_of_treat))
        params = {'n_estimators': cf_dic['boot'], 'max_features': 'sqrt',
                  'bootstrap': True, 'oob_score': False,
                  'n_jobs': gen_dic['mp_parallel'],
                  'random_state': 42, 'verbose': False}
        if len(np.unique(y_dat_match)) < 10:
            y_rf_obj_all = RandomForestClassifier(**params)
        else:
            y_rf_obj_all = RandomForestRegressor(**params)
        for midx, d_val in enumerate(d_values):
            y_rf_obj = deepcopy(y_rf_obj_all)
            d_m = d_dat == d_val
            d_m = d_m.reshape(len(d_m))
            x_dat_m = x_dat[d_m, :].copy()
            y_dat_m = y_dat_match[d_m].copy()
            y_rf_obj.fit(x_dat_m, y_dat_m.ravel())
            x_dat_neu[:, midx] = y_rf_obj.predict(x_dat)
        x_dat = x_dat_neu
    if (cf_dic['match_nn_prog_score']
        and ((cf_dic['mtot'] == 1) or (cf_dic['mtot'] == 4))
            and (len(x_df.columns) < 2 * no_of_treat)):
        if gen_dic['with_output'] and gen_dic['verbose']:
            txt = '\nPrognostic scores not used for matching '
            txt += '(too few covariates)'
            ps.print_mcf(gen_dic, txt, summary=False)
    if (cf_dic['mtot'] == 1) or (cf_dic['mtot'] == 4):
        y_match = np.empty((obs, no_of_treat))
        # determine distance metric of Mahalanobis matching
        k = np.shape(x_dat)
        if k[1] > 1:
            # No gain in using GPU
            cov_x_inv = get_inv_cov(x_dat, cf_dic['nn_main_diag_only'],
                                    cuda=False)
        else:
            cov_x_inv = np.ones((1, 1))
        if int_dic['cuda'] and len(cov_x_inv) > gen_dic['no_of_treat']:
            # GPU leads to massive gains for Mahalanobis matching, but not
            # when using the low-dimensional prognostic score
            maxworkers, cuda = 1, True
        else:
            cuda = False
            # If covariance is low dimensional, using prange is fastest
            if (isinstance(cov_x_inv, (int, float))
                or len(cov_x_inv) <= gen_dic['no_of_treat']
                    or gen_dic['mp_parallel'] < 1.5):
                maxworkers = 1
            else:
                # Use ray for the higher dimensional cases
                if gen_dic['mp_automatic']:
                    maxworkers = gp_sys.find_no_of_workers(
                        gen_dic['mp_parallel'], gen_dic['sys_share'])
                else:
                    maxworkers = gen_dic['mp_parallel']
        if gen_dic['with_output'] and gen_dic['verbose']:
            print('Matching')
            txt = f'\nNumber of parallel processes for Ray: {maxworkers}'
            if int_dic['cuda'] and not cuda:
                txt += (' (GPU is currently not used for NN matching as it is'
                        ' too slow)')
            elif cuda:
                txt += '  GPU is used instead of multiple CPUs.'
            ps.print_mcf(gen_dic, txt, summary=False)
        if maxworkers == 1:
            for i_treat, i_value in enumerate(d_values):
                if cuda:
                    y_match[:, i_treat] = nn_neighbour_mcf2_cuda(
                        y_dat, x_dat, d_dat, obs, cov_x_inv, i_value, i_treat
                        )[0].flatten()
                elif gen_dic['mp_parallel'] < 1.5:
                    y_match[:, i_treat] = nn_neighbour_mcf2(
                        y_dat, x_dat, d_dat, obs, cov_x_inv, i_value, i_treat
                        )[0].flatten()
                else:  # Parallelized with prange of Numba (fastest)
                    y_match[:, i_treat] = nn_neighbour_mcf2_parallel(
                        y_dat, x_dat, d_dat, obs, cov_x_inv, i_value, i_treat
                        )[0].flatten()
        else:
            if int_dic['mp_ray_shutdown'] or int_dic['ray_or_dask'] == 'dask':
                if maxworkers > ceil(no_of_treat/2):
                    maxworkers = ceil(no_of_treat/2)
            if int_dic['ray_or_dask'] == 'ray':
                if not ray.is_initialized():
                    ray.init(num_cpus=maxworkers, include_dashboard=False)
                x_dat_ref = ray.put(x_dat)
                still_running = [ray_nn_neighbour_mcf2.remote(
                    y_dat, x_dat_ref, d_dat, obs, cov_x_inv, d_values[idx],
                    idx, cuda) for idx in range(no_of_treat)]
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for res in finished_res:
                        y_match[:, res[1]] = res[0]
                if 'refs' in int_dic['mp_ray_del']:
                    del x_dat_ref
                if 'rest' in int_dic['mp_ray_del']:
                    del finished_res, finished
                if int_dic['mp_ray_shutdown']:
                    ray.shutdown()
    else:
        y_match = np.zeros((obs, no_of_treat))
    treat_val_str = [str(i) for i in d_values]
    y_match_name = [var_dic['y_tree_name'][0] + "_NN" + treat_val_str[i]
                    for i in range(no_of_treat)]
    var_dic['y_match_name'] = y_match_name
    y_match_df = pd.DataFrame(data=y_match, columns=y_match_name,
                              index=data_df.index)
    data_new_df = pd.concat([data_df, y_match_df], axis=1)
    if gen_dic['with_output'] and print_out:
        ps.print_mcf(gen_dic, '\nMatched outcomes', summary=True)
        ps.print_descriptive_df(gen_dic, data_new_df, varnames=y_match_name,
                                summary=True)
    if gen_dic['with_output'] and int_dic['descriptive_stats'] and print_out:
        txt = '\nStatistics on matched neighbours of variable used for'
        txt += ' tree building'
        ps.print_mcf(gen_dic, txt, summary=True)
        d_name = (var_dic['grid_nn_name']
                  if gen_dic['d_type'] == 'continuous' else var_dic['d_name'])
        ps.statistics_by_treatment(
            gen_dic, data_new_df, d_name, var_dic['y_match_name'],
            only_next=gen_dic['d_type'] == 'continuous', summary=False,
            data_train_dic=mcf_.data_train_dict)
    return data_new_df, var_dic


@ray.remote
def ray_nn_neighbour_mcf2(y_dat, x_dat, d_dat, obs, cov_x_inv, treat_value,
                          i_treat, cuda):
    """Make procedure compatible for Ray."""
    if cuda:
        y_all, i_treat = nn_neighbour_mcf2_cuda(
            y_dat, x_dat, d_dat, obs, cov_x_inv, treat_value, i_treat)
    else:
        y_all, i_treat = nn_neighbour_mcf2(y_dat, x_dat, d_dat, obs, cov_x_inv,
                                           treat_value, i_treat)
    y_all = y_all.flatten()
    return y_all, i_treat


def nn_neighbour_mcf2_cuda(y_dat_np, x_dat_np, d_dat_np, obs, cov_x_inv_np,
                           treat_value, i_treat):
    """Find nearest neighbour-y in subsamples by value of d.

    Cuda version is faster for large problems. For small problems cuda and non-
    cuda vesions are fast.

    Parameters
    ----------
    y_dat : Numpy array: Outcome variable
    x_dat : Numpy array: Covariates
    d_dat : Numpy array: Treatment
    obs : INT64: Sample size
    cov_x_inv : Numpy array: inverse of covariance matrix
    treat_value : int. Treatment value
    i_treat : Position treat_values investigated

    Returns
    -------
    y_all : Numpy series with matched values.
    i_treat: see above (included to ease mulithreading which may confuse
                        positions).

    """
    # Do boolean mask on cpu instead of GPU (slow)
    mask = (d_dat_np == treat_value).reshape(-1)
    x_t_np = x_dat_np[mask, :]
    y_t_np = y_dat_np[mask]
    y_all_np = np.zeros_like(y_dat_np)
    y_all_np[mask] = y_dat_np[mask]
    all_idx = np.arange(obs)
    non_treat_idx = all_idx[np.logical_not(mask)]
    # Create tensors from numpy arrays
    prec = 32
    x_dat = torch.tensor(x_dat_np, dtype=mcf_c.tdtype('float', prec))
    x_t = torch.tensor(x_t_np, dtype=mcf_c.tdtype('float', prec))
    y_dat = torch.tensor(y_dat_np, dtype=mcf_c.tdtype('float', prec))
    y_all = torch.tensor(y_all_np, dtype=mcf_c.tdtype('float', prec))
    y_t = torch.tensor(y_t_np, dtype=mcf_c.tdtype('float', prec))
    d_dat = torch.tensor(d_dat_np, dtype=mcf_c.tdtype('int', 32))
    cov_x_inv = torch.tensor(cov_x_inv_np, dtype=mcf_c.tdtype('float', prec))
    non_treat_idx = torch.tensor(non_treat_idx)

    # Send all data to GPU
    x_dat = x_dat.to("cuda")
    x_t = x_t.to("cuda")
    y_dat = y_dat.to("cuda")
    y_t = y_t.to("cuda")
    d_dat = d_dat.to("cuda")
    cov_x_inv = cov_x_inv.to("cuda")
    non_treat_idx = non_treat_idx.to('cuda')

    # To use full GPU speed this loop needs to be expressed in tensor
    # operations. This is possible, but tensors will be too large.
    for idx in non_treat_idx:
        diff = x_t - x_dat[idx, :]
        product = torch.matmul(diff, cov_x_inv) * diff
        dist = torch.sum(product, axis=1)
        min_ind = torch.where(dist <= (dist.min() + 1e-15))
        y_all[idx] = torch.mean(y_t[min_ind])
    y_all = y_all.to('cpu')
    y_all_np = y_all.numpy()
    return y_all_np, i_treat


@njit
def nn_neighbour_mcf2(y_dat, x_dat, d_dat, obs, cov_x_inv, treat_value,
                      i_treat):
    """Find nearest neighbour-y in subsamples by value of d.

    Parallelized with ray. This is slower than the version below.

    Parameters
    ----------
    y_dat : Numpy array: Outcome variable
    x_dat : Numpy array: Covariates
    d_dat : Numpy array: Treatment
    obs : INT64: Sample size
    cov_x_inv : Numpy array: inverse of covariance matrix
    treat_values : Numpy array: possible values of D
    i_treat : Position treat_values investigated

    Returns
    -------
    y_all : Numpy series with matched values.
    i_treat: see above (included to ease mulithreading which may confuse
                        positions).

    """
    mask = (d_dat == treat_value).reshape(-1)
    # x_t = x_dat[mask, :]
    x_t = x_dat[mask]
    y_t = y_dat[mask]
    y_all = np.zeros_like(y_dat)
    y_all[mask] = y_t

    for i in range(obs):
        if treat_value != d_dat[i]:
            # x_diff = x_t - x_dat[i, :]
            x_diff = x_t - x_dat[i]
            dist = np.sum((x_diff @ cov_x_inv) * x_diff, axis=1)
            min_ind = np.where(dist <= (dist.min() + 1e-15))
            y_all[i] = np.mean(y_t[min_ind])
    return y_all, i_treat


@njit(parallel=True)
def nn_neighbour_mcf2_parallel(y_dat, x_dat, d_dat, obs, cov_x_inv, treat_value,
                               i_treat):
    """Find nearest neighbour-y in subsamples by value of d.

    Parallelized on the CPU (faster than using ray on the other function).

    Parameters
    ----------
    y_dat : Numpy array: Outcome variable
    x_dat : Numpy array: Covariates
    d_dat : Numpy array: Treatment
    obs : INT64: Sample size
    cov_x_inv : Numpy array: inverse of covariance matrix
    treat_values : Numpy array: possible values of D
    i_treat : Position treat_values investigated

    Returns
    -------
    y_all : Numpy series with matched values.
    i_treat: see above (included to ease mulithreading which may confuse
                        positions).

    """
    # Currently, does not run if cov_x_inv is matrix (at least a larger one)
    mask = (d_dat == treat_value).reshape(-1)
    # x_t = x_dat[mask, :]
    x_t = x_dat[mask]
    y_t = y_dat[mask]
    y_all = np.zeros_like(y_dat)
    y_all[mask] = y_dat[mask]
    all_idx = np.arange(obs)
    non_treat_idx = all_idx[~mask]
    n_obs = len(non_treat_idx)
    for idx in prange(n_obs):
        i = int(non_treat_idx[idx])
        # x_diff = x_t - x_dat[i, :]
        x_diff = x_t - x_dat[i]
        diffcovinv = x_diff @ cov_x_inv
        dist = np.sum(diffcovinv * x_diff, axis=1)
        min_ind = np.where(dist <= (dist.min() + 1e-15))
        y_all[i] = np.mean(y_t[min_ind])
    return y_all, i_treat


def get_inv_cov(x_dat, main_diag_only, cuda=False):
    """Get inverse of covariance matrix."""
    cols_x = x_dat.shape[1]
    if cuda:
        x_dat_t = torch.from_numpy(x_dat.T)
        x_dat_t = x_dat_t.to('cuda')
        try:
            cov_x = torch.cov(x_dat_t)
        except AttributeError:
            x_dat_t = x_dat_t.to(torch.float32)
            cov_x = torch.cov(x_dat_t)
        if main_diag_only:
            cov_x = cov_x * torch.eye(cols_x, device='cuda')   # only main diag
        rank_not_ok = True
        counter = 0
        while rank_not_ok:
            if counter == 20:
                cov_x = cov_x * torch.eye(cols_x, device='cuda')
            if counter > 20:
                cov_x_inv = torch.eye(cols_x, device='cuda')
                break
            if torch.linalg.matrix_rank(cov_x) < cols_x:
                cov_x += 0.5 * torch.diag(cov_x) * torch.eye(cols_x,
                                                             device='cuda')
                counter += 1
            else:
                cov_x_inv_t = torch.linalg.inv(cov_x)
                rank_not_ok = False
        cov_x_inv_t = cov_x_inv_t.to('cpu')
        cov_x_inv = cov_x_inv_t.numpy()
    else:
        try:
            cov_x = np.cov(x_dat, rowvar=False)
        except AttributeError:
            cov_x = np.cov(x_dat.astype(np.float32), rowvar=False)
        if main_diag_only:
            cov_x = cov_x * np.eye(cols_x)   # only main diag
        rank_not_ok = True
        counter = 0
        while rank_not_ok:
            if counter == 20:
                cov_x *= np.eye(cols_x)
            if counter > 20:
                cov_x_inv = np.eye(cols_x)
                break
            if np.linalg.matrix_rank(cov_x) < cols_x:
                cov_x += 0.5 * np.diag(cov_x) * np.eye(cols_x)
                counter += 1
            else:
                cov_x_inv = np.linalg.inv(cov_x)
                rank_not_ok = False
    return cov_x_inv
