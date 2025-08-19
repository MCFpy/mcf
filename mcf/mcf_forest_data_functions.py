"""Created on Sat Jun  3 07:51:00 2023.

Contains the data related functions needed for building the forst.

@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy
from numba import njit, prange

import numpy as np
import pandas as pd
import ray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch

from mcf import mcf_cuda_functions as mcf_c
from mcf import mcf_data_functions as mcf_data
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_gp_sys
from mcf import mcf_print_stats_functions as mcf_ps


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
    x_name, x_type = mcf_gp.get_key_values_in_list(mcf_.var_x_type)
    x_type = np.array(x_type)
    x_name2, x_values = mcf_gp.get_key_values_in_list(mcf_.var_x_values)
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
    if mcf_.int_dict['bigdata_train'] and data_np.dtype == np.float64:
        data_np = data_np.astype(np.float32)  # Saves some memory

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
            grid_m = mcf_gp.grid_log_scale(m_max, m_min, cf_dic['m_grid'])
            grid_m = [int(idx) for idx in grid_m]
    if (isinstance(cf_dic['n_min_values'], (list, tuple, np.ndarray))
            and len(cf_dic['n_min_values']) > 1):
        cf_dic['n_min_values'] = sorted(cf_dic['n_min_values'], reverse=True)
    cf_dic['m_values'] = grid_m   # changes this dictionary globally
    return cf_dic


def nn_matched_outcomes(mcf_, data_df, print_out=True):
    """Find nearest neighbours (master file)."""
    if len(data_df) > 50000:   # Otherwise ***_issue17 is faster
        data_new_df, mcf_.var_dict = nn_matched_outcomes_072(mcf_,
                                                             data_df,
                                                             print_out
                                                             )
    else:
        d_name, d_values, no_of_treat = mcf_data.get_treat_info(mcf_)
        data_nn, txt = nn_matched_outcomes_issue_17(
           data_df,
           mcf_.var_x_type,
           mcf_.var_dict['y_tree_name'],
           mcf_.var_dict['y_tree_name_unc'],
           mcf_.cf_dict['match_nn_prog_score'],
           mcf_.lc_dict['yes'],
           d_name,
           d_values,
           no_of_treat,
           mcf_.cf_dict['mtot'],
           mcf_.cf_dict['boot'],
           mcf_.gen_dict['mp_parallel'],
           mcf_.cf_dict['nn_main_diag_only'],
           )

        data_new_df = pd.concat((data_df, data_nn), axis=1)
        mcf_.var_dict['y_match_name'] = list(data_nn.columns)

        if mcf_.gen_dict['with_output']:
            mcf_ps.print_mcf(mcf_.gen_dict, txt, summary=True)
            if print_out:
                mcf_ps.print_mcf(mcf_.gen_dict,
                                 '\nMatched outcomes',
                                 summary=True
                                 )
                mcf_ps.print_descriptive_df(
                    mcf_.gen_dict,
                    data_new_df,
                    varnames=mcf_.var_dict['y_match_name'],
                    summary=True
                    )
                if mcf_.int_dict['descriptive_stats']:
                    txt = '\nStatistics on matched neighbours of variable used '
                    txt += 'for tree building'
                    mcf_ps.print_mcf(mcf_.gen_dict, txt, summary=True)
                    d_name = (mcf_.var_dict['grid_nn_name']
                              if mcf_.gen_dict['d_type'] == 'continuous'
                              else mcf_.var_dict['d_name']
                              )
                    mcf_ps.statistics_by_treatment(
                        mcf_.gen_dict, data_new_df,
                        d_name, mcf_.var_dict['y_match_name'],
                        only_next=mcf_.gen_dict['d_type'] == 'continuous',
                        summary=False,
                        data_train_dic=mcf_.data_train_dict)

    return data_new_df, mcf_.var_dict


def nn_matched_outcomes_072(mcf_, data_df, print_out=True):
    """Find nearest neighbours."""
    var_dic, gen_dic, cf_dic = mcf_.var_dict, mcf_.gen_dict, mcf_.cf_dict
    lc_dic, var_x_type = mcf_.lc_dict, mcf_.var_x_type
    int_dic = mcf_.int_dict
    if gen_dic['x_type_1'] or gen_dic['x_type_2']:  # Expand cat var to dummy
        var_names_unordered = mcf_gp.dic_get_list_of_key_by_item(var_x_type,
                                                                 [1, 2])
        x_dummies = pd.get_dummies(
            data_df[var_names_unordered].astype('category'),
            dtype=int)
    else:
        x_dummies = None
    if gen_dic['x_type_0']:
        var_names_ordered = mcf_gp.dic_get_list_of_key_by_item(var_x_type, [0])
        x_ord = data_df[var_names_ordered]
    else:
        x_ord = None
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
    else:
        y_dat_match = None
    d_name, d_values, no_of_treat = mcf_data.get_treat_info(mcf_)
    d_dat = data_df[d_name].to_numpy()
    obs = len(x_dat)                        # Reduce x_dat to prognostic scores
    if (cf_dic['match_nn_prog_score']
        and ((cf_dic['mtot'] == 1) or (cf_dic['mtot'] == 4))
            and (len(x_df.columns) >= 2 * no_of_treat)):

        if gen_dic['with_output'] and gen_dic['verbose']:
            mcf_ps.print_mcf(gen_dic,
                             '\nComputing prognostic score for matching',
                             summary=False)

        cf_dic['nn_main_diag_only'] = False   # Use full Mahalanobis matrix
        x_dat_neu = np.empty((obs, no_of_treat))

        params = {'n_estimators': cf_dic['boot'],
                  'max_features': 'sqrt',
                  'bootstrap': True,
                  'oob_score': False,
                  'n_jobs': gen_dic['mp_parallel'],
                  'random_state': 42,
                  'verbose': False}

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
        and (cf_dic['mtot'] in [1, 4])
            and (len(x_df.columns) < 2 * no_of_treat)):
        if gen_dic['with_output'] and gen_dic['verbose']:
            txt = '\nPrognostic scores not used for matching '
            txt += '(too few covariates)'
            mcf_ps.print_mcf(gen_dic, txt, summary=False)
    if cf_dic['mtot'] in [1, 4]:
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
                    maxworkers = mcf_gp_sys.find_no_of_workers(
                        gen_dic['mp_parallel'], gen_dic['sys_share'])
                else:
                    maxworkers = gen_dic['mp_parallel']
        if gen_dic['with_output'] and gen_dic['verbose']:
            print('Method used: Matching')
            if maxworkers > 1:
                txt = f'\nNumber of parallel processes (matching): {maxworkers}'
            else:
                txt = ''
            if int_dic['cuda'] and not cuda:
                txt += (' (GPU is currently not used for NN matching as it is'
                        ' too slow)')
            elif cuda:
                txt += '  GPU is used instead of multiple CPUs.'
            if txt != '':
                mcf_ps.print_mcf(gen_dic, txt, summary=False)

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
                        y_dat.astype(np.float64),
                        x_dat.astype(np.float64),
                        d_dat.astype(np.float64),
                        obs, cov_x_inv, i_value, i_treat
                        )[0].flatten()
        else:
            if not ray.is_initialized():
                mcf_gp_sys.init_ray_with_fallback(
                    maxworkers, int_dic, gen_dic,
                    ray_err_txt='Ray did not start in nearest neighbour '
                    'matching'
                    )
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
            # if int_dic['mp_ray_shutdown']:
            #     ray.shutdown()
            #     mcf_ps.print_mcf(gen_dic, 'Ray is shuting down.',
            #                      summary=False)
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
        mcf_ps.print_mcf(gen_dic, '\nMatched outcomes', summary=True)
        mcf_ps.print_descriptive_df(gen_dic, data_new_df, varnames=y_match_name,
                                    summary=True)
    if gen_dic['with_output'] and int_dic['descriptive_stats'] and print_out:
        txt = '\nStatistics on matched neighbours of variable used for'
        txt += ' tree building'
        mcf_ps.print_mcf(gen_dic, txt, summary=True)
        d_name = (var_dic['grid_nn_name']
                  if gen_dic['d_type'] == 'continuous' else var_dic['d_name'])
        mcf_ps.statistics_by_treatment(
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
    x_t = x_dat[mask]
    y_t = y_dat[mask]
    y_all = np.zeros_like(y_dat)
    y_all[mask] = y_dat[mask]
    all_idx = np.arange(obs)
    non_treat_idx = all_idx[~mask]
    n_obs = len(non_treat_idx)
    for idx in prange(n_obs):
        i = int(non_treat_idx[idx])
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


def nn_matched_outcomes_issue_17(
    df: pd.DataFrame,
    var_x_type: dict[str, int],
    y_tree_name: list[str],
    y_tree_name_unc: list[str],
    match_nn_prog_score: bool,
    lc_yes: bool,
    d_name: str,
    d_values: list[int],
    no_of_treats: int,
    mtot: int,
    boot: int,
    mp_parallel: int,
    nn_main_diag_only: bool,
    use_n_gb_memory: int = 8,
) -> pd.DataFrame:
    """
    Estimate potential outcomes using nearest neighbour matching.

    Finds nearest neighbours for each unit under different treatment conditions
    using Mahalanobis distance based on covariates (X) or estimated prognostic
    scores. It then imputes the potential outcome for a given treatment `d` for
    a unit `i` by averaging the observed outcomes of its nearest neighbour(s)
    found within the group of units that actually received treatment `d`.

    More efficient procedure based on the one provided by Dieter Verbeemen in
    issue 17.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features (X), the treatment indicator (D),
        and the outcome variable(s) (Y).
    var_x_type : dict[str, int]
        Dictionary mapping covariate names (keys) to their type (values).
        Types: 0 = ordered (continuous/ordinal), 1 = unordered type 1,
        2 = unordered type 2.
        Unordered variables will be one-hot encoded.
    y_tree_name : list[str]
        List containing the single name of the observed outcome variable used
        for matching and potentially for training the prognostic score model.
        Example: ['outcome'].
    y_tree_name_unc : list[str]
        List containing the single name of the outcome variable used for
        training the prognostic score model *if* `yes` is True. This might be
        an uncentered version of the outcome. Example: ['outcome_uncentered'].
    ...
    use_n_gb_memory: int
        Estimate the amount of GB you want to use to calculate the nearest
        neighbours.
        If the matrix-broadcasting is too large, the calculation will be
        chunked.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the matched potential outcomes for each treatment
        condition.

    Notes
    -----
    - Matching is based on Mahalanobis distance, calculated either on original
      (potentially one-hot encoded) covariates or on estimated prognostic
      scores.
    - Prognostic scores are estimated using treatment-specific RandomForest
      models
      if `match_nn_prog_score` is True and conditions are met.
    - Unordered variables specified in `var_x_type` (types 1 or 2) are
      automatically one-hot encoded before matching or prognostic score
      calculation.
    - The order of columns derived from `var_x_type` matters for reproducibility
      if `match_nn_prog_score` is True, due to feature sub-sampling in
      RandomForest.
      The implementation tries to maintain order but relies on Python dictionary
      ordering (guaranteed since Python 3.7).
    - Distance calculations are chunked to manage memory usage.
    - If the covariance matrix for Mahalanobis distance is singular or
      near-singular,
      regularization is attempted by adding scaled diagonal elements. If this
      fails, the identity matrix is used.

    """
    # Get the ordered and unordered variables
    var_names_ordered = [var_name for var_name, type_ in var_x_type.items()
                         if type_ in [0]]
    var_names_unordered = [var_name for var_name, type_ in var_x_type.items()
                           if type_ in [1, 2]]
    txt = ''
    # categorize the unordered variables
    df[var_names_unordered] = df[var_names_unordered].astype("category")

    # Get the numpy x and y data
    df_x_o = df[var_names_ordered]

    if var_names_unordered:
        df_x_u = pd.get_dummies(
            data=df[var_names_unordered],
            columns=var_names_unordered,
            prefix=var_names_unordered,
            dtype=int,
            )
        np_x = pd.concat([df_x_u, df_x_o], axis=1).values.astype(np.float32)
    else:
        np_x = df_x_o.values.astype(np.float32)

    np_y = df[y_tree_name].values.astype(np.float32)
    np_d = df[d_name].values.astype(np.float32)

    # Get th amount of observations
    n_observations = np_x.shape[0]  # Reduce x_dat to prognostic scores

    if match_nn_prog_score:

        # Set np_match_y
        if lc_yes:
            np_match_y = df[y_tree_name_unc].values
        else:
            np_match_y = np_y

# y_dat = data_df[var_dic['y_tree_name']].to_numpy()
# if cf_dic['match_nn_prog_score']:
#     if lc_dic['yes']:
#         y_dat_match = data_df[var_dic['y_tree_name_unc']].to_numpy()
#     else:
#         y_dat_match = y_dat
# else:
#     y_dat_match = None

        # Create a pseudo feature matrix
        # If the "number of features" is >= 2 * "number of treatments"
        #
        # --> Train for each treatment a model and predict
        # -- If you've 4 treatments, then the new pseudo feature matrix contains
        #     4 columns with each a prediction
        if (mtot in [1, 4]) and (np_x.shape[1] >= (2 * no_of_treats)):
            txt += '\nComputing prognostic score for matching'
            np_x_pseudo = np.empty((n_observations, no_of_treats))
            params = {
                "n_estimators": boot,
                "max_features": "sqrt",
                "bootstrap": True,
                "oob_score": False,
                "n_jobs": mp_parallel,
                "random_state": 42,
                "verbose": False,
                }

            # Train for each individual treatment
            for idx, d_treatment in enumerate(d_values):

                # Create a model, either a classifier or a regressor
                if np.unique(np_match_y).shape[0] < 10:
                    rfc = RandomForestClassifier(**params)
                else:
                    rfc = RandomForestRegressor(**params)

                # Train on the treatment group
                treatments = (np_d == d_treatment).squeeze()
                tmp_np_x = np_x[treatments].copy()
                tmp_np_y = np_match_y[treatments].squeeze().copy()

                # train the model
                rfc.fit(tmp_np_x, tmp_np_y)

                # predict on the full dataset
                np_x_pseudo[:, idx] = rfc.predict(np_x)

            # overwrite np_x
            np_x = np_x_pseudo.copy()

        else:
            # if gen_dic["with_output"] and gen_dic["verbose"]:
            #     txt = "\nPrognostic scores not used for matching "
            #     txt += "(too few covariates)"
            #     ps.print_mcf(gen_dic, txt, summary=False)
            txt += ("\nPrognostic scores not used for matching "
                    "(too few covariates)"
                    )
    else:
        np_match_y = None

    if mtot in [1, 4]:

        # determine distance metric of Mahalanobis matching
        n_cols = np_x.shape[1]
        inv_cov_x = np.ones((1, 1))
        if n_cols > 1:

            # calculate covariance matrix if we've more than one column
            cov_x = np.cov(np.array(np_x), rowvar=False)

            # # only main diag
            if nn_main_diag_only:
                cov_x = cov_x * np.eye(cov_x.shape[1])

            for i in range(21):
                if i == 20:
                    cov_x *= np.eye(n_cols)
                if np.linalg.matrix_rank(cov_x) < n_cols:
                    cov_x += 0.5 * np.diag(cov_x) * np.eye(n_cols)
                else:
                    inv_cov_x = np.linalg.inv(cov_x)
                    break
            else:
                # if attempt 20 didn't work, the for loop was never been broken
                # therefor we'll enter the "else" part
                # and we have to set the covariance matrix to the identity
                # matrix
                inv_cov_x = np.eye(n_cols)
        inv_cov_x = inv_cov_x.astype(np.float32)
        #
        # Calculate the distances
        #

        # Placeholder for the matched outcomes
        np_match_y = np.empty((n_observations, no_of_treats), dtype=np.float32)

        # Loop over the treatment groups
        for i_treat, i_value in enumerate(d_values):

            # mask the treatment group
            mask = (np_d == i_value).squeeze()

            # set the result
            np_match_y[:, i_treat] = np_y.squeeze()

            # get the masked values
            temp_np_x = np_x[mask, :]
            temp_np_control = np_x[~mask, :]
            temp_np_y = np_y[mask, :]

            # chunk the mask
            n_chunks = np.ceil((4 * mask[mask].sum() * (~mask).sum()
                                * np_x.shape[1]) / (use_n_gb_memory * 1024**3))
            chunk_size = int(len(temp_np_control) // n_chunks) + 1

            # calculate the differences in chunks
            for i_chunk in range(0, len(temp_np_control), chunk_size):

                # calculate the diff and distances
                x_diff = (temp_np_x[None, :, :]
                          - temp_np_control[i_chunk: i_chunk + chunk_size,
                                            None, :])  # type: ignore
                dist = np.sum((x_diff @ inv_cov_x) * x_diff, axis=2)

                # Find minimal distance matches
                is_min_dist = dist <= (dist.min(axis=1, keepdims=True) + 1e-15)

                # n_element for the mean
                n_min = is_min_dist.sum(axis=1)

                # calculate the mean
                np_match_y[np.flatnonzero(~mask)[i_chunk: i_chunk + chunk_size],
                           i_treat] = ((is_min_dist @ temp_np_y).squeeze()
                                       / n_min)  # type: ignore

    else:
        np_match_y = np.zeros((n_observations, no_of_treats))

    # Create the names of the new columns
    match_y_names = [f"{y_tree_name[0]}_NN_{d}" for d in d_values]

    return (pd.DataFrame(data=np_match_y,
                         columns=match_y_names, index=df.index,
                         ), txt)
