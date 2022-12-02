"""Created on April, 4, 2022.

Optimal Policy Trees: Functions - Python implementation

Can be used under Creative Commons Licence CC BY-SA
Michael Lechner, SEW, University of St. Gallen, Switzerland

# -*- coding: utf-8 -*-
"""
from math import inf

from dask.distributed import Client, as_completed
import pandas as pd
import numpy as np
import ray

from mcf import general_purpose as gp
from mcf import optp_tree_functions as optp_t
from mcf import optp_print as optp_p


def black_box_allocation(indata, preddata, c_dict, v_dict, seed):
    """
    Organise the estimation of the black-box (PO based) allocations.

    Parameters
    ----------
    indata : String
        Training data.
    preddata : String
        Prediction data.
    c_dict : Dict.
        Control parameters.
    v_dict : Dict
        Variables.

    Returns
    -------
    None.

    """
    rng = np.random.default_rng(seed)
    if indata != preddata:
        if v_dict['polscore_desc_name'] is None:
            vars_to_check = v_dict['polscore_name']
        else:
            vars_to_check = (v_dict['polscore_name']
                             + v_dict['polscore_desc_name'])
        use_pred_data = vars_in_data_file(preddata, vars_to_check)
    else:
        use_pred_data = False
    data_files = [indata, preddata] if use_pred_data else [indata]
    for idx, data_file in enumerate(data_files):
        if c_dict['only_if_sig_better_vs_0']:
            po_np, data_df = optp_t.adjust_policy_score(data_file, c_dict,
                                                        v_dict)
        else:
            data_df = pd.read_csv(data_file)
            po_np = data_df[v_dict['polscore_name']].to_numpy()
        allocation = bb_allocation(po_np.copy(), data_df.copy(), c_dict,
                                   v_dict, rng)
        if c_dict['bb_bootstraps'] > 0:
            maxworkers = c_dict['no_parallel']
            obs = len(po_np)
            if c_dict['_ray_or_dask'] == 'ray':
                if not ray.is_initialized():
                    ray.init(num_cpus=maxworkers, include_dashboard=False)
                po_np_ref, data_df_ref = ray.put(po_np), ray.put(data_df)
                still_running = [ray_bb_allocation_boot.remote(
                    b_idx, obs, po_np_ref, data_df_ref, c_dict, v_dict)
                        for b_idx in range(c_dict['bb_bootstraps'])]
                idx, allocation_b = 0, [None] * c_dict['bb_bootstraps']
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for ret_all_i in finished_res:
                        if c_dict['with_output']:
                            gp.share_completed(idx+1,
                                               c_dict['bb_bootstraps'])
                        allocation_b[idx] = ret_all_i
                        idx += 1
            elif c_dict['_ray_or_dask'] == 'dask':
                with Client(n_workers=maxworkers) as clt:
                    po_np_ref = clt.scatter(po_np)
                    data_df_ref = clt.scatter(data_df)
                    ret_fut = [
                        clt.submit(dask_bb_allocation_boot, b_idx, obs,
                                   po_np_ref, data_df_ref, c_dict, v_dict)
                        for b_idx in range(c_dict['bb_bootstraps'])]
                    idx, allocation_b = 0, [None] * c_dict['bb_bootstraps']
                    for _, res in as_completed(ret_fut, with_results=True):
                        if c_dict['with_output']:
                            gp.share_completed(idx+1, c_dict['bb_bootstraps'])
                        allocation_b[idx] = res
                        idx += 1
            else:   # no multiprocessing
                allocation_b = []
                for b_idx in range(c_dict['bb_bootstraps']):
                    boot_ind = rng.integers(low=0, high=len(po_np),
                                            size=len(po_np))
                    po_np_b = po_np[boot_ind, :].copy()
                    data_df_b = data_df.iloc[boot_ind, :].copy()
                    allocation_b.append(bb_allocation(po_np_b, data_df_b,
                                                      c_dict, v_dict, rng))
                    # This is list of lists, since allocation contains a list
                    # of the different allocation schemes
                    if c_dict['with_output']:
                        gp.share_completed(b_idx+1, c_dict['bb_bootstraps'])
            allocation = add_boot_stats_to_allocation(allocation, allocation_b)
        if c_dict['with_output']:
            if idx == 0 and v_dict['d_name'] is not None:
                treatment = data_df[v_dict['d_name']]  # pylint: disable=E1136
                treatment = treatment.to_numpy()
            else:
                treatment = None
            optp_p.bb_allocation_stats(allocation, c_dict, v_dict, data_file)
        black_box_alloc_pred = allocation if idx == 1 else None
    return black_box_alloc_pred


@ray.remote
def ray_bb_allocation_boot(boot, obs, po_np, data_df, c_dict, v_dict):
    """Do bootstrapping black box allocation with ray."""
    rng = np.random.default_rng((10+boot)**2+121)
    boot_ind = rng.integers(low=0, high=obs, size=obs)
    po_np_b = po_np[boot_ind, :].copy()
    data_df_b = data_df.iloc[boot_ind, :].copy()
    return bb_allocation(po_np_b, data_df_b, c_dict, v_dict, rng)


def dask_bb_allocation_boot(boot, obs, po_np, data_df, c_dict, v_dict):
    """Do bootstrapping black box allocation with dask."""
    rng = np.random.default_rng((10+boot)**2+121)
    boot_ind = rng.integers(low=0, high=obs, size=obs)
    po_np_b = po_np[boot_ind, :].copy()
    data_df_b = data_df.iloc[boot_ind, :].copy()
    return bb_allocation(po_np_b, data_df_b, c_dict, v_dict, rng)


def add_boot_stats_to_allocation(allocation, allocation_b):
    """Create Bootstrap stats and add to results dictionaries in allocation."""
    for a_idx, alloc in enumerate(allocation):
        boot_data = collect_data_all_boots(allocation_b, a_idx)
        alloc['results'].update(add_boot_stats_to_results(boot_data))
    return allocation


def add_boot_stats_to_results(boot_data):
    """Create bootstatistics and add to original results_dictionary."""
    def std_quantile(data, quants):
        if data is not None:
            std = np.std(data)
            quantiles = np.quantile(data, quants)
            return std, quantiles
        return None, None

    no_boot = len(boot_data['score'])
    if no_boot > 99:
        quants = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    elif no_boot > 50:
        quants = (0.10, 0.25, 0.50, 0.75, 0.90)
    elif no_boot > 20:
        quants = (0.25, 0.50, 0.75)
    else:
        quants = (0.33, 0.50, 0.66)
    if boot_data['score_add'] is not None:
        no_vars_add = boot_data['score_add'].shape[1]
    else:
        no_vars_add = None
    score_std, score_q = std_quantile(boot_data['score'], quants)
    score_m_obs_std, score_m_obs_q = std_quantile(boot_data['score_m_obs'],
                                                  quants)
    score_change_std, score_change_q = std_quantile(boot_data['score_change'],
                                                    quants)
    score_change_m_obs_std, score_change_m_obs_q = std_quantile(
        boot_data['score_change_m_obs'], quants)
    if no_vars_add is not None:
        score_add_std = np.zeros(no_vars_add)
        score_add_m_obs_std = np.zeros_like(score_add_std)
        score_add_q = np.zeros((no_vars_add, len(quants)))
        score_add_m_obs_q = np.zeros_like(score_add_q)
        score_change_add_std = np.zeros_like(score_add_std)
        score_change_add_m_obs_std = np.zeros_like(score_add_std)
        score_change_add_q = np.zeros_like(score_add_q)
        score_change_add_m_obs_q = np.zeros_like(score_add_q)
        for i in range(no_vars_add):
            score_add_std[i], score_add_q[i, :] = std_quantile(
                boot_data['score_add'][i], quants)
            score_add_m_obs_std[i], score_add_m_obs_q[i, :] = std_quantile(
                boot_data['score_add_m_obs'][i], quants)
            score_change_add_std[i], score_change_add_q[i, :] = std_quantile(
                boot_data['score_change_add'][i], quants)
            ret = std_quantile(boot_data['score_change_add_m_obs'][i], quants)
            score_change_add_m_obs_std[i] = ret[0]
            score_change_add_m_obs_q[i, :] = ret[1]
    else:
        score_add_std = score_add_m_obs_std = score_add_q = None
        score_add_m_obs_q = score_change_add_std = None
        score_change_add_m_obs_std = score_change_add_q = None
        score_change_add_m_obs_q = None
    bootstat_dict = {
        'score_std': score_std, 'score_q': score_q,
        'score_change_std': score_change_std, 'score_change_q': score_change_q,
        'score_add_std': score_add_std, 'score_add_q': score_add_q,
        'score_change_add_std': score_change_add_std,
        'score_change_add_q': score_change_add_q,
        'score_m_obs_std': score_m_obs_std, 'score_m_obs_q': score_m_obs_q,
        'score_change_m_obs_std': score_change_m_obs_std,
        'score_change_m_obs_q': score_change_m_obs_q,
        'score_add_m_obs_std': score_add_m_obs_std,
        'score_add_m_obs_q': score_add_m_obs_q,
        'score_change_add_m_obs_std': score_change_add_m_obs_std,
        'score_change_add_m_obs_q': score_change_add_m_obs_q,
        'quants': quants}
    return bootstat_dict


def collect_data_all_boots(allocation, a_idx):
    """Collect the data from the bootstrap of a single allocation."""
    no_boot = len(allocation)
    for b_idx, alloc_boot in enumerate(allocation):
        result = alloc_boot[a_idx]['results']
        if b_idx == 0:
            score = np.zeros(no_boot)
            score_m_obs = np.zeros_like(score)
            obs_by_treat = np.zeros((no_boot, result['no_of_treat']))
            score_change = np.zeros_like(score)
            score_change_m_obs = np.zeros_like(score)
            obs_change = np.zeros_like(score)
            if result['name_add'] is not None:
                score_add = np.zeros((no_boot, len(result['score_add'])))
                score_add_m_obs = np.zeros_like(score_add)
                score_change_add = np.zeros_like(score_add)
                score_change_add_m_obs = np.zeros_like(score_add)
            else:
                score_add = score_change_add = None
                score_add_m_obs = score_change_add_m_obs = None
        score[b_idx] = result['score']
        obs_by_treat[b_idx] = result['obs_by_treat']
        score_change[b_idx] = result['score_change']
        obs_change[b_idx] = result['obs_change']
        score_m_obs[b_idx] = result['score_m_obs']
        score_change_m_obs[b_idx] = result['score_change_m_obs']
        if result['name_add'] is not None:
            score_add[b_idx, :] = result['score_add']
            score_change_add[b_idx, :] = result['score_change_add']
            score_add_m_obs[b_idx, :] = result['score_add_m_obs']
            score_change_add_m_obs[b_idx, :] = result['score_change_add_m_obs']
    boot_data = {'score': score, 'obs_by_treat': obs_by_treat,
                 'score_change': score_change, 'obs_change': obs_change,
                 'score_add': score_add, 'score_change_add': score_change_add,
                 'score_m_obs': score_m_obs,
                 'score_change_m_obs': score_change_m_obs,
                 'score_add_m_obs': score_add_m_obs,
                 'score_change_add_m_obs': score_change_add_m_obs}
    return boot_data


def observed_fct(treatment):
    """Compute observed allocation."""
    dic = {'type': 'Observed allocation'}
    dic['alloc'] = treatment.copy()
    return dic


def largest_gain_fct(po_np):
    """Compute allocation with largest gains."""
    dic = {'type': 'Unrestricted: Largest gain'}
    dic['alloc'] = np.argmax(po_np, axis=1)
    return dic


def random_fct(no_treat, no_obs, rng):
    """Compute a random allocation."""
    dic = {'type': 'Unrestricted: random'}
    dic['alloc'] = rng.integers(0, high=no_treat, size=no_obs)
    return dic


def random_rest_fct(no_treat, no_obs, max_by_cat, rng):
    """Compute random allocation under restrictions."""
    dic = {'type': 'Restricted: random'}
    dic['alloc'] = np.zeros(no_obs)
    so_far_by_cat = np.zeros_like(max_by_cat)
    for idx in range(no_obs):
        for _ in range(10):
            draw = rng.integers(0, high=no_treat, size=1)
            max_by_cat_draw = max_by_cat[draw]  # pylint: disable=E1136
            if so_far_by_cat[draw] <= max_by_cat_draw:
                so_far_by_cat[draw] += 1
                dic['alloc'][idx] = draw
                break
    return dic


def largest_gain_rest_fct(po_np, no_treat, no_obs, max_by_cat, largest_gain):
    """Compute allocation based on largest gains, under restrictions."""
    dic = {'type': 'Restricted: Largest gain'}
    dic['alloc'], val_best_treat = np.zeros(no_obs), np.empty(no_obs)
    for i in range(no_obs):
        val_best_treat[i] = (po_np[i, largest_gain['alloc'][i]]
                             - po_np[i, 0])
    order_best_treat = np.flip(np.argsort(val_best_treat))
    dic['alloc'] = largest_gain_rest_idx_fct(order_best_treat,
                                             largest_gain['alloc'], max_by_cat,
                                             po_np.copy(), no_treat)
    return dic


def largest_gain_rest_random_order_fct(po_np, no_treat, no_obs, max_by_cat,
                                       largest_gain, rng):
    """Compute allocation based on first come first served, restricted."""
    dic = {'type':  'Restricted: Largest gain - first come first served'}
    order_random = np.arange(no_obs)
    rng.shuffle(order_random)
    dic['alloc'] = largest_gain_rest_idx_fct(
        order_random, largest_gain['alloc'], max_by_cat, po_np.copy(),
        no_treat)
    return dic


def largest_gain_rest_other_var_fct(po_np, no_treat, max_by_cat,
                                    v_dict, largest_gain, data_df):
    """Compute allocation of largest gain under restriction, order by var."""
    dic = {'type': ('Restricted: Largest gain - based on ' +
           str(*v_dict['bb_rest_variable']))}
    order_other_var = np.flip(
        np.argsort(data_df[v_dict['bb_rest_variable']].to_numpy(), axis=0))
    order_other_var = [x[0] for x in order_other_var]
    dic['alloc'] = largest_gain_rest_idx_fct(
        order_other_var, largest_gain['alloc'], max_by_cat, po_np.copy(),
        no_treat)
    return dic


def largest_gain_rest_idx_fct(order_treat, largest_gain_alloc, max_by_cat,
                              po_np, no_treat):
    """Get index of largest gain under restr. for each obs with given order."""
    def helper_largest_gain(best_last, po_np_i, so_far_by_cat, max_by_cat):
        po_np_i[best_last] = -inf
        best = np.argmax(po_np_i)
        if so_far_by_cat[best] <= max_by_cat[best]:
            so_far_by_cat[best] += 1
            success = True
        else:
            success = False
        # otherwise it remains at the zero default
        return so_far_by_cat, best, success

    so_far_by_cat = np.zeros_like(max_by_cat)
    largest_gain_rest = np.zeros_like(largest_gain_alloc)
    for i in order_treat:
        best_1 = largest_gain_alloc[i]
        if so_far_by_cat[best_1] <= max_by_cat[best_1]:
            so_far_by_cat[best_1] += 1
            largest_gain_rest[i] = best_1
        else:
            if no_treat > 2:
                so_far_by_cat, best_2, success = helper_largest_gain(
                    best_1, po_np[i], so_far_by_cat, max_by_cat)
                if success:
                    largest_gain_rest[i] = best_2
                else:
                    if no_treat > 3:
                        so_far_by_cat, best_3, success = helper_largest_gain(
                             best_2, po_np[i], so_far_by_cat, max_by_cat)
                        if success:
                            largest_gain_rest[i] = best_3
    return largest_gain_rest


def po_alloc_draw(no_obs, no_treat, effect_vs_0_np, effect_vs_0_se_np, rng):
    """Draw (normalized) policy scores"""
    po_alloc_np = np.zeros((no_obs, no_treat))
    po_alloc_np[:, 1:] = rng.normal(effect_vs_0_np, effect_vs_0_se_np)
    return po_alloc_np


def bb_allocation(po_np, data_df, c_dict, v_dict, rng):
    """
    Generate the various black-box allocations.

    Parameters
    ----------
    po_df : Dataframe
        Contains the relevant potential outcomes.
    data_df : Dataframe
        Contains dataframe with all variables.
    c_dict : Dict
        Controls.
    v_dict : Dict
        Variables.
    rng : Seeded random number generator.

    Returns
    -------
    allocation : Dict
        Dictionary of allocations.
    """
    no_obs, no_treat = po_np.shape
    allocations = []
    if v_dict['d_name'] is None:
        treatment = None
    else:
        treatment = np.int64(data_df[v_dict['d_name']].to_numpy().flatten())
        allocations.append(observed_fct(treatment))
    if c_dict['bb_stochastic']:
        po_alloc_np = po_alloc_draw(
            no_obs, no_treat, data_df[v_dict['effect_vs_0']].to_numpy(),
            data_df[v_dict['effect_vs_0_se']].to_numpy(), rng)
    else:
        po_alloc_np = po_np
    largest_gain = largest_gain_fct(po_alloc_np)
    allocations.append(largest_gain)
    allocations.append(random_fct(no_treat, no_obs, rng))
    if c_dict['restricted']:
        max_by_cat = np.int64(
            np.floor(no_obs * np.array(c_dict['max_shares'])))
        allocations.append(random_rest_fct(no_treat, no_obs, max_by_cat, rng))
        allocations.append(largest_gain_rest_fct(
            po_alloc_np, no_treat, no_obs, max_by_cat, largest_gain))
        allocations.append(largest_gain_rest_random_order_fct(
            po_alloc_np, no_treat, no_obs, max_by_cat, largest_gain, rng))
        if v_dict['bb_rest_variable']:
            allocations.append(largest_gain_rest_other_var_fct(
                po_alloc_np, no_treat, max_by_cat, v_dict, largest_gain,
                data_df))
    if v_dict['polscore_desc_name'] is not None:
        po_descr_np = data_df[v_dict['polscore_desc_name']].to_numpy()
    else:
        po_descr_np = None
    for alloc in allocations:
        alloc['results'] = compute_outcomes(alloc, po_np, po_descr_np,
                                            treatment, no_treat, no_obs,
                                            v_dict['polscore_desc_name'])
    return allocations


def vars_in_data_file(preddata, var_list):
    """Check if potential outcomes are in prediction data."""
    headers = pd.read_csv(filepath_or_buffer=preddata, nrows=0)
    header_list = [s.upper() for s in list(headers.columns.values)]
    var_in_pred = all(i in header_list for i in var_list)
    return var_in_pred


def compute_outcomes(alloc, po_np, po_desc_np, treatment, no_of_treat, obs,
                     desc_name):
    """Compute some outcome measures of allocation."""
    results = {}
    results['obs_by_treat'] = np.zeros(no_of_treat)
    for i in range(no_of_treat):
        results['obs_by_treat'][i] = np.sum(alloc['alloc'] == i)
    results['score'] = results['score_change'] = results['obs_change'] = 0
    results['score_m_obs'] = results['score_change_m_obs'] = 0
    results['changers'] = np.zeros(obs) > 10   # all elements are False
    alloc['alloc'] = np.int64(alloc['alloc'])
    for i, _ in enumerate(po_np):
        treat_alloc_i = alloc['alloc'][i]
        results['score'] += po_np[i, treat_alloc_i]
        if treatment is not None:
            change_i = po_np[i, treat_alloc_i] - po_np[i, treatment[i]]
            results['score_m_obs'] += change_i
            if treat_alloc_i != treatment[i]:
                results['score_change'] += po_np[i, treat_alloc_i]
                results['obs_change'] += 1
                results['changers'][i] = True
                results['score_change_m_obs'] += change_i
    results['obs'] = obs
    results['no_of_treat'] = no_of_treat
    if desc_name is not None:
        ret = compute_alloc_other_outcomes(desc_name, po_desc_np, no_of_treat,
                                           alloc['alloc'], results['changers'],
                                           treatment)
        results['name_add'] = desc_name
        results['score_add'], results['score_change_add'] = ret[0], ret[1]
        results['score_add_m_obs'] = ret[2]
        results['score_change_add_m_obs'] = ret[3]
    else:
        results['name_add'] = None
        results['score_add'] = results['score_change_add'] = None
        results['score_add_m_obs'] = results['score_change_add_m_obs'] = None
    return results


def compute_alloc_other_outcomes(var_name, po_np, no_of_treat, alloc,
                                 changers=None, treatment=None):
    """Compute effects using additional estimated potential outcomes."""
    if treatment is None:
        changers = None
    number_var = round(len(var_name) / no_of_treat)
    assert len(var_name) == po_np.shape[1], ('Wrong dimensions of additional'
                                             + f' outcome variable {var_name}')
    all_idx_tmp = np.arange(po_np.shape[1])
    all_idx = np.reshape(all_idx_tmp, (number_var, no_of_treat))
    score = np.zeros(number_var)
    score_m_obs, score_changers = np.zeros_like(score), np.zeros_like(score)
    score_changers_m_obs = np.zeros_like(score)
    for var_idx, idx in enumerate(all_idx):
        po_this_var = po_np[:, idx]
        for i, _ in enumerate(po_np):
            score[var_idx] += po_this_var[i, alloc[i]]
            if changers is not None:
                change_i = (po_this_var[i, alloc[i]]
                            - po_this_var[i, treatment[i]])
                score_m_obs[var_idx] += change_i
                if changers[i]:
                    score_changers[var_idx] += po_this_var[i, alloc[i]]
                    score_changers_m_obs[var_idx] += change_i
    return score, score_changers, score_m_obs, score_changers_m_obs
