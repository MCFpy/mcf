"""
Created on Mon Jun 19 15:51:30 2023.

@author: MLechner

Contains the functions needed for computing the IATE.

@author: MLechner
-*- coding: utf-8 -*-
"""
from math import ceil

import numpy as np
import torch

from mcf import mcf_cuda_functions as mcf_cuda
from mcf import mcf_estimation_cuda_functions as mcf_est_cuda
from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats_functions as ps
from mcf import mcf_general_sys as mcf_sys


def iate_cuda(weights_list, cl_dat_np, no_of_cluster, w_dat_np, w_ate_np,
              y_dat_np, no_of_out, n_y, ct_dic, int_dic, gen_dic, p_dic,
              iate_se_flag, se_boot_iate, iate_m_ate_flag, n_x,
              no_of_treat_dr):
    """Compute IATE and SE of IATE using GPU."""
    # Transform relevant input to tensor and move to gpu
    precision = 32
    weights, cl_dat, w_dat, w_ate, y_dat = tensor_to_gpu(
        weights_list, cl_dat_np, w_dat_np, w_ate_np, y_dat_np,
        int_dic['weight_as_sparse'], iate_m_ate_flag, precision)

    pot_y = torch.empty((n_x, no_of_treat_dr, no_of_out), device='cuda',
                        dtype=mcf_cuda.tdtype('float', precision))
    pot_y_m_ate = torch.empty_like(pot_y) if iate_m_ate_flag else None
    pot_y_var = torch.empty_like(pot_y) if iate_se_flag else None
    pot_y_m_ate_var = torch.empty_like(pot_y) if (
        iate_se_flag and iate_m_ate_flag) else None
    share_censored = torch.zeros(no_of_treat_dr,  device='cuda',
                                 dtype=mcf_cuda.tdtype('float', precision))
    l1_to_9 = [None] * n_x
    if gen_dic['d_type'] == 'continuous':
        d_values_dr = torch.from_numpy(ct_dic['d_values_dr_np'])
    else:
        d_values_dr = torch.tensor(
            gen_dic['d_values'], dtype=mcf_cuda.tdtype('int', precision))
    d_values_dr = d_values_dr.to('cuda')

    # The loop operates on GPUs thus avoiding lots of data transfer
    for idx in range(n_x):
        if int_dic['weight_as_sparse']:
            weights_idx = [weights_tx[idx] for weights_tx in weights]
        else:
            weights_idx = weights[idx]
        ret_all_i = iate_func1_for_cuda(
            idx, weights_idx, cl_dat, no_of_cluster, w_dat, w_ate,
            y_dat, no_of_out, n_y, ct_dic, int_dic, gen_dic, p_dic,
            iate_se_flag, se_boot_iate, iate_m_ate_flag, d_values_dr,
            precision=precision)
        (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
            share_censored) = assign_ret_all_i_cuda(
                pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
                share_censored, ret_all_i, n_x, idx)
        if int_dic['with_output'] and int_dic['verbose']:
            mcf_gp.share_completed(idx+1, n_x)
    (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, share_censored, l1_to_9
     ) = gpu_to_numpy(pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                      share_censored, l1_to_9)
    return (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
            share_censored)


def gpu_to_numpy(pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                 share_censored, l1_to_9):
    """Move tensor back to cpu and convert to numpy array."""
    def gpu_np(variable):
        if torch.is_tensor(variable):
            variable = variable.to('cpu')
            return variable.numpy()
        return variable

    pot_y_np = gpu_np(pot_y)
    pot_y_var_np = gpu_np(pot_y_var) if pot_y_var is not None else None
    pot_y_m_ate_np = gpu_np(pot_y_m_ate) if pot_y_m_ate is not None else None
    pot_y_m_ate_var_np = (gpu_np(pot_y_m_ate_var)
                          if pot_y_m_ate_var is not None else None)
    share_censored_np = gpu_np(share_censored)
    l1_to_9_list = [None] * len(l1_to_9)
    for idx, l1_to_9_idx in enumerate(l1_to_9):
        l1_to_9_list[idx] = [gpu_np(var) for var in l1_to_9_idx]
    return (pot_y_np, pot_y_var_np, pot_y_m_ate_np, pot_y_m_ate_var_np,
            share_censored_np, l1_to_9_list)


def assign_ret_all_i_cuda(pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                          l1_to_9, share_censored, ret_all_i, n_x, idx=None):
    """Use to avoid duplicate code."""
    if idx is None:
        idx = ret_all_i[0]
    pot_y[idx, :, :] = ret_all_i[1]
    if pot_y_m_ate is not None:
        pot_y_m_ate[idx, :, :] = ret_all_i[3]
    if pot_y_var is not None:
        pot_y_var[idx, :, :] = ret_all_i[2]
    if pot_y_m_ate_var is not None:
        pot_y_m_ate_var[idx, :, :] = ret_all_i[4]
    l1_to_9[idx] = ret_all_i[5]
    share_censored += ret_all_i[6] / n_x
    return (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
            share_censored)


def iate_func1_for_cuda(idx, weights_i, cl_dat, no_of_cluster, w_dat, w_ate,
                        y_dat, no_of_out, n_y, ct_dic, int_dic, gen_dic, p_dic,
                        iate_se_flag, se_boot_iate, iate_m_ate_flag,
                        d_values_dr, precision=32):
    """
    Compute function to be looped over observations for CUDA.

    Parameters
    ----------
    idx : Int. Counter.
    weights_i : List of int. Indices of non-zero weights.
                Alternative: Sparse tensor.
    cl_dat : Tensor. Cluster variable.
    no_of_cluster : Int. Number of clusters.
    w_dat : Tensor. Sampling weights.
    w_ate : Tensor. Weights for ATE.
    y_dat : Tensor. Outcome variable.
    no_of_out : Int. Number of outcomes.
    n_y : Int. Length of outcome data.
    ct_dic, int_dic, gen_dic, p_dic : Dict. Parameters.
    iate_se_flag : Boolean. Compute standard errors.
    se_boot_iate : Boolean. Compute bootstrap standard errors.
    iate_m_ate_flag : Boolean. Compute difference to average potential outcome.
    precision: Int. Precision for torch tensors.

    Returns
    -------
    idx: Int. Counter.
    pot_y_i: Tensor.
    pot_y_var_i: Tensor.
    pot_y_m_ate_i: Tensor (or None).
    pot_y_m_ate_var_i: Tensor (or None).
    l1_to_9: Tuple of lists.
    """
    def get_walli(w_index, n_y, w_i, w_i_unc):
        w_all_i = torch.zeros(n_y, device='cuda',
                              dtype=mcf_cuda.tdtype('float', precision))
        w_all_i[w_index] = w_i
        w_all_i_unc = torch.zeros_like(
            w_all_i, device='cuda', dtype=mcf_cuda.tdtype('float', precision))
        w_all_i_unc[w_index] = w_i_unc
        return w_all_i, w_all_i_unc

    if gen_dic['d_type'] == 'continuous':
        print('WARNING: IATE computation for continuous treatment on cuda not'
              'yet tested at all. Most likely it will break.')
        continuous = True
        # d_values_dr = torch.from_numpy(ct_dic['d_values_dr_np'])
        # d_values_dr = d_values_dr.to('cuda')
        no_of_treat = ct_dic['grid_w']
        # What data types are these variables --> tensor?
        i_w01, i_w10 = ct_dic['w_to_dr_int_w01'], ct_dic['w_to_dr_int_w10']
        index_full = ct_dic['w_to_dr_index_full']
        no_of_treat_dr = len(d_values_dr)
    else:
        # continuous, d_values_dr = False, gen_dic['d_values']
        continuous = False
        no_of_treat = no_of_treat_dr = gen_dic['no_of_treat']
    pot_y_i = torch.empty((no_of_treat_dr, no_of_out), device='cuda',
                          dtype=mcf_cuda.tdtype('float', precision))
    if iate_m_ate_flag:
        pot_y_m_ate_i = torch.empty_like(
            pot_y_i, device='cuda', dtype=mcf_cuda.tdtype('float', precision))
    else:
        pot_y_m_ate_i = None
    share_i = torch.zeros(no_of_treat_dr, device='cuda',
                          dtype=mcf_cuda.tdtype('float', precision))
    if iate_se_flag:
        pot_y_var_i = torch.empty_like(
            pot_y_i, device='cuda', dtype=mcf_cuda.tdtype('float', precision))
        if iate_m_ate_flag:
            pot_y_m_ate_var_i = torch.empty_like(pot_y_i, device='cuda')
        else:
            pot_y_m_ate_var_i = None
        cluster_std = p_dic['cluster_std']
    else:
        pot_y_var_i = pot_y_m_ate_var_i = None
        cluster_std = False
    if cluster_std:
        w_add = torch.zeros((no_of_treat_dr, no_of_cluster), device='cuda',
                            dtype=mcf_cuda.tdtype('float', precision))
    else:
        w_add = torch.zeros((no_of_treat_dr, n_y), device='cuda',
                            dtype=mcf_cuda.tdtype('float', precision))
    w_add_unc = torch.zeros((no_of_treat_dr, n_y), device='cuda',
                            dtype=mcf_cuda.tdtype('float', precision))
    # Loop over treatment
    for t_idx in range(no_of_treat):
        extra_weight_p1 = continuous and t_idx < no_of_treat - 1
        if int_dic['weight_as_sparse']:
            # w_index = weights_i[t_idx].indices()
            # w_i_t = weights_i[t_idx].values()
            coalesced_tensor = weights_i[t_idx].coalesce()
            w_index = coalesced_tensor.indices().reshape(-1)
            w_i_t = coalesced_tensor.values()
        else:
            w_index = weights_i[t_idx][0]    # Indices of non-zero weights
            w_i_t = weights_i[t_idx][1].clone()
        if extra_weight_p1:
            if int_dic['weight_as_sparse']:
                coalesced_tensor_p1 = weights_i[t_idx+1].coalesce()
                w_index_p1 = coalesced_tensor_p1.indices().reshape(-1)
            else:
                w_index_p1 = weights_i[t_idx+1][0]
            w_index_both = torch.unique(torch.cat((w_index, w_index_p1)))
            w_i = torch.zeros(n_y, device='cuda',
                              dtype=mcf_cuda.tdtype('float', precision))
            w_i[w_index] = w_i_t
            w_i_p1 = np.zeros_like(w_i)
            if int_dic['weight_as_sparse']:
                w_i_p1[w_index_p1] = coalesced_tensor_p1.values()
            else:
                w_i_p1[w_index_p1] = weights_i[t_idx+1][1].clone()
            w_i = w_i[w_index_both]
            w_i_p1 = w_i_p1[w_index_both]
        else:
            w_index_both = w_index
            w_i = w_i_t
        if gen_dic['weighted']:
            w_t = w_dat[w_index].reshape(-1)
            w_i = w_i * w_t
            if extra_weight_p1:
                w_t_p1 = w_dat[w_index_p1].reshape(-1)
                w_i_p1 = w_i_p1 * w_t_p1
        else:
            w_t = None
            if extra_weight_p1:
                w_t_p1 = None
        w_i_sum = torch.sum(w_i)
        if (not (1-1e-10) < w_i_sum < (1+1e-10)) and not continuous:
            w_i = w_i / w_i_sum
        w_i_unc = w_i.clone()
        if p_dic['max_weight_share'] < 1 and not continuous:
            w_i, _, share_i[t_idx] = mcf_gp.bound_norm_weights_cuda(
                w_i, p_dic['max_weight_share'])
        if extra_weight_p1:
            w_i_unc_p1 = w_i_p1.clone()
        if cluster_std:
            w_all_i, w_all_i_unc = get_walli(w_index, n_y, w_i, w_i_unc)
            cl_i = cl_dat[w_index]
            if extra_weight_p1:
                w_all_i_p1, w_all_i_unc_p1 = get_walli(w_index_p1, n_y, w_i_p1,
                                                       w_i_unc_p1)
                cl_i_both = cl_dat[w_index_both]
        else:
            cl_i = cl_i_both = None
        for o_idx in range(no_of_out):
            if continuous:
                y_dat_cont = y_dat[w_index_both, o_idx]
                for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                    if extra_weight_p1:
                        w_i_cont = w10 * w_i + w01 * w_i_p1
                        w_i_unc_cont = w10 * w_i_unc + w01 * w_i_unc_p1
                        w_t_cont = (None if w_t is None
                                    else w10 * w_t + w01 * w_t_p1)
                        cl_i_cont = cl_i_both
                    else:
                        w_i_cont, w_t_cont, cl_i_cont = w_i, w_t, cl_i_both
                        w_i_unc_cont = w_i_unc
                    w_i_cont = w_i_cont / torch.sum(w_i_cont)
                    if w_t_cont is not None:
                        w_t_cont = w_t_cont / torch.sum(w_t_cont)
                    w_i_unc_cont = w_i_unc_cont / torch.sum(w_i_unc_cont)
                    if p_dic['max_weight_share'] < 1:
                        (w_i_cont, _, share_cont
                         ) = mcf_gp.bound_norm_weights_cuda(
                             w_i_cont, p_dic['max_weight_share'])
                        if i == 0:
                            share_i[t_idx] = share_cont

                    ret = mcf_est_cuda.weight_var_cuda(
                        w_i_cont, y_dat_cont, cl_i_cont, gen_dic, p_dic,
                        weights=w_t_cont, se_yes=iate_se_flag,
                        bootstrap=se_boot_iate, keep_all=int_dic['keep_w0'],
                        precision=precision)
                    ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                    pot_y_i[ti_idx, o_idx] = ret[0]
                    if iate_se_flag:
                        pot_y_var_i[ti_idx, o_idx] = ret[1]
                    if cluster_std:
                        w_cont = (w10 * w_all_i + w01 * w_all_i_p1
                                  if extra_weight_p1 else w_all_i)
                        ret2 = mcf_est_cuda.aggregate_cluster_pos_w_cuda(
                            cl_dat, w_cont, y_dat[:, o_idx], sweights=w_dat,
                            precision=precision)
                        if o_idx == 0:
                            w_add[ti_idx, :] = ret2[0].clone()
                            if w_ate is None:
                                w_diff = (w10 * w_all_i_unc
                                          + w01 * w_all_i_unc_p1)
                            else:
                                if extra_weight_p1:
                                    w_ate_cont = (w10 * w_ate[t_idx, :] +
                                                  w01 * w_ate[t_idx+1, :])
                                    w_ate_cont /= torch.sum(w_ate_cont)
                                    w_diff = w_all_i_unc - w_ate_cont
                                else:
                                    w_diff = w_all_i_unc - w_ate[t_idx, :]
                        if iate_m_ate_flag:
                            ret = mcf_est_cuda.weight_var_cuda(
                                w_diff, y_dat[:, o_idx], cl_dat, gen_dic,
                                p_dic, normalize=False, weights=w_dat,
                                bootstrap=se_boot_iate, se_yes=iate_se_flag,
                                keep_all=int_dic['keep_w0'],
                                precision=precision)
                    else:
                        if o_idx == 0:
                            w_add[ti_idx, w_index_both] = ret[2]
                            w_i_unc_sum = torch.sum(w_i_unc_cont)
                            if not (1-1e-10) < w_i_unc_sum < (1+1e-10):
                                w_add_unc[ti_idx, w_index_both] = (
                                    w_i_unc_cont / w_i_unc_sum)
                            else:
                                w_add_unc[ti_idx, w_index_both] = w_i_unc_cont
                            if w_ate is None:
                                w_diff = w_add_unc[ti_idx, :]
                            else:
                                if extra_weight_p1:
                                    w_ate_cont = (w10 * w_ate[t_idx, :] +
                                                  w01 * w_ate[t_idx+1, :])
                                    w_ate_cont /= torch.sum(w_ate_cont)
                                    w_diff = w_add_unc[ti_idx, :] - w_ate_cont
                                else:
                                    w_diff = (w_add_unc[ti_idx, :]
                                              - w_ate[t_idx, :])
                        if iate_m_ate_flag:
                            ret = mcf_est_cuda.weight_var_cuda(
                                w_diff, y_dat[:, o_idx], None, gen_dic, p_dic,
                                normalize=False, weights=w_dat,
                                bootstrap=se_boot_iate, se_yes=iate_se_flag,
                                keep_all=int_dic['keep_w0'],
                                precision=precision)
                    if iate_m_ate_flag:
                        pot_y_m_ate_i[ti_idx, o_idx] = ret[0]
                    if iate_se_flag and iate_m_ate_flag:
                        pot_y_m_ate_var_i[ti_idx, o_idx] = ret[1]
                    if not extra_weight_p1:
                        break
            else:  # discrete treatment
                ret = mcf_est_cuda.weight_var_cuda(
                    w_i, y_dat[w_index, o_idx], cl_i, gen_dic, p_dic,
                    weights=w_t, se_yes=iate_se_flag, bootstrap=se_boot_iate,
                    keep_all=int_dic['keep_w0'], precision=precision)
                pot_y_i[t_idx, o_idx] = ret[0]
                if iate_se_flag:
                    pot_y_var_i[t_idx, o_idx] = ret[1]
                if cluster_std:
                    ret2 = mcf_est_cuda.aggregate_cluster_pos_w_cuda(
                        cl_dat, w_all_i, y_dat[:, o_idx], sweights=w_dat,
                        precision=precision)
                    if o_idx == 0:
                        w_add[t_idx, :] = ret2[0].clone()
                        if w_ate is None:
                            w_diff = w_all_i_unc  # Dummy if no w_ate
                        else:
                            w_diff = w_all_i_unc - w_ate[t_idx, :]
                    if iate_m_ate_flag:
                        ret = mcf_est_cuda.weight_var_cuda(
                            w_diff, y_dat[:, o_idx], cl_dat, gen_dic, p_dic,
                            normalize=False, weights=w_dat,
                            bootstrap=se_boot_iate,
                            se_yes=iate_se_flag, keep_all=int_dic['keep_w0'],
                            precision=precision)
                else:
                    if o_idx == 0:
                        w_add[t_idx, w_index] = ret[2]
                        w_i_unc_sum = torch.sum(w_i_unc)
                        if not (1-1e-10) < w_i_unc_sum < (1+1e-10):
                            w_add_unc[t_idx, w_index] = w_i_unc / w_i_unc_sum
                        else:
                            w_add_unc[t_idx, w_index] = w_i_unc
                        if w_ate is None:
                            w_diff = w_add_unc[t_idx, :]
                        else:
                            w_diff = w_add_unc[t_idx, :] - w_ate[t_idx, :]
                    if iate_m_ate_flag:
                        ret = mcf_est_cuda.weight_var_cuda(
                            w_diff, y_dat[:, o_idx], None, gen_dic, p_dic,
                            normalize=False, weights=w_dat,
                            bootstrap=se_boot_iate,
                            se_yes=iate_se_flag, keep_all=int_dic['keep_w0'],
                            precision=precision)
                if iate_m_ate_flag:
                    pot_y_m_ate_i[t_idx, o_idx] = ret[0]
                if iate_m_ate_flag and iate_se_flag:
                    pot_y_m_ate_var_i[t_idx, o_idx] = ret[1]
    l1_to_9 = mcf_est_cuda.analyse_weights_cuda(
        w_add, None, gen_dic, p_dic, ate=False, continuous=continuous,
        no_of_treat_cont=no_of_treat_dr, d_values_cont=d_values_dr,
        precision=precision)
    return (idx, pot_y_i, pot_y_var_i, pot_y_m_ate_i, pot_y_m_ate_var_i,
            l1_to_9, share_i)


def tensor_to_gpu(weights_l_np, cl_dat_np, w_dat_np, w_ate_np, y_dat_np,
                  weight_as_sparse, iate_m_ate_flag, precision=32):
    """Convert to tensors and move to gpu."""
    if weight_as_sparse:
        weights = [scipy_sparse_to_torch_sparse(weight, precision)
                   for weight in weights_l_np]
    else:
        weights = torch.from_numpy(weights_l_np)
        weights = weights.to(mcf_cuda.tdtype('float', precision))
        weights = weights.to('cuda')
    if cl_dat_np is not None:
        cl_dat = torch.from_numpy(cl_dat_np)
        cl_dat = cl_dat.to(mcf_cuda.tdtype('float', precision))
        cl_dat = cl_dat.to('cuda')
    else:
        cl_dat = None
    if w_dat_np is not None:
        w_dat = torch.from_numpy(w_dat_np)
        w_dat = w_dat.to(mcf_cuda.tdtype('float', precision))
        w_dat = w_dat.to('cuda')
    else:
        w_dat = None
    y_dat = torch.from_numpy(y_dat_np)
    y_dat = y_dat.to(mcf_cuda.tdtype('float', precision))
    y_dat = y_dat.to('cuda')
    if iate_m_ate_flag:
        w_ate = torch.from_numpy(w_ate_np)
        w_ate = w_ate.to(mcf_cuda.tdtype('float', precision))
        w_ate = w_ate.to('cuda')
    else:
        w_ate = None

    return weights, cl_dat, w_dat, w_ate, y_dat


def scipy_sparse_to_torch_sparse(scipy_sparse_matrix, precision=32):
    """Convert a scipy csr sparse matrich to a sparse tensor."""
    # It would be more efficient to avoid the conversion to coo, but this seems
    # to currently impossible
    coo_matrix = scipy_sparse_matrix.tocoo()
    sparse_tensor = torch.sparse_coo_tensor(
        torch.tensor([coo_matrix.row.tolist(), coo_matrix.col.tolist()]),
        torch.tensor(coo_matrix.data),
        size=coo_matrix.shape, device='cuda',
        dtype=mcf_cuda.tdtype('float', precision)
        )
    # CSR tensor, which would be more efficient seem not yet to work properly
    # sparse_tensor = torch.sparse_csr_tensor(
    #     coo_matrix.row, coo_matrix.col, coo_matrix.data,
    #     size=coo_matrix.shape, device='cuda'
    #     )
    return sparse_tensor


def max_batch_size(n_x, weights, max_share_weights, weight_as_sparse, gen_dic):
    """Find maximum batchsize for CUDA computations."""
    # in gigabytes
    in_gb = 1024 ** -3
    total_memory = torch.cuda.get_device_properties('cuda').total_memory * in_gb
    allocated_memory = torch.cuda.memory_allocated(device='cuda') * in_gb
    cached_memory = torch.cuda.memory_reserved(device='cuda') * in_gb
    free_memory = total_memory - (allocated_memory - cached_memory)

    txt = '\n' + f"Total GPU memory:        {total_memory:.3f} GB"
    txt += '\n' + f"Allocated GPU memory:    {allocated_memory:.3f} GB"
    txt += '\n' + f"Cached GPU memory:       {cached_memory:.3f} GB"
    txt += '\n' + f"Free GPU memory:         {free_memory:.3f} GB"

    weight_memory = mcf_sys.print_size_weight_matrix(
        weights, weight_as_sparse, gen_dic['no_of_treat'], no_text=True) * in_gb
    txt += '\n' + f"GPU memory needed for weight matrix: {weight_memory:.3} GB"
    no_batches = ceil(weight_memory / (free_memory * max_share_weights))
    batch_max_size = ceil(n_x / no_batches)
    txt += '\n' + f'# of batches: {no_batches}'
    txt += ('\n' + "GPU memory needed for one batch of weight matrix: "
            f"{weight_memory/no_batches:.2f} GB")
    ps.print_mcf(gen_dic, txt, summary=True)
    return batch_max_size
