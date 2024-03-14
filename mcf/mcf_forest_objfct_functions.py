"""
Contains functions to compute objective functions of mcf.

Created on Mon Oct 30 13:09:25 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from numba import njit
import numpy as np
import torch

from mcf import mcf_cuda_functions as mcf_c


def mcf_mse(y_dat, y_nn, d_dat, w_dat, n_obs, mtot, no_of_treat, treat_values,
            w_yes=False, splitting=False, cuda=False):
    """Compute average mse for the data passed. Based on different methods.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_dat : Numpy Nx1 vector. Treatment.
    w_dat : Numpy Nx1 vector. Weights (or 0)
    n : INT. Leaf size.
    mtot : INT. Method.
    no_of_treat : INT. Number of treated.
    treat_values : List of INT. Treatment values.
    w_yes : Boolean. Weighted estimation.
    splitting : Boolean. Default is False.
    cuda : Boolean. Use cuda if True.

    Returns
    -------
    mse : Mean squared error (average not acccount of number of obs).
    treat_share: 1D Numpy array. Treatment shares.

    """
    min_obs_for_cuda = 10000000000
    # CUDA is currently not used here (ineffective when used from many
    # processes simultanously. Too much traffic.)
    if cuda and len(y_dat) > min_obs_for_cuda:
        mse_mce, treat_shares, no_of_obs_by_treat = mcf_mse_cuda(
            y_dat, y_nn, d_dat, w_dat, n_obs, mtot, no_of_treat,
            treat_values, w_yes, splitting)
    else:
        if w_yes or mtot in (2, 3):
            mse_mce, treat_shares, no_of_obs_by_treat = mcf_mse_not_numba(
                y_dat, y_nn, d_dat, w_dat, n_obs, mtot, no_of_treat,
                treat_values, w_yes, splitting)
        else:
            mse_mce, treat_shares, no_of_obs_by_treat = mcf_mse_numba(
                y_dat, y_nn, d_dat, n_obs, mtot, no_of_treat,
                np.array(treat_values, dtype=np.int8))
    return mse_mce, treat_shares, no_of_obs_by_treat


def mcf_mse_cuda(y_dat_np, y_nn_np, d_dat_np, w_dat_np, n_obs, mtot,
                 no_of_treat, treat_values_list, w_yes, splitting=False):
    """Compute average mse for the data passed (cuda).

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_all : Numpy Nx1 vector. Treatment.
    w_dat : Numpy Nx1 vector. Weights (or 0)
    n : INT. Leaf size.
    mtot : INT. Method.
    no_of_treat : INT. Number of treated.
    treat_values : List of INT. Treatment values.
    w_yes: Boolean. Weighted estimation.
    splitting: Boolean. Default is False.

    Returns
    -------
    mse : Mean squared error (average not acccount of number of obs).
    treat_share: 1D Numpy array. Treatment shares.

    """
    prec = 32
    y_dat, y_nn, d_dat, w_dat, treat_values = tensors_to_gpu(
        y_dat_np, y_nn_np, d_dat_np, w_dat_np, treat_values_list, w_yes)

    treat_shares = (torch.empty(no_of_treat, device='cuda',
                                dtype=mcf_c.tdtype('float', prec))
                    if mtot in (1, 4) else 0)
    mse_mce = torch.zeros((no_of_treat, no_of_treat), device='cuda',
                          dtype=mcf_c.tdtype('float', 32))
    no_of_obs_by_treat = torch.zeros(no_of_treat, device='cuda',
                                     dtype=mcf_c.tdtype('int', 32))
    for m_idx in range(no_of_treat):
        d_m = d_dat == treat_values[m_idx]   # d_m is Boolean
        n_m = len(y_dat[d_m])
        no_of_obs_by_treat[m_idx] = n_m
        if w_yes:
            w_m = w_dat[d_m]
            y_m_mean = mean_t_w(y_dat[d_m], w_m, dim=0)
            mse_m = mean_t_w((y_dat[d_m] - y_m_mean)**2, w_m, dim=0)
        else:
            y_m_mean = torch.mean(y_dat[d_m], dim=0)
            mse_m = torch.sum(y_dat[d_m] ** 2) / n_m - (y_m_mean**2)
        if mtot in (1, 4):
            treat_shares[m_idx] = n_m / n_obs
        if mtot in (1, 3, 4):
            mse_mce[m_idx, m_idx] = mse_m
        if mtot != 3:
            mce_ml = 0
            for v_idx in range(m_idx + 1, no_of_treat):
                if mtot == 2:  # Variance of effects mtot = 2
                    d_l = d_dat == treat_values[v_idx]   # d_l is Boolean
                    if w_yes:
                        y_l_mean = mean_t_w(y_dat[d_l], w_dat[d_l], dim=0)
                    else:
                        y_l_mean = torch.mean(y_dat[d_l], dim=0)
                    mce_ml = (y_m_mean - y_l_mean)**2
                else:
                    d_ml = (d_dat == treat_values[v_idx]) | (
                        d_dat == treat_values[m_idx])
                    d_ml = d_ml[:, 0]
                    y_nn_m = y_nn[d_ml, m_idx]
                    y_nn_l = y_nn[d_ml, v_idx]
                    if w_yes:
                        w_ml = w_dat[d_ml].reshape(-1)
                        if splitting and (no_of_treat == 2):
                            mce_ml = ((mean_t_w(y_nn_m, w_ml, dim=0))
                                      * (mean_t_w(y_nn_l, w_ml, dim=0)) * (-1))
                        else:
                            mce_ml = mean_t_w(
                                (y_nn_m - mean_t_w(y_nn_m, w_ml, dim=0))
                                * (y_nn_l - mean_t_w(y_nn_l, w_ml, dim=0)),
                                w_ml, dim=0)
                    else:
                        aaa = (torch.mean(y_nn_m, dim=0)
                               * torch.mean(y_nn_l, dim=0))
                        bbb = torch.sum(y_nn_m * y_nn_l) / len(y_nn_m)
                        mce_ml = bbb - aaa
                mse_mce[m_idx, v_idx] = mce_ml

    mse_mce_np, treat_shares_np, no_of_obs_by_treat_np = tensors_to_np_to_cpu(
        mse_mce, treat_shares, no_of_obs_by_treat)

    return mse_mce_np, treat_shares_np, no_of_obs_by_treat_np


def mean_t_w(tensor, weights, dim=None):
    """
    Compute the weighted mean of a tensor along a specified dimension.

    Parameters
    ----------
    tensor : torch.Tensor. Input tensor.
    weights : torch.Tensor: Tensor of weights.
    dim : int, optional: Dimension along which to compute the mean.

    Returns
    -------
    weighted_mean : torch.Tensor: Weighted mean.

    """
    weighted_sum = torch.sum(tensor * weights, dim=dim)
    sum_of_weights = torch.sum(weights, dim=dim)
    weighted_mean = weighted_sum / sum_of_weights
    return weighted_mean


def tensors_to_np_to_cpu(mse_mce, treat_shares, no_of_obs_by_treat):
    """Copy tensors to cpu and convert to numpy arrays."""
    mse_mce = mse_mce.to('cpu')
    treat_shares = treat_shares.to('cpu')
    no_of_obs_by_treat = no_of_obs_by_treat.to('cpu')

    mse_mce_np = mse_mce.numpy()
    treat_shares_np = treat_shares.numpy()
    no_of_obs_by_treat_np = no_of_obs_by_treat.numpy()

    return mse_mce_np, treat_shares_np, no_of_obs_by_treat_np


def tensors_to_gpu(y_dat_np, y_nn_np, d_dat_np, w_dat_np, treat_values_list,
                   w_yes):
    """Get the tensors and copy them to gpu."""
    y_dat = torch.from_numpy(y_dat_np)
    y_nn = torch.from_numpy(y_nn_np)
    d_dat = torch.from_numpy(d_dat_np)
    w_dat = torch.from_numpy(w_dat_np) if w_yes else None
    treat_values = torch.tensor(treat_values_list)

    y_dat = y_dat.to("cuda")
    y_nn = y_nn.to("cuda")
    d_dat = d_dat.to("cuda")
    if w_yes:
        w_dat = w_dat.to("cuda")
    treat_values = treat_values.to("cuda")

    return y_dat, y_nn, d_dat, w_dat, treat_values


def mcf_mse_not_numba(y_dat, y_nn, d_dat, w_dat, n_obs, mtot, no_of_treat,
                      treat_values, w_yes, splitting=False):
    """Compute average mse for the data passed. Based on different methods.

    CURRENTLY ONLY USED FOR WEIGHTED.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_all : Numpy Nx1 vector. Treatment.
    w_dat : Numpy Nx1 vector. Weights (or 0)
    n : INT. Leaf size.
    mtot : INT. Method.
    no_of_treat : INT. Number of treated.
    treat_values : List of INT. Treatment values.
    w_yes: Boolean. Weighted estimation.
    splitting: Boolean. Default is False.

    Returns
    -------
    mse : Mean squared error (average not acccount of number of obs).
    treat_share: 1D Numpy array. Treatment shares.
    no_of_obs_by_treat : 1D Numpy array. Observations by treatment.

    """
    treat_shares = np.empty(no_of_treat) if mtot in (1, 4) else 0
    mse_mce = np.zeros((no_of_treat, no_of_treat))
    no_of_obs_by_treat = np.zeros(no_of_treat)
    for m_idx in range(no_of_treat):
        d_m = d_dat == treat_values[m_idx]   # d_m is Boolean
        n_m = len(y_dat[d_m])
        no_of_obs_by_treat[m_idx] = n_m
        if w_yes:
            w_m = w_dat[d_m]
            y_m_mean = np.average(y_dat[d_m], weights=w_m, axis=0)
            mse_m = np.average(np.square(y_dat[d_m] - y_m_mean),
                               weights=w_m, axis=0)
        else:
            y_m_mean = np.average(y_dat[d_m], axis=0)
            mse_m = np.dot(y_dat[d_m], y_dat[d_m]) / n_m - (y_m_mean**2)
        if mtot in (1, 4):
            treat_shares[m_idx] = n_m / n_obs
        if mtot in (1, 3, 4):
            mse_mce[m_idx, m_idx] = mse_m
        if mtot != 3:
            mce_ml = 0
            for v_idx in range(m_idx + 1, no_of_treat):
                if mtot == 2:  # Variance of effects mtot = 2
                    d_l = d_dat == treat_values[v_idx]   # d_l is Boolean
                    if w_yes:
                        y_l_mean = np.average(y_dat[d_l], weights=w_dat[d_l],
                                              axis=0)
                    else:
                        y_l_mean = np.average(y_dat[d_l], axis=0)
                    mce_ml = (y_m_mean - y_l_mean)**2
                else:
                    d_ml = (d_dat == treat_values[v_idx]) | (
                        d_dat == treat_values[m_idx])
                    d_ml = d_ml[:, 0]
                    y_nn_m, y_nn_l = y_nn[d_ml, m_idx], y_nn[d_ml, v_idx]
                    if w_yes:
                        w_ml = w_dat[d_ml].reshape(-1)
                        if splitting and (no_of_treat == 2):
                            mce_ml = ((np.average(y_nn_m, weights=w_ml,
                                                  axis=0)) *
                                      (np.average(y_nn_l, weights=w_ml,
                                                  axis=0)) * (-1))
                        else:
                            mce_ml = np.average(
                                (y_nn_m - np.average(y_nn_m, weights=w_ml,
                                                     axis=0)) *
                                (y_nn_l - np.average(y_nn_l, weights=w_ml,
                                                     axis=0)),
                                weights=w_ml, axis=0)
                    else:
                        aaa = np.average(y_nn_m, axis=0) * np.average(y_nn_l,
                                                                      axis=0)
                        bbb = np.dot(y_nn_m, y_nn_l) / len(y_nn_m)
                        mce_ml = bbb - aaa
                mse_mce[m_idx, v_idx] = mce_ml
    return mse_mce, treat_shares, no_of_obs_by_treat


@njit
def mcf_mse_numba(y_dat, y_nn, d_dat, n_obs, mtot, no_of_treat, treat_values):
    """Compute average mse for the data passed. Based on different methods.

       WEIGHTED VERSION DOES NOT YET WORK. TRY with next Numba version.
       Need to change list format soon.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_dat : Numpy Nx1 vector. Treatment.
    d_bin_dat : Numpy Nx1 vector. Treatment larger 0.
    n : INT. Leaf size.
    mtot : INT. Method.
    no_of_treat : INT. Number of treated.
    treat_values : 1D Numpy array of INT. Treatment values.
    cont. Boolean. Continuous treatment.

    Returns
    -------
    mse : Mean squared error (average not acccount of number of obs).
    treat_share: 1D Numpy array. Treatment shares.
    no_of_obs_by_treat : 1D Numpy array. Observations by treatment.
    """
    obs = len(y_dat)
    treat_shares = np.zeros(no_of_treat) if mtot in (1, 3, 4) else np.zeros(1)
    mse_mce = np.zeros((no_of_treat, no_of_treat))
    no_of_obs_by_treat = np.zeros(no_of_treat)
    for m_idx in range(no_of_treat):
        d_m = d_dat == treat_values[m_idx]   # d_m is Boolean
        n_m = np.sum(d_m)
        no_of_obs_by_treat[m_idx] = n_m
        y_m = np.empty(n_m)
        j = 0
        for i in range(obs):
            if d_m[i]:
                y_m[j] = y_dat[i, 0]
                j += 1
        y_m_mean = np.sum(y_m) / n_m
        mse_m = np.dot(y_m, y_m) / n_m - (y_m_mean**2)
        if mtot in (1, 3, 4):
            treat_shares[m_idx] = n_m / n_obs
            mse_mce[m_idx, m_idx] = mse_m
        if mtot != 3:
            mce_ml = 0
            for v_idx in range(m_idx + 1, no_of_treat):
                d_l = d_dat == treat_values[v_idx]   # d_l is Boolean
                n_l = np.sum(d_l)
                if mtot == 2:  # Variance of effects mtot = 2
                    y_l = np.empty(n_l)
                    j = 0
                    for i in range(obs):
                        if d_l[i]:
                            y_l[j] = y_dat[i, 0]
                            j += 1
                    y_l_mean = np.sum(y_l) / n_l
                    mce_ml = (y_m_mean - y_l_mean)**2
                elif mtot in (1, 4):
                    d_ml = (d_dat == treat_values[v_idx]) | (
                        d_dat == treat_values[m_idx])
                    n_ml = np.sum(d_ml)
                    y_nn_l = np.empty(n_ml)
                    y_nn_m = np.empty_like(y_nn_l)
                    j = 0
                    for i in range(obs):
                        if d_ml[i]:
                            y_nn_l[j] = y_nn[i, v_idx]
                            y_nn_m[j] = y_nn[i, m_idx]
                            j += 1
                    aaa = np.sum(y_nn_m) / n_ml * np.sum(y_nn_l) / n_ml
                    bbb = np.dot(y_nn_m, y_nn_l) / n_ml
                    mce_ml = bbb - aaa
                mse_mce[m_idx, v_idx] = mce_ml
    return mse_mce, treat_shares, no_of_obs_by_treat


@njit
def compute_mse_mce(mse_mce, mtot, no_of_treat):
    """Sum up MSE parts for use in splitting rule and else."""
    if no_of_treat > 4:
        if mtot in (1, 4):
            mse = no_of_treat * np.trace(mse_mce) - mse_mce.sum()
        elif mtot == 2:
            mse = 2 * np.trace(mse_mce) - mse_mce.sum()
        elif mtot == 3:
            mse = np.trace(mse_mce)
    else:
        mse = mce = 0
        for m_idx in range(no_of_treat):
            if mtot in (1, 4):
                mse_a = (no_of_treat - 1) * mse_mce[m_idx, m_idx]
            else:
                mse_a = mse_mce[m_idx, m_idx]
            mse += mse_a
            if mtot != 3:
                for v_idx in range(m_idx+1, no_of_treat):
                    mce += mse_mce[m_idx, v_idx]
        mse -= 2 * mce
    return mse


@njit
def mcf_penalty(shares_l, shares_r):
    """Generate the (unscaled) penalty.

    Parameters
    ----------
    shares_l : Numpy array. Treatment shares left.
    shares_r : Numpy array. Treatment shares right.

    Returns
    -------
    penalty : Numpy float. Penalty of split.

    """
    diff = (shares_l - shares_r) ** 2
    penalty = 1 - (np.sum(diff) / len(shares_l))
    return penalty


@njit
def get_avg_mse_mce(mse_mce, obs_by_treat, mtot, no_of_treat):
    """Bring MSE_MCE matrix in average form."""
    mse_mce_avg = mse_mce.copy()
    for m_idx in range(no_of_treat):
        if obs_by_treat[m_idx] > 0:
            mse_mce_avg[m_idx, m_idx] = (mse_mce[m_idx, m_idx]
                                         / obs_by_treat[m_idx])
        if mtot != 3:
            for v_idx in range(m_idx+1, no_of_treat):
                if obs_by_treat[m_idx] + obs_by_treat[v_idx] > 0:
                    mse_mce_avg[m_idx, v_idx] = mse_mce[m_idx, v_idx] / (
                        obs_by_treat[m_idx] + obs_by_treat[v_idx])
    return mse_mce_avg


@njit
def add_rescale_mse_mce(mse_mce, obs_by_treat, mtot, no_of_treat,
                        mse_mce_add_to, obs_by_treat_add_to):
    """Rescale MSE_MCE matrix and update observation count."""
    mse_mce_sc = np.zeros((no_of_treat, no_of_treat))
    obs_by_treat_new = obs_by_treat + obs_by_treat_add_to
    for m_idx in range(no_of_treat):
        mse_mce_sc[m_idx, m_idx] = mse_mce[m_idx, m_idx] * obs_by_treat[m_idx]
        if mtot != 3:
            for v_idx in range(m_idx+1, no_of_treat):
                mse_mce_sc[m_idx, v_idx] = mse_mce[m_idx, v_idx] * (
                    obs_by_treat[m_idx] + obs_by_treat[v_idx])
    mse_mce_new = mse_mce_add_to + mse_mce_sc
    return mse_mce_new, obs_by_treat_new


@njit
def add_mse_mce_split(mse_mce_l, mse_mce_r, obs_by_treat_l, obs_by_treat_r,
                      mtot, no_of_treat):
    """Sum up MSE parts of use in splitting rule."""
    mse_mce = np.zeros((no_of_treat, no_of_treat))
    obs_by_treat = obs_by_treat_l + obs_by_treat_r
    for m_idx in range(no_of_treat):
        if obs_by_treat[m_idx] > 0:
            mse_mce[m_idx, m_idx] = (
                (mse_mce_l[m_idx, m_idx] * obs_by_treat_l[m_idx]
                 + mse_mce_r[m_idx, m_idx] * obs_by_treat_r[m_idx])
                / obs_by_treat[m_idx])
        if mtot != 3:
            for v_idx in range(m_idx+1, no_of_treat):
                n_ml_l = obs_by_treat_l[m_idx] + obs_by_treat_l[v_idx]
                n_ml_r = obs_by_treat_r[m_idx] + obs_by_treat_r[v_idx]
                if n_ml_l + n_ml_r > 0:
                    mse_mce[m_idx, v_idx] = (
                        (mse_mce_l[m_idx, v_idx] * n_ml_l
                         + mse_mce_r[m_idx, v_idx] * n_ml_r)
                        / (n_ml_l + n_ml_r))
    return mse_mce
