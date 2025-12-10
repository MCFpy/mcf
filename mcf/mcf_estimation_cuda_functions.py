"""
Contains functions for the estimation of various effect.

Created on Mon Jun 19 17:50:33 2023.

# -*- coding: utf-8 -*-
@author: MLechner
"""
from math import sqrt, pi
from typing import Any

try:
    import torch  # type: ignore[import]
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]

from mcf import mcf_cuda_functions as mcf_c
from mcf import mcf_print_stats_functions as mcf_ps


def weight_var_cuda(w0_dat: torch.Tensor,
                    y0_dat: torch.Tensor,
                    cl_dat: torch.Tensor | None,
                    gen_cfg: Any,
                    p_cfg: Any,
                    normalize: bool = True,
                    w_for_diff: torch.Tensor | None = None,
                    weights: torch.Tensor | None = None,
                    bootstrap: int = 0,
                    keep_some_0: bool = False,
                    se_yes: bool = True,
                    keep_all: bool = False,
                    precision: int = 32,
                    zero_tol: float = 1e-15,
                    sum_tol: float = 1e-12,
                    ) -> tuple[torch.Tensor,
                               torch.Tensor | None,
                               torch.Tensor]:
    '''Generate the weight-based variance (CUDA version).'''

    device = w0_dat.device
    dtype = mcf_c.tdtype('float', precision)

    w_dat = w0_dat.to(device=device, dtype=dtype).clone()
    y_dat = y0_dat.to(device=device, dtype=dtype).clone()
    cl = cl_dat.to(device=device) if cl_dat is not None else None

    # Cluster aggregation for non-bootstrap SEs
    if p_cfg.cluster_std and (cl is not None) and (bootstrap < 1):
        if not gen_cfg.weighted:
            weights = None
        w_dat, y_dat, _, _ = aggregate_cluster_pos_w_cuda(
            cl, w_dat, y_dat,
            norma=normalize, sweights=weights, precision=precision,
            zero_tol=zero_tol, sum_tol=sum_tol,
            )
        if w_for_diff is not None:
            w_dat = w_dat - w_for_diff.to(device=device, dtype=dtype)

    # If no SEs, no bootstrap / extra zero handling
    if not p_cfg.iate_se:
        keep_some_0 = False
        bootstrap = 0

    # Optional normalization of weights
    if normalize:
        sum_w_dat = w_dat.sum().abs()
        sum_w_val = float(sum_w_dat)
        close_to_zero = sum_w_val < zero_tol
        close_to_one = abs(sum_w_val - 1.0) < sum_tol
        if not (close_to_zero or close_to_one):
            w_dat = w_dat / sum_w_dat

    w_ret = w_dat.clone()

    # Mask of observations to keep
    if keep_all:
        w_pos = torch.ones_like(w_dat, dtype=torch.bool, device=device)
    else:
        # use non-zero only to speed up
        w_pos = w_dat.abs() > zero_tol

    only_copy = bool(torch.all(w_pos))

    # Optionally keep a random subset of zero-weight obs
    if keep_some_0 and not only_copy:
        sum_pos = int(w_pos.sum().item())
        obs_all = int(w_dat.numel())
        sum_0 = obs_all - sum_pos
        zeros_to_keep = int(round(0.05 * obs_all))
        zeros_to_switch = sum_0 - zeros_to_keep

        if zeros_to_switch <= 2:
            only_copy = True
        else:
            zero_idx = (~w_pos).nonzero(as_tuple=False).view(-1)
            zeros_to_switch = min(zeros_to_switch, int(zero_idx.numel()))
            if zeros_to_switch > 0:
                g_cuda = torch.Generator(device=device)
                g_cuda.manual_seed(123345)
                perm = torch.randperm(zero_idx.numel(),
                                      generator=g_cuda,
                                      device=device,)[:zeros_to_switch]
                idx_to_true = zero_idx[perm]
                w_pos[idx_to_true] = True

    # Apply mask
    if only_copy:
        w_dat2 = w_dat.clone()
        y_use = y_dat
        cl_use = cl
    else:
        w_dat2 = w_dat[w_pos]
        y_use = y_dat[w_pos]
        cl_use = cl[w_pos] if cl is not None else None

    obs = int(w_dat2.numel())
    if obs < 5:
        est = torch.tensor(0.0, device=device, dtype=dtype)
        variance = (torch.tensor(1.0, device=device, dtype=dtype,) if se_yes
                    else None
                    )
        return est, variance, w_ret

    # Point estimate
    est = torch.dot(w_dat2, y_use.reshape(-1))

    variance: torch.Tensor | None
    if se_yes:
        # Bootstrap SE
        if bootstrap > 1:
            # Precompute cluster info for block bootstrap
            if p_cfg.cluster_std and (cl_use is not None) and not only_copy:
                cl_rounded = torch.round(cl_use)
                unique_cl = torch.unique(cl_rounded)
                obs_cl = int(unique_cl.numel())
            else:
                cl_rounded = None
                unique_cl = None
                obs_cl = 0

            g_cuda = torch.Generator(device=device)
            g_cuda.manual_seed(123345)
            est_b = torch.empty(bootstrap, device=device, dtype=dtype)

            for b_idx in range(bootstrap):
                if (p_cfg.cluster_std and
                        (cl_rounded is not None) and
                        not only_copy):
                    # Block bootstrap on clusters
                    idx_cl = torch.randint(low=0,
                                           high=obs_cl,
                                           size=(obs_cl,),
                                           generator=g_cuda,
                                           device=device,
                                           dtype=torch.long,
                                           )
                    cl_boot = unique_cl[idx_cl]

                    idx_list: list[torch.Tensor] = []
                    for cl_i in cl_boot:
                        mask = cl_rounded == cl_i
                        idx_i = torch.nonzero(
                            mask,
                            as_tuple=False,
                        ).view(-1)
                        if idx_i.numel() > 0:
                            idx_list.append(idx_i)
                    if idx_list:
                        idx = torch.cat(idx_list, dim=0)
                    else:
                        idx = torch.empty(0, dtype=torch.long, device=device)
                else:
                    # IID bootstrap
                    idx = torch.randint(low=0,
                                        high=obs,
                                        size=(obs,),
                                        generator=g_cuda,
                                        device=device,
                                        dtype=torch.long,
                                        )
                w_b = w_dat2[idx].clone()
                if normalize:
                    sum_w_b = w_b.sum().abs()
                    sum_w_b_val = float(sum_w_b)
                    close_zero = sum_w_b_val < zero_tol
                    close_one = abs(sum_w_b_val - 1.0) < sum_tol
                    if not (close_zero or close_one):
                        w_b = w_b / sum_w_b

                est_b[b_idx] = torch.dot(w_b, y_use[idx])

            variance = torch.var(est_b)

        # Analytic SE
        else:
            if p_cfg.cond_var:
                sort_ind = torch.argsort(w_dat2)
                y_s = y_use[sort_ind]
                w_s = w_dat2[sort_ind]

                if p_cfg.knn:
                    k = int(round(p_cfg.knn_const * sqrt(obs) * 2.0))
                    k = max(k, p_cfg.knn_min_k)
                    k = min(k, obs // 2)

                    exp_y_cond_w, var_y_cond_w = moving_avg_mean_var_cuda(
                        y_s, k, precision=precision,
                        )
                else:
                    band = bandwidth_nw_rule_of_thumb_cuda(
                        w_s,
                        kernel=p_cfg.nw_kern, zero_tol=zero_tol,
                        ) * p_cfg.nw_bandw

                    exp_y_cond_w = nadaraya_watson_cuda(
                        y_s, w_s, w_s,
                        p_cfg.nw_kern, band, zero_tol=zero_tol,
                        )
                    var_y_cond_w = nadaraya_watson_cuda(
                        (y_s - exp_y_cond_w) ** 2, w_s, w_s,
                        p_cfg.nw_kern, band, zero_tol=zero_tol,
                        )
                variance = torch.dot(w_s ** 2, var_y_cond_w)
                variance = variance + obs * torch.var(w_s * exp_y_cond_w)
            else:
                variance = obs * torch.var(w_dat2 * y_use)
    else:
        variance = None

    return est, variance, w_ret


def aggregate_cluster_pos_w_cuda(cl_dat: torch.Tensor,
                                 w_dat: torch.Tensor,
                                 y_dat: torch.Tensor | None = None,
                                 norma: bool = True,
                                 w2_dat: torch.Tensor | None = None,
                                 sweights: torch.Tensor | None = None,
                                 y2_compute: bool = False,
                                 precision: int = 32,
                                 zero_tol: float = 1e-15,
                                 sum_tol: float = 1e-12,
                                 ) -> tuple[torch.Tensor,
                                            torch.Tensor | None,
                                            torch.Tensor | None,
                                            torch.Tensor | None,]:
    """Aggregate weighted cluster means (CUDA version).

    Parameters
    ----------
    cl_dat : Tensor, shape (n,). Cluster indicators.
    w_dat : Tensor, shape (n,). Weights.
    y_dat : Tensor, shape (n,) or (n, q). Outcomes (optional).
    norma : bool. If True, normalize aggregated weights to sum to 1.
    w2_dat : Tensor, shape (n,). Second set of weights (optional).
    sweights : Tensor, shape (n,). Additional (sampling) weights (optional).
    y2_compute : bool. If True, compute second set of aggregated outcomes.
    precision : int. Floating precision selector for mcf_c.tdtype.
    zero_tol : float. Threshold below which weights are treated as zero.
    sum_tol : float. Tolerance for checking sums close to 1.

    Returns
    -------
    w_agg : Tensor, shape (n_clusters,). Aggregated weights (possibly
                                                             normalized).
    y_agg : Tensor or None, shape (n_clusters, q). Aggregated outcomes.
    w2_agg : Tensor or None, shape (n_clusters,). Second set of aggregated
                                                  weights.
    y2_agg : Tensor or None, shape (n_clusters, q). Second set of aggregated
                                                    outcomes.
    """

    device = cl_dat.device
    dtype = mcf_c.tdtype('float', precision)

    cl_flat = cl_dat.reshape(-1)
    w_flat = w_dat.reshape(-1)

    if y2_compute and w2_dat is None:
        raise ValueError('y2_compute=True requires w2_dat to be provided.')

    # Unique clusters
    cluster_vals = torch.unique(cl_flat)
    n_clusters = cluster_vals.numel()

    # Mask for “positive” (non-negligible) weights
    w_pos = w_flat.abs() > zero_tol

    # Handle outcomes
    if y_dat is not None:
        if y_dat.ndim == 1:
            y_dat = y_dat.reshape(-1, 1)
        else:
            y_dat = y_dat.reshape(-1, y_dat.shape[1])
        q_obs = y_dat.shape[1]

        y_agg = torch.zeros((n_clusters, q_obs), device=device, dtype=dtype,)
    else:
        y_agg = None
        q_obs = 0  # not used if y_dat is None

    # Second outcomes
    if y2_compute and y_dat is not None:
        y2_agg = torch.zeros((n_clusters, q_obs), device=device, dtype=dtype,)
    else:
        y2_agg = None

    # Second weight set
    if w2_dat is not None:
        w2_flat = w2_dat.reshape(-1)
        w2_agg = torch.zeros(n_clusters, device=device, dtype=dtype)
        w2_pos = w2_flat.abs() > zero_tol
    else:
        w2_flat = None
        w2_agg = None
        w2_pos = None

    # Optional sampling weights
    if sweights is not None:
        sweights_flat = sweights.reshape(-1)
    else:
        sweights_flat = None

    # Aggregation loop over clusters
    w_agg = torch.zeros(n_clusters, device=device, dtype=dtype)

    for j, cl_val in enumerate(cluster_vals):
        in_cluster = cl_flat == cl_val
        in_cluster_pos = in_cluster & w_pos

        if not torch.any(in_cluster_pos):
            continue

        # Main weight aggregation
        w_cluster = w_flat[in_cluster_pos]
        w_agg[j] = w_cluster.sum()

        # Aggregated outcomes (first set)
        if y_dat is not None:
            y_cluster = y_dat[in_cluster_pos, :]
            if sweights_flat is None:
                eff_w = w_cluster
            else:
                eff_w = w_cluster * sweights_flat[in_cluster_pos].reshape(-1)

            # Weighted mean over all outcome dimensions at once
            # shape: (1, n_in_cluster) @ (n_in_cluster, q_obs) -> (1, q_obs)
            num = eff_w.unsqueeze(0) @ y_cluster
            y_agg[j, :] = (num / w_agg[j]).squeeze(0)

        # Second weight set and outcomes
        if w2_flat is not None:
            # Note: sum over all in_cluster (as in original code)
            w2_agg[j] = w2_flat[in_cluster].sum()

            if y2_compute and y_dat is not None:
                in_cluster_pos2 = in_cluster & w2_pos
                if torch.any(in_cluster_pos2):
                    y_cluster2 = y_dat[in_cluster_pos2, :]
                    if sweights_flat is None:
                        eff_w2 = w2_flat[in_cluster_pos2]
                    else:
                        eff_w2 = (w2_flat[in_cluster_pos2]
                                  * sweights_flat[in_cluster_pos2].reshape(-1)
                                  )
                    num2 = eff_w2.unsqueeze(0) @ y_cluster2
                    y2_agg[j, :] = (num2 / w2_agg[j]).squeeze(0)

    # Optional normalization of aggregated weights to sum to 1
    if norma:
        sum_w_agg = w_agg.sum()
        sum_w_agg_val = sum_w_agg.item()
        if not 1.0 - sum_tol < sum_w_agg_val < 1.0 + sum_tol:
            w_agg = w_agg / sum_w_agg_val

        if w2_agg is not None:
            sum_w2_agg = w2_agg.sum()
            sum_w2_agg_val = sum_w2_agg.item()
            if not 1.0 - sum_tol < sum_w2_agg_val < 1.0 + sum_tol:
                w2_agg = w2_agg / sum_w2_agg_val

    return w_agg, y_agg, w2_agg, y2_agg


def moving_avg_mean_var_cuda(data: torch.Tensor,
                             k: int,
                             mean_and_var: bool = True,
                             precision: int = 32,
                             ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute moving average of mean and variance (CUDA/version-agnostic).

    Parameters
    ----------
    data : Tensor. 1d dependent variable. Typically sorted.
    k : int. Number of neighbours in the moving window.
    mean_and_var : bool. Whether to compute both mean and variance.

    Returns
    -------
    mean : Tensor, same length as data.
    var : Tensor if mean_and_var is True, otherwise None.
    """

    if data.ndim != 1:
        raise ValueError('moving_avg_mean_var_cuda expects a 1d tensor.')

    device = data.device
    dtype = mcf_c.tdtype('float', precision)

    # Ensure correct dtype on same device
    data = data.to(device=device, dtype=dtype)

    obs = data.shape[0]
    k = int(k)

    if k <= 0:
        raise ValueError('k must be a positive integer.')

    if k >= obs:
        mean_val = data.mean()
        mean = torch.full((obs,), mean_val, device=device, dtype=dtype)
        if mean_and_var:
            var_val = data.var(unbiased=False)
            var = torch.full((obs,), var_val, device=device, dtype=dtype,)
        else:
            var = None

        return mean, var

    # Rolling mean via cumulative sum
    cumsum = torch.cumsum(data, dim=0)
    cumsum[k:] = cumsum[k:] - cumsum[:-k]
    mean = cumsum[k - 1:] / k

    if mean_and_var:
        data_s = data ** 2
        cumsum_s = torch.cumsum(data_s, dim=0)
        cumsum_s[k:] = cumsum_s[k:] - cumsum_s[:-k]
        mean_s = cumsum_s[k - 1:] / k
        var = mean_s - mean ** 2
    else:
        var = None

    # Pad to original length by repeating edge values
    pad_before = (obs - mean.shape[0]) // 2
    pad_after = obs - mean.shape[0] - pad_before

    if pad_before > 0 or pad_after > 0:
        mean = torch.cat([mean[0].view(1).repeat(pad_before),
                          mean,
                          mean[-1].view(1).repeat(pad_after),
                          ],
                         )
        if mean_and_var:
            var = torch.cat([var[0].view(1).repeat(pad_before),
                             var,
                             var[-1].view(1).repeat(pad_after),
                             ],
                            )
    return mean, var


def bandwidth_nw_rule_of_thumb_cuda(data: torch.Tensor,
                                    kernel: int = 1,
                                    zero_tol: float = 1e-15,
                                    ) -> torch.Tensor:
    """Rule-of-thumb bandwidth for NW kernel regression (CUDA-friendly).

    Li & Racine (2004), Nonparametric Econometrics, bottom of p. 66.

    Returns
    -------
    bandwidth : 0-dim tensor (scalar) on same device/dtype as data.
    """

    data = data.reshape(-1)
    obs = data.shape[0]

    if obs < 5:
        raise ValueError(f'Only {obs} observations for bandwidth selection.')

    device = data.device
    dtype = data.dtype

    # IQR / 1.349
    q = torch.tensor([75.0, 25.0], device=device)
    q75, q25 = torch.percentile(data, q)
    iqr = (q75 - q25) / 1.349

    # Standard deviation
    std = torch.std(data)

    # sss = min(std, iqr) if iqr > zero_tol else std  (all in tensor-land)
    sss = torch.where(iqr > zero_tol, torch.minimum(std, iqr), std,)
    # Basic ROT bandwidth
    bandwidth = sss * (obs ** -0.2)

    # Kernel-specific constants
    if kernel == 1:        # Epanechnikov
        bandwidth = bandwidth * 2.34
    elif kernel == 2:      # Gaussian
        bandwidth = bandwidth * 1.06

    # Enforce lower bound: if bandwidth < zero_tol -> set to 1.0
    one = torch.tensor(1.0, device=device, dtype=dtype)
    bandwidth = torch.where(bandwidth < zero_tol, one, bandwidth)

    return bandwidth



def nadaraya_watson_cuda(y_dat: torch.Tensor,
                         x_dat: torch.Tensor,
                         grid: torch.Tensor,
                         kernel: int,
                         bandwidth: float | torch.Tensor,
                         zero_tol: float = 1e-15,
                         ) -> torch.Tensor:
    """Compute Nadaraya-Watson one dimensional nonparametric regression (cuda).

    Parameters
    ----------
    y_dat : Tensor. Dependent variable.
    x_dat :  Tensor. Independent variable.
    grid : Tensor. Values of x for which to create predictions.
    kernel : Int. 1: Epanechikov  2: Normal.
    bandwidth : Float. Bandwidth.

    Returns
    -------
    estimate : Tensor. Estimated quantity.

    """
    f_yx = kernel_density_y_cuda(y_dat, x_dat, grid, kernel, bandwidth)
    f_x = kernel_density_cuda(x_dat, grid, kernel, bandwidth)
    f_x = torch.clamp(f_x, min=zero_tol)

    return f_yx / f_x


def kernel_density_cuda(data: torch.Tensor,
                        grid: torch.Tensor,
                        kernel: int,
                        bandwidth: float | torch.Tensor,
                        ) -> torch.Tensor:
    """Compute nonparametric density estimate on a grid (CUDA version).

    Parameters
    ----------
    data : Tensor, shape (n,) or (n, 1). Sample points.
    grid : Tensor, shape (m,) or (m, 1). Evaluation grid.
    kernel : int. 1: Epanechnikov, 2: Gaussian (as in kernel_proc_cuda).
    bandwidth : float or 0-dim tensor. Bandwidth.

    Returns
    -------
    f_grid : Tensor, shape (m,). Estimated density at grid points.
    """

    # Ensure 1d
    data = data.reshape(-1)
    grid = grid.reshape(-1)

    device = data.device
    dtype = data.dtype

    # Move grid to same device/dtype as data (no cost if already there)
    grid = grid.to(device=device, dtype=dtype)

    # Bandwidth as scalar tensor on same device/dtype
    bw = torch.as_tensor(bandwidth, device=device, dtype=dtype)
    if bw <= 0:
        raise ValueError(f'Bandwidth must be positive, got {float(bw)}.')

    # differ has shape (n, m)
    differ = torch.subtract.outer(data, grid)

    # Kernel evaluation; expects tensor on correct device
    y_dach_i = kernel_proc_cuda(differ / bw, kernel)

    # Average over observations and divide by bandwidth
    f_grid = y_dach_i.mean(dim=0) / bw

    return f_grid


def kernel_density_y_cuda(y_dat: torch.Tensor,
                          x_dat: torch.Tensor,
                          grid: torch.Tensor,
                          kernel: int,
                          bandwidth: float | torch.Tensor,
                          ) -> torch.Tensor:
    """Estimate E[y * K((x - grid) / h)] style quantity (CUDA version).

    Parameters
    ----------
    y_dat : Tensor, shape (n,) or (n, q). Dependent variable.
    x_dat : Tensor, shape (n,). Independent variable.
    grid : Tensor, shape (m,) or (m, 1). Evaluation grid.
    kernel : int. 1: Epanechnikov, 2: Gaussian.
    bandwidth : float or 0-dim tensor. Bandwidth.

    Returns
    -------
    f_grid : Tensor, shape (m,) if y_dat was 1d, else (m, q).
    """

    # Ensure 1d x and grid
    x = x_dat.reshape(-1)
    grid = grid.reshape(-1)

    if y_dat.ndim == 1:
        y = y_dat.reshape(-1, 1)
        squeeze_out = True
    elif y_dat.ndim == 2:
        y = y_dat
        squeeze_out = False
    else:
        raise ValueError('y_dat must be 1d or 2d.')

    if y.shape[0] != x.shape[0]:
        raise ValueError('First dimension of y_dat and x_dat must match.')

    device = x.device
    dtype = x.dtype

    # Move everything to same device/dtype
    x = x.to(device=device, dtype=dtype)
    grid = grid.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    bw = torch.as_tensor(bandwidth, device=device, dtype=dtype,)
    if bw <= 0:
        raise ValueError(f'Bandwidth must be positive, got {float(bw)}.')

    # differ has shape (n_obs, n_grid)
    differ = x.unsqueeze(1) - grid.unsqueeze(0)

    y_dach_i = kernel_proc_cuda(differ / bw, kernel)

    # y_dach_i: (n, m), y: (n, q)  -> broadcasting ok
    # mean over observations (dim 0)
    f_grid = (y_dach_i * y).mean(dim=0) / bw

    if squeeze_out:
        f_grid = f_grid.squeeze(-1)

    return f_grid


def kernel_proc_cuda(data: torch.Tensor, kernel: int) -> torch.Tensor:
    """Feed data through kernel for nonparametric estimation (cuda version).

    This function works for matrices and vectors.

    Parameters
    ----------
    data : Tensor. Data.
    kernel : Int. 1: Epanechikov  2: Normal.

    Returns
    -------
    y : Tensor. Kernel applied to data.

    """
    if kernel == 1:
        abs_data = data.abs()
        inside = abs_data < 1
        y_dat = torch.zeros_like(data)
        y_dat[inside] = 0.75 * (1.0 - abs_data[inside] ** 2)

        return y_dat

    if kernel == 2:  # This works for matrices
        # Gaussian kernel: (1/sqrt(2π)) * exp(-u^2 / 2)
        inv_sqrt_2pi = 1.0 / sqrt(2.0 * pi)

        return inv_sqrt_2pi * torch.exp(-0.5 * data ** 2)

    raise ValueError('Only Epanechikov and normal kernel supported.')



def analyse_weights_cuda(weights: torch.Tensor,
                         title: str | None,
                         gen_cfg: Any,
                         p_cfg: Any,
                         ate: bool = True,
                         continuous: bool = False,
                         no_of_treat_cont: int | None = None,
                         d_values_cont: torch.Tensor | None = None,
                         precision: int = 32,
                         iv: bool = False,
                         sum_tol: float = 1e-12,
                         zero_tol: float = 1e-15,
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor, torch.Tensor,
                                    str,
                                    ]:
    """Describe the weights (CUDA version)."""

    device = weights.device
    float_dtype = mcf_c.tdtype('float', precision)
    int_dtype = mcf_c.tdtype('int', precision)

    txt = ''

    if ate:
        txt += '\n' * 2 + '=' * 100
        if iv:
            txt += '\nAnalysis of weights: '
        else:
            txt += '\nAnalysis of weights (normalised to add to 1): '
        if title is not None:
            txt += title

    if continuous:
        if no_of_treat_cont is None:
            raise ValueError('no_of_treat_cont must be provided when '
                             'continuous is True.'
                             )
        no_of_treat = no_of_treat_cont
    else:
        no_of_treat = gen_cfg.no_of_treat

    # Allocate result containers on correct device / dtype
    larger_0 = torch.zeros(no_of_treat, device=device, dtype=int_dtype,)
    equal_0 = torch.zeros_like(larger_0)

    mean_pos = torch.zeros( no_of_treat, device=device, dtype=float_dtype,)
    std_pos = torch.zeros_like(mean_pos)
    gini_all = torch.zeros_like(mean_pos)
    gini_pos = torch.zeros_like(mean_pos)

    share_largest_q = torch.zeros((no_of_treat, 3),
                                  device=device, dtype=float_dtype,
                                  )
    sum_larger = torch.zeros((no_of_treat, len(p_cfg.q_w)),
                             device=device, dtype=float_dtype,
                             )
    obs_larger = torch.zeros_like(sum_larger)

    # Precompute row sums
    sum_weights = weights.sum(dim=1)

    # Quantiles for "largest q" shares
    quantiles = torch.tensor([0.99, 0.95, 0.90],
                             device=device, dtype=float_dtype,
                             )
    few_obs_flag = False

    for j in range(no_of_treat):
        sum_j = float(sum_weights[j])

        close_to_one = abs(sum_j - 1.0) < sum_tol
        close_to_zero = abs(sum_j) < sum_tol

        if (not (close_to_one or close_to_zero)) and (not iv):
            w_j = weights[j] / sum_weights[j]
        else:
            w_j = weights[j]

        # Positive / relevant weights (by absolute value > sum_tol)
        w_pos = w_j[torch.abs(w_j) > sum_tol]
        n_pos = int(w_pos.numel())
        n_all = int(w_j.numel())

        larger_0[j] = n_pos
        equal_0[j] = n_all - n_pos

        if n_pos > 0:
            mean_pos[j] = w_pos.mean()
            std_pos[j] = w_pos.std()

            gini_all[j] = gini_coeff_pos_cuda(w_j, zero_tol=zero_tol) * 100.0
            gini_pos[j] = gini_coeff_pos_cuda(w_pos, zero_tol=zero_tol) * 100.0

            w_pos_abs = w_pos.abs()

            if n_pos > 5:
                # Shares of total weight in upper quantiles
                qqq = torch.quantile(w_pos_abs, quantiles)
                for i in range(3):
                    thr = qqq[i] - zero_tol
                    share_largest_q[j, i] = (w_pos_abs[w_pos_abs >= thr].sum()
                                             * 100.0
                                             )

                # Shares and counts above fixed thresholds q_w
                for idx_q, val in enumerate(p_cfg.q_w):
                    thr = val - zero_tol
                    mask = w_pos_abs >= thr
                    sum_larger[j, idx_q] = w_pos_abs[mask].sum() * 100.0
                    obs_larger[j, idx_q] = mask.sum().float() / n_pos * 100.0
            else:
                if gen_cfg.with_output:
                    few_obs_flag = True
        else:
            # No positive weights at all in this group
            if gen_cfg.with_output:
                few_obs_flag = True

    if few_obs_flag and gen_cfg.with_output:
        txt += '\nLess than 5 observations in some groups.'

    if ate:
        txt += mcf_ps.txt_weight_stat(
            larger_0, equal_0, mean_pos, std_pos,
            gini_all, gini_pos,
            share_largest_q, sum_larger, obs_larger,
            gen_cfg, p_cfg,
            continuous=continuous, d_values_cont=d_values_cont,
            )
    return (larger_0, equal_0, mean_pos, std_pos,
            gini_all, gini_pos,
            share_largest_q, sum_larger, obs_larger,
            txt,
            )


def gini_coeff_pos_cuda(x_dat: torch.Tensor,
                        zero_tol: float = 1e-15,
                        ) -> torch.Tensor:
    """Compute Gini coefficient of a tensor with values >= 0 (CUDA version)."""
    x_flat = x_dat.reshape(-1)
    sss = x_flat.sum()

    # sss is a tensor - use .item() for the scalar comparison
    if (sss > zero_tol).item():
        n = x_flat.numel()
        rrr = torch.argsort(-x_flat)
        ranks = torch.arange(1, n + 1,
                             device=x_flat.device, dtype=x_flat.dtype,
                             )
        num = torch.sum((ranks - 1) * x_flat[rrr]) + sss
        gini = 1 - 2 * num / (n * sss)

        return gini

    # If total weight is ~0, return 0 as a scalar tensor on same device/dtype
    return torch.zeros((), device=x_flat.device, dtype=x_flat.dtype)
