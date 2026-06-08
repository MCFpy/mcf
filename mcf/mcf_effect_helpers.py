"""
Created on Fri Apr 17 09:21:37 2026.

# -*- coding: utf-8 -*-

@author: MLechner

Helper functions for effect estimation.

"""
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from itertools import compress, chain, repeat
from time import time
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from mcf.mcf_estimation_generic import bandwidth_nw_rule_of_thumb, kernel_proc
from mcf.mcf_estimation import average_vals

type ArrayLike = NDArray[Any] | None
if TYPE_CHECKING:
    from mcf.mcf_init import GenCfg, VarCfg
    from mcf.mcf_init_predict import PCfg


@dataclass(slots=True, frozen=True, kw_only=True)
class EffParaFlags:
    """Holds the flags for the efficiency versions of the parameters."""

    ate_eff: bool
    gate_eff: bool
    iate_eff: bool
    qiate_eff: bool


@dataclass(slots=True, kw_only=True)
class EffectDicts:
    """Effect dictionaries updated during folds and rounds."""

    ate_dic: list[dict[str, Any] | None] = field(default_factory=list)
    bala_dic: list[dict[str, Any] | None] = field(default_factory=list)
    bgate_dic: list[dict[str, Any] | None] = field(default_factory=list)
    bgate_m_ate_dic: list[dict[str, Any] | None] = field(default_factory=list)
    cbgate_dic: list[dict[str, Any] | None] = field(default_factory=list)
    cbgate_m_ate_dic: list[dict[str, Any] | None] = field(default_factory=list)
    gate_dic: list[dict[str, Any] | None] = field(default_factory=list)
    gate_m_ate_dic: list[dict[str, Any] | None] = field(default_factory=list)
    iate_dic: list[dict[str, Any] | None] = field(default_factory=list)
    iate_m_ate_dic: list[dict[str, Any] | None] = field(default_factory=list)
    qiate_dic: list[dict[str, Any] | None] = field(default_factory=list)
    qiate_m_med_dic: list[dict[str, Any] | None] = field(default_factory=list)
    qiate_m_opp_dic: list[dict[str, Any] | None] = field(default_factory=list)

    bgate_est_dic: dict[str, Any] | None = None
    cbgate_est_dic: dict[str, Any] | None = None
    gate_est_dic: dict[str, Any] | None = None
    qiate_est_dic: dict[str, Any] | None = None

    txt_am: str = ''
    txt_b: str = ''


@dataclass(slots=True, kw_only=True)
class TimeState:
    """Keep the time for the computation of the indiviual components for effect estimation."""

    weight: float = 0.0
    ate: float = 0.0
    bala: float = 0.0
    bgate: float = 0.0
    cbgate: float = 0.0
    iate: float = 0.0
    qiate: float = 0.0
    gate: float = 0.0


def inst_eff_flags(gen_cfg: 'GenCfg') -> EffParaFlags:
    """Create an instance of EffParaFlags."""
    return EffParaFlags(ate_eff = gen_cfg.ate_eff,
                        gate_eff = gen_cfg.gate_eff,
                        iate_eff = gen_cfg.iate_eff,
                        qiate_eff = gen_cfg.qiate_eff,
                        )

def inst_effect_dicts(eff_flags_dc: EffParaFlags) -> EffectDicts:
    """Create an instance of EffectDicts."""
    def init_list_dic(eff: bool) -> list[None]:
        return [None, None] if eff else [None]

    return EffectDicts(ate_dic = init_list_dic(eff_flags_dc.ate_eff),
                       bala_dic = init_list_dic(eff_flags_dc.ate_eff),
                       bgate_dic = init_list_dic(eff_flags_dc.gate_eff),
                       bgate_m_ate_dic = init_list_dic(eff_flags_dc.gate_eff),
                       cbgate_dic = init_list_dic(eff_flags_dc.gate_eff),
                       cbgate_m_ate_dic = init_list_dic(eff_flags_dc.gate_eff),
                       gate_dic = init_list_dic(eff_flags_dc.gate_eff),
                       gate_m_ate_dic = init_list_dic(eff_flags_dc.gate_eff),
                       iate_dic = init_list_dic(eff_flags_dc.iate_eff),
                       iate_m_ate_dic = init_list_dic(eff_flags_dc.iate_eff),
                       qiate_dic = init_list_dic(eff_flags_dc.qiate_eff),
                       qiate_m_med_dic = init_list_dic(eff_flags_dc.qiate_eff),
                       qiate_m_opp_dic = init_list_dic(eff_flags_dc.qiate_eff),
                       gate_est_dic = None,
                       bgate_est_dic = None,
                       cbgate_est_dic = None,
                       qiate_est_dic = None,
                       txt_b = '',
                       txt_am = '',
                       )


def aggregate_effects(effect_dc: 'EffectDicts'):
    """Aggregate the effects from the different folds."""
    effect_dc.ate_dic = average_vals(effect_dc.ate_dic)
    effect_dc.bala_dic = average_vals(effect_dc.bala_dic)
    effect_dc.gate_dic = average_vals(effect_dc.gate_dic)
    effect_dc.gate_m_ate_dic = average_vals(effect_dc.gate_m_ate_dic)

    effect_dc.bgate_dic = average_vals(effect_dc.bgate_dic)
    effect_dc.bgate_m_ate_dic = average_vals(effect_dc.bgate_m_ate_dic)
    effect_dc.cbgate_dic = average_vals(effect_dc.cbgate_dic)
    effect_dc.cbgate_m_ate_dic = average_vals(effect_dc.cbgate_m_ate_dic)

    effect_dc.qiate_dic = average_vals(effect_dc.qiate_dic)
    effect_dc.qiate_m_med_dic = average_vals(effect_dc.qiate_m_med_dic)
    effect_dc.qiate_m_opp_dic = average_vals(effect_dc.qiate_m_opp_dic)

    effect_dc.iate_dic = average_vals(effect_dc.iate_dic)
    effect_dc.iate_m_ate_dic = average_vals(effect_dc.iate_m_ate_dic)


def accumulate_weights(weights: ArrayLike, weights_idx: ArrayLike) -> tuple[ArrayLike, float]:
    """Accumulate weights for more aggregrated effects."""
    time_start = time()
    if weights is None:
        if weights_idx is not None and not weights_idx.flags.writeable:
            weights_idx = weights_idx.copy()
        return weights_idx, time() - time_start

    if weights.flags.writeable:
        weights += weights_idx
    else:
        weights = weights + weights_idx

    return weights, time() - time_start


def single_treatment_pos(d_value: float,
                         d_values: NDArray[Any],
                         i_d_val: NDArray[Any], *,
                         zero_tol: float,
                         ) -> int:
    """Get position of treatment in i_d_val."""
    pos_arr = i_d_val[np.isclose(d_value, d_values, atol=zero_tol, rtol=zero_tol)]
    if pos_arr.size != 1:
        raise ValueError(f'Expected one treatment match, got {pos_arr.size}.')
    return int(pos_arr.item())


def need_ate_weights(round_: str, eff_flags: EffParaFlags, gate_yes: bool) -> bool:
    """Decide if ATE weights need to be computed."""
    return round_ == 'regular' or eff_flags.ate_eff or (eff_flags.gate_eff and gate_yes)


def w_diff_cont_func(t_idx: int | np.integer,
                     a_idx: int | np.integer, *,
                     no_of_treat: int,
                     w_gate_cont: NDArray[Any],
                     w_ate: NDArray[Any],
                     w10: float | np.floating,
                     w01: float | np.floating,
                     ) -> NDArray[Any]:
    """Compute weights for difference in continuous case."""
    w_ate_cont = w_ate[a_idx, t_idx, :] if t_idx == no_of_treat - 1 else (
        w10 * w_ate[a_idx, t_idx, :] + w01 * w_ate[a_idx, t_idx+1, :])
    w_ate_cont = w_ate_cont / np.sum(w_ate_cont)
    w_diff_cont = w_gate_cont - w_ate_cont

    return w_diff_cont


def assign_pot(y_pot: NDArray[Any],
               y_pot_var: NDArray[Any],
               y_pot_mate: NDArray[Any],
               y_pot_mate_var: NDArray[Any], *,
               results_fut_zj: Any,
               zj_idx: int | np.integer,
               ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Reduce repetetive code."""
    y_pot[zj_idx, :, :, :] = results_fut_zj[0]
    y_pot_var[zj_idx, :, :, :] = results_fut_zj[1]
    y_pot_mate[zj_idx, :, :, :] = results_fut_zj[2]
    y_pot_mate_var[zj_idx, :, :, :] = results_fut_zj[3]

    return y_pot, y_pot_var, y_pot_mate, y_pot_mate_var


def assign_w(w_gate: NDArray[Any],
             w_gate_unc: NDArray[Any],
             w_censored: NDArray[Any],
             results_fut_zj: Any,
             zj_idx: int,
             )-> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Reduce repetetive code."""
    w_gate[zj_idx, :, :, :] = results_fut_zj[4]
    w_gate_unc[zj_idx, :, :, :] = results_fut_zj[5]
    w_censored[zj_idx, :, :] = results_fut_zj[7]

    return w_gate, w_gate_unc, w_censored


def make_treat_comp_label(d_values: Any, *, continuous: bool) -> list[str]:
    """Create treatment-comparison labels."""
    treat_comp_label = []

    for t1_idx, t1_lab in enumerate(d_values):
        for t2_idx in range(t1_idx + 1, len(d_values)):
            treat_comp_label.append(str(d_values[t2_idx]) + 'vs' + str(t1_lab))
        if continuous:
            break

    return treat_comp_label


def normalize_ate_weights(w_ate: NDArray[Any], *,
                          no_of_tgates: int,
                          no_of_treat: int,
                          zero_tol: float,
                          sum_tol: float,
                          ) -> NDArray[Any]:
    """Normalize ATE weights over observations for each target population and treatment."""
    w_ate_sum = np.sum(w_ate, axis=2)

    for a_idx in range(no_of_tgates):
        for t_idx in range(no_of_treat):
            if abs(w_ate_sum[a_idx, t_idx]) <= zero_tol:
                raise ValueError('ATE weights sum to zero.')

            if not ((1 - sum_tol) < w_ate_sum[a_idx, t_idx] < (1 + sum_tol)):
                w_ate[a_idx, t_idx, :] = w_ate[a_idx, t_idx, :] / w_ate_sum[a_idx, t_idx]

    return w_ate


def gate_kernel_bandwidth(z_p: NDArray[Any],
                          z_name_j: int | np.integer,
                          z_smooth: bool, *,
                          zero_tol: float,
                          bandwidth_factor: float,
                          ) -> tuple[int | None, float | None]:
    """Return kernel and bandwidth for smoothed GATEs."""
    if not z_smooth:
        return None, None

    kernel = 1  # Epanechnikov
    bandw_z = bandwidth_nw_rule_of_thumb(z_p[:, z_name_j], zero_tol=zero_tol) * bandwidth_factor

    return kernel, bandw_z


def gate_smooth_norm_factor(z_dat: NDArray[Any],
                            z_val: int | float, *,
                            bandwidth: int | float,
                            kernel: int,
                            sum_tol: float,
                            ) -> float:
    """Compute global normalization factor for smoothed GATE weights."""
    w_z_val = kernel_proc((z_dat - z_val) / bandwidth, kernel)
    w_z_val = w_z_val[w_z_val > sum_tol]

    w_z_sum = np.sum(w_z_val)
    if w_z_sum <= sum_tol:
        raise ValueError('Smoothing weights sum to zero.')

    return len(w_z_val) / w_z_sum


def init_gate_pot_arrays(*, no_of_zval: int,
                         no_of_tgates: int,
                         no_of_treat_dr: int,
                         no_of_out: int,
                         gen_tv_yes: bool,
                         no_of_treat_per_main: int = 1,
                         ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Initialize GATE potential-outcome arrays."""
    pot = np.empty((no_of_zval, no_of_tgates, no_of_treat_dr, no_of_out))
    pot_var = np.empty_like(pot)
    pot_mate = np.empty_like(pot)
    pot_mate_var = np.empty_like(pot)

    if gen_tv_yes:
        pot = np.repeat(pot, repeats=no_of_treat_per_main, axis=2)
        pot_var = np.repeat(pot_var, repeats=no_of_treat_per_main, axis=2)
        pot_mate = np.repeat(pot_mate, repeats=no_of_treat_per_main, axis=2)
        pot_mate_var = np.repeat(pot_mate_var, repeats=no_of_treat_per_main, axis=2)

    return pot, pot_var, pot_mate, pot_mate_var


def get_w_rel_z(z_dat: NDArray[Any],
                z_val: int | float, *,
                weights_all: list[list[NDArray[Any]]] | list[Any] | None,
                smooth_it: bool,
                bandwidth: int | float = 1,
                kernel: int = 1,
                w_is_csr: bool = False,
                sum_tol: float = 1e-8,
                normalize_smooth_z: bool = True,
                relevant_data_points_only: bool = False,
                ) -> tuple[list[list[NDArray[Any]]] | list[Any] | None,
                           NDArray[Any], NDArray[Any]
                           ]:
    """
    Get relevant observations and their weights.

    Parameters
    ----------
    z_dat : 1D Numpy array. Data.
    z_val : Int or float. Evaluation point.
    weights_all : List of lists of lists. MCF weights.
    smooth_it : Bool. Use smoothing (True) or select data.
    bandwidth : Float. Bandwidth for weights. Default is 1.
    kernel : Int. 1: Epanechikov. 2: Normal. Default is 1.
    w_is_csr : Boolean. If weights are saved as sparse csv matrix.

    Returns
    -------
    weights : List of list of list. Relevant observations.
    relevant_data_points : 1D Numpy array of Bool. True if data will be used.
    w_z_val : Numpy array. Weights.

    """
    if smooth_it:
        w_z_val = kernel_proc((z_dat - z_val) / bandwidth, kernel)
        relevant_data_points = w_z_val > sum_tol
        if not relevant_data_points_only:
            w_z_val = w_z_val[relevant_data_points]
            if normalize_smooth_z:
                w_z_val = w_z_val / np.sum(w_z_val) * len(w_z_val)  # Normalise
    else:
        relevant_data_points = np.isclose(z_dat, z_val)  # Creates tuple
        w_z_val = None

    if relevant_data_points_only:
        return None, relevant_data_points, None

    if weights_all is None:
        weights = None
    else:
        if w_is_csr:
            iterator = len(weights_all)
            weights = [weights_all[t_idx][relevant_data_points, :] for t_idx in
                       range(iterator)]
        else:
            weights = list(compress(weights_all, relevant_data_points))

    return weights, relevant_data_points, w_z_val


def prepare_gate_z_context(z_names: list[str] | tuple[str, ...] | str,
                           var_x_values: dict[str, Any], *,
                           smooth_yes: bool,
                           z_name_smooth: list[str] | tuple[str, ...] | set[str] | None,
                           ) -> tuple[list[str], list[Any], list[bool]]:
    """Prepare GATE heterogeneity-variable names, evaluation values, and smoothing flags."""
    z_name_l = [z_names] if isinstance(z_names, str) else list(z_names)
    z_name_smooth_set = set(z_name_smooth or ())

    z_values_l = [var_x_values[z_name] for z_name in z_name_l]
    z_smooth_l = [smooth_yes and z_name in z_name_smooth_set for z_name in z_name_l]

    return z_name_l, z_values_l, z_smooth_l


@dataclass(slots=True, kw_only=True)
class GateContext:
    """Full-sample GATE context used by low-memory chunk computations."""

    z_name_l: list[str]
    z_values_l: list[Any]
    z_smooth_l: list[bool]
    kernel_bandwidth_l: list[tuple[int | None, float | None]]
    var_x_values: dict[str, Any]
    smooth_yes: bool
    z_name_smooth: list[str] | None


def add_dataclass_attributes_inplace(obj1: Any, obj2: Any) -> None:
    """Add all numeric dataclass attributes of obj2 to obj1 in place."""
    if not (is_dataclass(obj1) and is_dataclass(obj2)):
        raise TypeError('Both inputs must be dataclass instances.')
    if type(obj1) is not type(obj2):
        raise TypeError('Both inputs must have the same dataclass type.')

    for f in fields(obj1):
        setattr(obj1, f.name, getattr(obj1, f.name) + getattr(obj2, f.name))


def adjust_var_mult_duplicates(matches: NDArray[Any] | list[Any]) -> float:
    """Compute multiplier of variance adjusting for duplicate matches."""
    matches_np = np.array(matches)
    unique_values, counts = np.unique(matches_np, return_counts=True)
    n_unique = len(unique_values)
    d_i = counts - 1
    var_dupl_mult = (n_unique + np.sum(2 * d_i + d_i**2)) / (n_unique + np.sum(d_i))

    return var_dupl_mult


def addsmoothvars(data_df: DataFrame,
                  var_cfg: 'VarCfg',
                  var_x_values: dict,
                  p_cfg: 'PCfg',
                  ) -> tuple[dict, dict, bool, list[str]]:
    """
    Find variables for which to smooth gates and evaluation points.

    Parameters
    ----------
    data_df: DataFrame. Prediction data.
    var_cfg : VarCfg Dataclass. Variables.
    var_x_values : Dict. Variables
    p_cfg : PCfg. Controls.

    Returns
    -------
    var_cfg_new : VarCfg Dataclass. Updated variables.
    var_x_values_new : Dict. Updated with evaluation points.
    smooth_yes : Bool. Indicator if smoothing will happen.
    ...
    """
    smooth_yes, z_name = False, var_cfg.z_name
    z_name_add = [name[:-4] for name in z_name if (name.endswith('catv')
                                                   and (len(name) > 4))]
    if z_name_add:
        smooth_yes = True
        var_cfg_new = deepcopy(var_cfg)
        var_x_values_new = deepcopy(var_x_values)
        data_np = data_df[z_name_add].to_numpy()
        for idx, name in enumerate(z_name_add):
            var_x_values_new[name] = smooth_gate_eva_values(data_np[:, idx],
                                                            p_cfg.gates_smooth_no_evalu_points,
                                                            )
            var_cfg_new.z_name.append(name)
    else:
        var_cfg_new, var_x_values_new = var_cfg, var_x_values

    return var_cfg_new, var_x_values_new, smooth_yes, z_name_add


def smooth_gate_eva_values(z_dat: NDArray[Any], no_eva_values: dict) -> list[NDArray[Any]]:
    """
    Get the evaluation points.

    Parameters
    ----------
    z_dat : Numpy 1D array. Data.
    no_eva_values : Int.

    Returns
    -------
    eva_values : List of numpy.float. Evaluation values.

    """
    unique_vals = np.unique(z_dat)
    obs = len(unique_vals)
    if no_eva_values >= obs:
        return list(unique_vals)
    quas = np.linspace(0.01, 0.99, no_eva_values)

    return np.unique(np.quantile(z_dat, quas)).tolist()


def ref_vals_cbgate(data_df: DataFrame,
                    var_x_type: dict,
                    var_x_values: dict,
                    no_eva_values: int = 50,
                    ) -> dict:
    """Compute reference values for moderated gates and balanced gates."""
    evaluation_values = {}
    obs = len(data_df)
    for vname in var_x_type.keys():
        ddf = data_df[vname]
        if not var_x_values[vname]:
            no_eva_values = min(no_eva_values, obs)
            quas = np.linspace(0.01, 0.99, no_eva_values)
            eva_val = np.unique(ddf.quantile(quas)).tolist()
        else:
            eva_val = var_x_values[vname].copy()
        evaluation_values.update({vname: eva_val})

    return evaluation_values


def ref_data_bgate(data_df: DataFrame,
                   z_name: str,
                   p_cfg: 'PCfg',
                   eva_values: dict,
                   x_name_balance_bgate: list[str], *,
                   with_output_verbose: bool = True,
                   zero_tol: float = 1e-10,
                   ) -> tuple[DataFrame, dict, list[int, str]]:
    """Create reference samples for covariates (BGATE)."""
    eva_values = eva_values[z_name]
    no_eval, obs, txt = len(eva_values), len(data_df), ''
    if z_name in x_name_balance_bgate:
        x_name_balance_bgate.remove(z_name)
    if z_name.endswith('catv') and z_name[:-4] in x_name_balance_bgate:
        x_name_balance_bgate.remove(z_name[:-4])
    if z_name + 'catv' in x_name_balance_bgate:  # the continuous variable
        x_name_balance_bgate.remove(z_name + 'catv')
    if not x_name_balance_bgate:
        raise ValueError('BGATE {z_name}: No variables left for balancing.')
    if obs / no_eval > 10:  # Save computation time by using random samples
        share = p_cfg.bgate_sample_share / no_eval
        if 0 < share < 1:
            rng = np.random.default_rng(seed=9324561)
            idx = rng.choice(obs, int(np.floor(obs * share)), replace=False)
            obs = len(idx)
            if with_output_verbose:
                txt += f' BGATE: {share:5.2%} random sample drawn.'
        else:
            idx = np.arange(obs)
    else:
        idx = np.arange(obs)
    new_idx_dataframe = list(chain.from_iterable(repeat(idx, no_eval)))
    data_new_df = data_df.loc[new_idx_dataframe, :]
    new_values_z = list(chain.from_iterable([[i] * obs for i in eva_values]))
    data_new_df.loc[:, z_name] = new_values_z
    if with_output_verbose:
        txt += (f'\nBGATEs are balanced with respect to {" ".join(x_name_balance_bgate)}'
                '\nBGATEs minus ATE are evaluated at fixed z-feature values (equally weighted).'
                )
    # Until here these are the identical steps to CBGATE. Next, observations in
    # data_all_df get substituted by their nearest neighbours in terms of
    # x_name_balance_bgate conditional on the z_name variables.
    # Get duplicates
    data_new_b_np = data_new_df[x_name_balance_bgate].to_numpy(copy=True)
    data_new_z_np = data_new_df[z_name].to_numpy(copy=True)
    data_org_b_np = data_df[x_name_balance_bgate].to_numpy()
    data_org_z_np = data_df[z_name].to_numpy()
    data_org_np = data_df.to_numpy()
    if data_org_b_np.shape[1] > 1:
        bz_cov_inv = invcovariancematrix(data_org_b_np)
        bz_cov_inv[-1, -1] *= 10
    # Give much more additional weight to the z-related component in matching
    incr = 0.2 * np.std(data_new_z_np)
    collect_matches = []
    for idx, z_value in enumerate(data_new_z_np):
        z_true = np.isclose(data_org_z_np, z_value, atol=zero_tol, rtol=zero_tol)
        test_val = 0
        for _ in range(5):
            if np.count_nonzero(z_true) > 10:
                break
            test_val += incr
            lower_ok = data_org_z_np >= z_value - test_val
            upper_ok = data_org_z_np <= z_value + test_val
            lower_upper_ok = lower_ok & upper_ok
            z_true = np.zeros_like(data_org_z_np, dtype=bool)
            z_true[lower_upper_ok] = True
        if not np.any(z_true):
            z_true = np.ones_like(data_org_z_np, dtype=bool)
        data_org_np_condz = data_org_np[z_true]
        data_org_b_np_condz = data_org_b_np[z_true]
        diff = data_org_b_np_condz - data_new_b_np[idx, :]
        if data_org_b_np.shape[1] > 1:
            dist = np.sum((diff @ bz_cov_inv) * diff, axis=1)
        else:
            dist = diff**2
        match_neigbour_idx = np.argmin(dist)
        data_new_df.iloc[idx] = data_org_np_condz[match_neigbour_idx, :]
        collect_matches.append(match_neigbour_idx)

    return data_new_df, eva_values, collect_matches, txt


def invcovariancematrix(data_np: NDArray[Any]) -> NDArray[Any]:
    """Compute inverse of covariance matrix and adjust for missing rank."""
    k = np.shape(data_np)
    if k[1] > 1:
        cov_x = np.cov(data_np, rowvar=False)
        rank_not_ok, counter = True, 0
        while rank_not_ok:
            if counter == 20:
                cov_x *= np.eye(k[1])
            elif counter > 20:
                cov_inv = np.eye(k[1])
                break
            if np.linalg.matrix_rank(cov_x) < k[1]:
                cov_x += 0.5 * np.diag(cov_x) * np.eye(k[1])
                counter += 1
            else:
                cov_inv = np.linalg.inv(cov_x)
                rank_not_ok = False

    return cov_inv


def ref_data_cbgate(data_df: DataFrame,
                    z_name: str,
                    p_cfg: 'PCfg',
                    eva_values: dict,
                    with_output_verbose: bool = True,
                    ) -> tuple[DataFrame, dict, str]:
    """Create reference samples for covariates (CBGATE)."""
    eva_values = eva_values[z_name]
    no_eval, obs, txt = len(eva_values), len(data_df), ''
    if obs / no_eval > 10:  # Save computation time by using random samples
        share = p_cfg.bgate_sample_share / no_eval
        if 0 < share < 1:
            rng = np.random.default_rng(seed=9324561)
            idx = rng.choice(obs, int(np.floor(obs * share)), replace=False)
            obs = len(idx)
            if with_output_verbose:
                txt += f' CBGATE: {share:5.2%} random sample drawn'
        else:
            idx = np.arange(obs)
    else:
        idx = np.arange(obs)
    new_idx_dataframe = list(chain.from_iterable(repeat(idx, no_eval)))
    data_all_df = data_df.loc[new_idx_dataframe, :]
    new_values_z = list(chain.from_iterable([[i] * obs for i in eva_values]))
    data_all_df.loc[:, z_name] = new_values_z
    if with_output_verbose:
        txt += '\nCBGATEs minus ATE are evaluated at fixed z-feature values (equally weighted).'

    return data_all_df, eva_values, txt
