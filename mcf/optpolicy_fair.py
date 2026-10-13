"""Created on Wed May  1 16:35:19 2024.

Functions for correcting variables w.r.t. protected variables.

@author: MLechner

# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.neighbors import KernelDensity

from mcf import mcf_estimation_generic as mcf_gf
from mcf import mcf_print_stats as mcf_ps
from mcf import optpolicy_fair_add as optp_fair_add

if TYPE_CHECKING:
    from mcf.optpolicy_init import GenCfg, VarCfg, FairCfg

def adjust_decision_variables(fair_cfg: 'FairCfg',
                              gen_cfg: 'GenCfg',
                              var_cfg: 'VarCfg',
                              data_df: pd.DataFrame, *,
                              training: bool = True,
                              seed: int = 1246546
                              ) -> tuple[pd.DataFrame, list[str], list[str], dict, 'FairCfg', str]:
    """Make decision var's (conditionally) independent of protected features."""
    # 1.categorize decision variables into groups whether to be treated as
    # continuous or unordered
    (x_cont_name, x_disc_name, fair_info_dict) = optp_fair_add.cont_or_discrete(
        data_df, var_cfg.x_ord_name, var_cfg.x_unord_name, fair_cfg.cont_min_values
        )
    # 2.Use respective method for fairness adjustments
    x_fair_ord_name, x_fair_unord_name, txt_report = [], [], ''
    data_fair_df = data_df.copy()
    if gen_cfg.method == 'policy_tree':
        if fair_cfg.adj_type != 'Quantiled':

            fair_cfg.adj_type = 'Quantiled'
            txt = ('\nAdjustment method for variables changed to '
                   "'Quantiled' (because method 'Policy Tree' is used)."
                   )
            txt_report += txt
        if fair_cfg.material_disc_method not in fair_cfg.discretization_methods:
            fair_cfg.material_disc_method = 'Kmeans'
            txt = ('\nDiscretization method for materially relevant variables '
                   "changed to 'Kmeans' (because method 'Policy Tree' is used)."
                   )
            txt_report += txt
        if (fair_cfg.protected_disc_method not
                in fair_cfg.discretization_methods):
            fair_cfg.protected_disc_method = 'Kmeans'
            txt = ('\nDiscretization method for protected variables '
                   "changed to 'Kmeans' (because method 'Policy Tree' is used)."
                   )
            txt_report += txt
    strata_number_fair = (gen_cfg.with_output and fair_cfg.solvefair_used
                          and fair_cfg.adjust_target in ('xvariables', 'scores_xvariables',)
                          )
    if x_cont_name:
        (data_fair_df, x_fair_cont_name, fair_cfg, txt_cont
         ) = fair_adjust_cont_disc(fair_cfg, gen_cfg, var_cfg, data_fair_df,
                                   x_cont_name, x_are_continous=True, seed=seed,
                                   with_output=True,
                                   strata_number_fair=strata_number_fair,
                                   decision_variable_adjust=True,
                                   )
        txt_report += txt_cont
        for name_idx, name in enumerate(x_cont_name):
            if name in var_cfg.x_ord_name:
                x_fair_ord_name.append(x_fair_cont_name[name_idx])

    if x_disc_name:
        (data_fair_df, x_fair_disc_name, fair_cfg, txt_disc
         ) = fair_adjust_cont_disc(fair_cfg, gen_cfg, var_cfg, data_fair_df,
                                   x_disc_name, x_are_continous=False,
                                   seed=seed, with_output=True,
                                   strata_number_fair=strata_number_fair,
                                   decision_variable_adjust=True,
                                   )
        txt_report += txt_disc
        for name_idx, name in enumerate(x_disc_name):
            if name in var_cfg.x_ord_name:
                x_fair_ord_name.append(x_fair_disc_name[name_idx])
            else:
                x_fair_unord_name.append(x_fair_disc_name[name_idx])

    if training and fair_cfg.data_train_for_pred_df is None:
        x_name = [*var_cfg.x_ord_name, *var_cfg.x_unord_name,
                  *var_cfg.protected_name, *var_cfg.material_name
                  ]
        x_name_extend = [var for var in (*var_cfg.material_unord_name,
                                         *var_cfg.protected_unord_name)
                         if var not in x_name
                         ]
        if x_name_extend:
            x_name.extend(x_name_extend)

        fair_cfg.data_train_for_pred_df = data_fair_df[x_name]
    if strata_number_fair:
        fair_cfg.protected_matrel = data_df[var_cfg.prot_mat_no_dummy_name]

    return data_fair_df, x_fair_ord_name, x_fair_unord_name, fair_info_dict, fair_cfg, txt_report


def adjust_scores(fair_cfg: 'FairCfg',
                  gen_cfg: 'GenCfg',
                  var_cfg: 'VarCfg',
                  data_df: pd.DataFrame,
                  seed: int = 1246546,
                  ) -> tuple[pd.DataFrame, list[str], dict, 'FairCfg', 'VarCfg', str]:
    """Remove effect of protected variables from policy score."""
    (data_fair_df, score_fair_name, fair_cfg, txt_report
     ) = fair_adjust_cont_disc(fair_cfg, gen_cfg, var_cfg, data_df,
                               var_cfg.polscore_name,
                               x_are_continous=True,
                               strata_number_fair=False,
                               seed=seed,
                               decision_variable_adjust=False
                               )
    if fair_cfg.consistency_test:
        tests_dict, text = test_for_consistency(fair_cfg, gen_cfg, var_cfg,
                                                data_fair_df,
                                                score_fair_name,
                                                seed=seed*2,
                                                title='Consistency test - '
                                                )
        txt_report += text
    else:
        tests_dict = {}

    # Change the names of variables (in particular scores) to be used for
    # policy learning.
    var_cfg = optp_fair_add.change_variable_names_fair(var_cfg, score_fair_name)

    return data_fair_df, score_fair_name, tests_dict, fair_cfg, var_cfg, txt_report


def fair_adjust_cont_disc(fair_cfg: 'FairCfg',
                          gen_cfg: 'GenCfg',
                          var_cfg: 'VarCfg',
                          data_df: pd.DataFrame,
                          x_name: list | tuple, *,
                          x_are_continous: bool = True,
                          strata_number_fair: bool = False,
                          seed: int = 1246546,
                          with_output: bool = True,
                          decision_variable_adjust: bool = False,
                          ) -> tuple[pd.DataFrame, list[str], 'FairCfg', str]:
    """Fairness-adjust continuous and discrete variables."""
    fair_cfg = deepcopy(fair_cfg)
    requested_adj_type = fair_cfg.adj_type

    x_np, protect_np, material_np = vars_to_numpy(data_df, x_name,
                                                  var_cfg.protected_name, var_cfg.material_name
                                                  )
    txt_report = ''
    if not x_are_continous and fair_cfg.adj_type != 'Quantiled':
        txt_report = ('\nAdjustment method for discrete variables changed '
                      f"from '{fair_cfg.adj_type}' to 'Quantiled'."
                      )
        fair_cfg.adj_type = 'Quantiled'
        if gen_cfg.with_output:
            mcf_ps.print_mcf(gen_cfg, txt_report, summary=True)

    txt_report += fair_method_message(gen_cfg, fair_cfg.adj_type,
                                      continuous_vars=x_are_continous)
    # Get fairness adjusted variables
    if x_are_continous:
        x_fair_np, cluster_fair_np, fair_cfg, txt = fair_continuous_xvars_fct(
            fair_cfg, gen_cfg, x_np, x_name, protect_np,
            material_np=material_np, seed=seed, title='', strata_number_fair=strata_number_fair,
            )
    else:
        x_fair_np, cluster_fair_np, fair_cfg, txt = fair_discrete_xvars_fct(
            fair_cfg, gen_cfg, x_np, protect_np,
            material_np=material_np, seed=seed, title='', strata_number_fair=strata_number_fair,
            )
    txt_report += txt

    # Convert numpy array of fair variable to pandas dataframe and add
    data_fair_df, x_fair_name, fair_strata_df, x_org_df = update_fair_df(data_df, x_name,
                                                                         x_fair_np, cluster_fair_np,
                                                                         )

    if gen_cfg.with_output and with_output:
        txt_report += optp_fair_add.fair_stats(gen_cfg, var_cfg, data_fair_df, x_name, x_fair_name,
                                               var_continuous=x_are_continous,
                                               cont_min_values=fair_cfg.cont_min_values,
                                               )
    if decision_variable_adjust:
        fair_cfg.fair_strata = fair_strata_df

        if fair_cfg.decision_vars_org_df is None:
            fair_cfg.decision_vars_org_df = x_org_df
        else:
            fair_cfg.decision_vars_org_df = pd.concat((fair_cfg.decision_vars_org_df, x_org_df),
                                                      axis=1
                                                      )
        add_name_dict = dict(zip(x_fair_name, x_name))
        if fair_cfg.decision_vars_fair_org_name is None:
            fair_cfg.decision_vars_fair_org_name = add_name_dict
        else:
            fair_cfg.decision_vars_fair_org_name.update(
                {k: v for k, v in add_name_dict.items()
                 if k not in fair_cfg.decision_vars_fair_org_name}
                )

        if fair_cfg.x_ord_org_name is None:
            fair_cfg.x_ord_org_name = var_cfg.x_ord_name

    fair_cfg.adj_type = requested_adj_type
    return data_fair_df, x_fair_name, fair_cfg, txt_report


def update_fair_df(data_df: pd.DataFrame,
                   x_name: list[str],
                   x_fair_np: NDArray[Any],
                   cluster_fair_np: NDArray[Any] | None,
                   ) -> tuple[pd.DataFrame, list[str], pd.DataFrame | None, pd.DataFrame]:
    """Add or replace adjusted columns while retaining one column per name."""
    x_fair_name = [name + '_fair' for name in x_name]
    conflicts = set(x_fair_name).intersection(x_name)
    if conflicts:
        raise ValueError('Adjusted names conflict with original variables in the same adjustment: '
                         f'{", ".join(sorted(conflicts))}. Rename these input columns.'
                         )
    x_fair_df = pd.DataFrame(x_fair_np, columns=x_fair_name)
    fair_strata_df = (None if cluster_fair_np is None
                      else pd.DataFrame(cluster_fair_np, columns=('cluster',)))
    x_org_df = data_df[x_name]
    data_df = data_df.reset_index(drop=True).drop(columns=x_fair_name, errors='ignore')
    data_fair_df = pd.concat((data_df, x_fair_df), axis=1)

    return data_fair_df, x_fair_name, fair_strata_df, x_org_df


def fair_method_message(gen_cfg: 'GenCfg',
                        adj_type: str,
                        continuous_vars: bool = True
                        ) -> str:
    """Print info about method used."""
    if gen_cfg.with_output:
        var_type = 'continuous' if continuous_vars else 'discrete'
        txt_report = ('\nMethod selected for fairness adjustment of '
                      f'{var_type} variables: {adj_type}'
                      )
        mcf_ps.print_mcf(gen_cfg, txt_report, summary=True)
    else:
        txt_report = ''

    return txt_report


def vars_to_numpy(data_df: pd.DataFrame,
                  vars_name: list[str] | tuple[str, ...] | None,
                  protected_name: list[str] | tuple[str, ...] | None,
                  material_name:  list[str] | tuple[str, ...] | None,
                  ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Transfer some dataframes to numpy."""
    vars_np = data_df[vars_name].to_numpy(copy=True) if vars_name else None
    protect_np = data_df[protected_name].to_numpy(copy=True) if protected_name else None
    material_np = data_df[material_name].to_numpy(copy=True) if material_name else None

    return vars_np, protect_np, material_np


def fair_continuous_xvars_fct(fair_cfg: 'FairCfg',
                              gen_cfg: 'GenCfg',
                              xvars_np: NDArray[Any],
                              xvars_name: list,
                              protect_np: NDArray[Any], *,
                              material_np: NDArray[Any] | None,
                              strata_number_fair: bool = False,
                              seed: int = 1234567,
                              title: str = '',
                              ) -> tuple[NDArray[Any], NDArray[Any], 'FairCfg', str]:
    """Use one of the different fairness adjustment methods."""
    adj = getattr(fair_cfg, 'adj_type')
    match adj:
        case 'Mean' | 'MeanVar':
            xvars_fair_np, txt = residualisation(fair_cfg, gen_cfg, xvars_np, xvars_name,
                                                 protect_np,
                                                 material_np=material_np, seed=seed, title=title
                                                 )
            cluster_fair_np = None
        case 'Quantiled':
            xvars_fair_np, cluster_fair_np, txt, fair_cfg = quantalisation(
                fair_cfg, gen_cfg, xvars_np, protect_np, material_np, seed=seed,
                title=title, continuous=True, strata_number_fair=strata_number_fair
                )
        case _:
            raise ValueError('Invalid method selected for fairness adjustment.')

    return xvars_fair_np, cluster_fair_np, fair_cfg, txt


def fair_discrete_xvars_fct(fair_cfg: 'FairCfg',
                            gen_cfg: 'GenCfg',
                            xvars_np: NDArray[Any],
                            protect_np: NDArray[Any], *,
                            material_np: NDArray[Any] | None,
                            strata_number_fair: bool = False,
                            seed: int = 1234567,
                            title: str = '',
                            ) -> tuple[NDArray[Any], NDArray[Any], 'FairCfg', str]:
    """Use one of the different fairness adjustment methods."""
    # 1. Add a little noise, so that obs with same value appear to be ordered.
    #    Importantly, this ordering is random.
    xvars_noise_np, unique_vals = optp_fair_add.add_noise_discrete(xvars_np.copy())
    # 2. Switch the values of the (semi-random) ranks from the conditional
    #    distribution to the (semi-random) ranks of the unconditional
    #    distribution
    xvars_fair_noise_np, cluster_fair_np, txt, fair_cfg = quantalisation(
        fair_cfg, gen_cfg, xvars_noise_np, protect_np, material_np, seed=seed,
        title=title, continuous=False, strata_number_fair=strata_number_fair,
        )
    # 3. Remove the noise so that the original scaling is retained (rounding).
    xvars_fair_np = optp_fair_add.remove_noise_discrete(xvars_fair_noise_np.copy(), unique_vals)

    return xvars_fair_np, cluster_fair_np, fair_cfg, txt


def quantalisation(fair_cfg: 'FairCfg',
                   gen_cfg: 'GenCfg',
                   xvars_np: NDArray[Any],
                   protect_np: NDArray[Any],
                   material_np: NDArray[Any] | None, *,
                   strata_number_fair: bool = False,
                   seed: int = 1246546,
                   title: str = '',
                   continuous: bool = True,
                   ) -> tuple[NDArray[Any], NDArray[Any] | None, str, 'FairCfg']:
    """Adjust by quantalisation similar to Strack & Yang (2024)."""
    xvars_np = np.asarray(xvars_np, dtype=float)
    txt_report = ''
    fair_cfg = deepcopy(fair_cfg)
    requested_disc_methods = (fair_cfg.protected_disc_method,
                              fair_cfg.material_disc_method)
    disc_methods = fair_cfg.discretization_methods
    if gen_cfg.with_output:
        if title is None or title == '':
            print_txt = '\nComputing '
        else:
            print_txt = '\n' + title + 'computing '
        var_type = 'continuous' if continuous else 'discrete'
        print_txt += f'fairness adjustments for {var_type} variables.'
        print(print_txt)

    # Check if materially relevant features should be treated as discrete
    if optp_fair_add.no_discretization(material_np, fair_cfg.material_max_groups):
        fair_cfg.material_disc_method = 'ExactCell'
        txt_report += ('\nMaterially relevant features have no or only a few values. '
                       'Exact cells are used without coarsening.'
                       )
    # Check if protected features should be treated as discrete
    if optp_fair_add.no_discretization(protect_np, fair_cfg.protected_max_groups):
        fair_cfg.protected_disc_method = 'ExactCell'
        txt_report += ('\nProtected features have no or only a few values. '
                       'Exact cells are used without coarsening.'
                       )
    # Discretize if needed, otherwise no change of data
    protect_np, material_np, txt_add = optp_fair_add.data_quantilized(fair_cfg, protect_np,
                                                                      material_np, seed
                                                                      )
    txt_report += txt_add
    # Get clusternumber for later use in Policy Trees
    if strata_number_fair:
        strata_fair_np, txt_add = cluster_number_within_cell_quantilization(protect_np, material_np)
        txt_report += txt_add
    else:
        strata_fair_np = None

    if ((fair_cfg.protected_disc_method in disc_methods)
            and ((material_np is None) or (fair_cfg.material_disc_method in disc_methods))):
        # No density estimation needed
        xvars_fair_np, txt_add = within_cell_quantilization(xvars_np, protect_np, material_np)
    else:
        xvars_fair_np, txt_add = kernel_quantilization(fair_cfg, xvars_np, protect_np, material_np)

    txt_report += txt_add
    if gen_cfg.with_output and title == '':
        mcf_ps.print_mcf(gen_cfg, txt_report, summary=True)

    (fair_cfg.protected_disc_method, fair_cfg.material_disc_method
     ) = requested_disc_methods
    return xvars_fair_np, strata_fair_np, txt_report, fair_cfg


def kernel_quantilization(fair_cfg: dict,
                          xvars_np: NDArray[Any],
                          protected_np: NDArray[Any],
                          material_np: NDArray[Any] | None,
                          ) -> tuple[NDArray[Any], str]:
    """Do within cell quantilization for arbitrary materially rel. features."""
    disc_methods = fair_cfg.discretization_methods
    txt_report = ('\nQuantile based method by Strack & Yang (2024) used for '
                  'materially relevant features with many variables.'
                  )
    no_of_xvars = xvars_np.shape[1]
    xvars_fair_np = xvars_np.copy()
    no_eval_point = 2000

    # Case of discrete protected and discrete materially relevant is dealt with
    # in other procedure

    if fair_cfg.protected_disc_method:
        vals_prot = np.unique(protected_np, return_counts=False)
    else:
        vals_prot = None

    if material_np is not None and fair_cfg.material_disc_method:
        vals_material = np.unique(material_np, return_counts=False)
    else:
        vals_material = None

    if protected_np.shape[1] == 1:
        protected_np = protected_np.reshape(-1, 1)
    if material_np is not None and material_np.shape[1] == 1:
        material_np = material_np.reshape(-1, 1)

    for idx_xvars in range(no_of_xvars):
        xvars = xvars_np[:, idx_xvars]
        if np.std(xvars) < 1e-8:
            continue
        quantiles_z = np.zeros_like(xvars)

        # Kernel density estimation conditional on protected variables
        xvars_all_grid = get_grid(xvars, no_eval_point)
        if fair_cfg.material_disc_method in disc_methods:
            if material_np is None:
                quantiles_z, _ = calculate_quantiles_kde(
                    xvars, protected_np, xvars_all_grid)
            else:
                for val in vals_material:
                    mask = material_np.reshape(-1) == val
                    xvars_grid = get_grid(xvars[mask], no_eval_point)
                    quantiles_z[mask], _ = calculate_quantiles_kde(
                        xvars[mask], protected_np[mask], xvars_grid)
        elif fair_cfg.protected_disc_method in disc_methods:
            for val in vals_prot:
                mask = protected_np.reshape(-1) == val
                # Find quantile in conditional data
                if material_np is None:
                    data_cond_np = None
                else:
                    data_cond_np = material_np[mask].copy()
                xvars_grid = get_grid(xvars[mask], no_eval_point)
                quantiles_z[mask], _ = calculate_quantiles_kde(
                    xvars[mask], data_cond_np, xvars_grid)

        else:  # Both groups of variables treated as continuous
            if material_np is None:
                data_cond_np = protected_np
            else:
                data_cond_np = np.concatenate((protected_np, material_np), axis=1)
            quantiles_z, _ = calculate_quantiles_kde(xvars, data_cond_np, xvars_all_grid)

        # Translate quantile to values of distribution conditional on
        # materially relevant variables only
        if fair_cfg.material_disc_method in disc_methods:
            if material_np is None:
                xvars_fair_np[:, idx_xvars] = values_from_quantiles(
                    xvars, quantiles_z)
            else:
                for val in vals_material:
                    mask = material_np.reshape(-1) == val
                    xvars_fair_np[mask, idx_xvars] = values_from_quantiles(xvars[mask],
                                                                           quantiles_z[mask]
                                                                           )
        else:
            _, xvars_fair_np[:, idx_xvars] = calculate_quantiles_kde(xvars, material_np,
                                                                     xvars_all_grid,
                                                                     quantile_data=quantiles_z
                                                                     )
    return xvars_fair_np, txt_report


def calculate_quantiles_kde(xvars: NDArray[Any],
                            data_cond: NDArray[Any] | None,
                            xvars_grid: NDArray[Any],
                            quantile_data: NDArray[Any] | None = None,
                            ) -> tuple[NDArray[Any] | None, NDArray[Any] | None]:
    """Integrate the Epanechnikov KDE over the grid bounds, without grid approximation."""
    xvars = np.asarray(xvars, dtype=float).reshape(-1)
    grid = np.asarray(xvars_grid, dtype=float).reshape(-1)
    if xvars.size == 0 or not np.all(np.isfinite(xvars)):
        raise ValueError('xvars must contain finite observations.')
    if len(grid) < 2 or not np.all(np.isfinite(grid)) or grid[0] >= grid[-1]:
        raise ValueError('xvars_grid must have finite, increasing bounds.')
    lower, upper = float(grid[0]), float(grid[-1])

    if data_cond is None:
        condition = np.empty((len(xvars), 0))
    else:
        condition = np.asarray(data_cond, dtype=float)
        if condition.ndim == 1:
            condition = condition.reshape(-1, 1)
        if condition.ndim != 2 or len(condition) != len(xvars):
            raise ValueError('data_cond must contain one row per xvars value.')
        if not np.all(np.isfinite(condition)):
            raise ValueError('data_cond must contain only finite values.')

    quantile_values = quantile_data is not None
    if quantile_values:
        quantile_data = np.asarray(quantile_data, dtype=float).reshape(-1)
        if len(quantile_data) != len(xvars):
            raise ValueError('quantile_data and xvars must have equal length.')
        if not np.all(np.isfinite(quantile_data)) or np.any((quantile_data < 0)
                                                          | (quantile_data > 1)):
            raise ValueError('quantile_data must contain finite values between 0 and 1.')

    joint_data = np.column_stack((condition, xvars))
    kde = KernelDensity(kernel='epanechnikov', bandwidth='silverman').fit(joint_data)
    bandwidth = float(kde.bandwidth_)
    result = np.empty(len(xvars), dtype=float)

    def cdf_at(point: float,
               centers: NDArray[Any],
               radii: NDArray[Any],
               weights: NDArray[Any],
               ) -> float:
        """Integrate the conditional mixture up to point."""
        with np.errstate(over='ignore'):
            position = np.clip((point - centers) / radii, -1, 1)
        unit = (position + 1) / 2
        return float(weights @ (unit * unit * (3 - 2 * unit)))

    for idx, value in enumerate(xvars):
        if condition.shape[1]:
            with np.errstate(over='ignore', invalid='ignore'):
                distance_sq = np.sum(((condition - condition[idx]) / bandwidth) ** 2, axis=1)
            support = np.maximum(1 - distance_sq, 0)
        else:
            support = np.ones(len(xvars))
        active = support > 0
        centers = xvars[active]
        radii = bandwidth * np.sqrt(support[active])
        weights = support[active] ** 1.5
        weights /= weights.sum()

        # At a fixed condition, each spherical kernel becomes a weighted 1D kernel.
        cdf_lower = cdf_at(lower, centers, radii, weights)
        mass = cdf_at(upper, centers, radii, weights) - cdf_lower
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError('The KDE has no positive probability within the evaluation bounds.')

        if not quantile_values:
            cumulative = cdf_at(float(value), centers, radii, weights)
            result[idx] = np.clip((cumulative - cdf_lower) / mass, 0, 1)
            continue

        quantile = float(quantile_data[idx])
        if quantile in (0, 1):
            result[idx] = lower if quantile == 0 else upper
            continue
        target = cdf_lower + quantile * mass
        left, right = lower, upper
        for _ in range(64):
            midpoint = left / 2 + right / 2
            if midpoint in (left, right):
                break
            if cdf_at(midpoint, centers, radii, weights) < target:
                left = midpoint
            else:
                right = midpoint
        result[idx] = right

    return (None, result) if quantile_values else (result, None)


def get_grid(data: np.array, no_eval_point: int | float) -> NDArray[Any]:
    """Get evaluation grid for densities."""
    lower = float(np.min(data))
    upper = float(np.max(data))

    if np.isclose(lower, upper, rtol=0, atol=1e-12):
        padding = max(abs(lower), 1) * 1e-6
        lower -= padding
        upper += padding

    grid = np.linspace(lower, upper, no_eval_point).reshape(-1, 1)

    return grid


def cluster_number_within_cell_quantilization(
        protected_np: NDArray[Any],
        material_np: NDArray[Any] | None,
        ) -> tuple[NDArray[Any], str]:
    """Do within cell quantilization for discrete univariate features."""
    if material_np is None:
        material_np = np.ones((protected_np.shape[0], 1))
    material_values = np.unique(material_np, return_counts=False)
    cluster_fair_np = np.zeros(len(protected_np), dtype=np.int64)
    cluster_no = 0
    for mat_val in material_values:
        indices_mat = np.where(material_np == mat_val)[0]
        if indices_mat.size == 0:  # Empty cluster
            continue

        prot_mat = protected_np[indices_mat].reshape(-1)
        vals_prot_mat = np.unique(prot_mat, return_counts=False)
        if len(vals_prot_mat) == 1:
            cluster_fair_np[indices_mat] = cluster_no
            cluster_no += 1
            continue
        for val in vals_prot_mat:
            mask = prot_mat == val
            index_mask = indices_mat[mask]
            cluster_fair_np[index_mask] = cluster_no
            cluster_no += 1

    txt_report = f'\nNumber of clusters (protected x materially relevant): {cluster_no}'

    return cluster_fair_np, txt_report


def within_cell_quantilization(xvars_np: NDArray[Any],
                               protected_np: NDArray[Any],
                               material_np: NDArray[Any] | None,
                               ) -> tuple[NDArray[Any], str]:
    """Do within cell quantilization for discrete univariate features."""
    txt_report = ('\nQuantile based method by Strack & Yang (2024) used for '
                  'discrete features.')
    no_of_xvars = xvars_np.shape[1]
    if material_np is None:
        material_np = np.ones((protected_np.shape[0], 1))
    material_values = np.unique(material_np, return_counts=False)
    xvars_fair_np = xvars_np.copy()
    for mat_val in material_values:
        indices_mat = np.where(material_np == mat_val)[0]
        if indices_mat.size == 0:
            continue

        prot_mat = protected_np[indices_mat].reshape(-1)
        vals_prot_mat = np.unique(prot_mat, return_counts=False)
        if len(vals_prot_mat) == 1:
            continue

        for idx_xvars in range(no_of_xvars):
            xvars_mat = xvars_np[indices_mat, idx_xvars]
            if np.std(xvars_mat) < 1e-8:  # No variation -> original value
                continue
            quantiles = np.zeros_like(xvars_mat)
            for val in vals_prot_mat:
                mask = prot_mat == val
                # Find quantile in conditional data
                quantiles[mask] = calculate_quantiles(xvars_mat[mask])
                # Translate quantile to values of distribution conditional on
                # materially relevant variables
            xvars_fair_np[indices_mat, idx_xvars] = values_from_quantiles(xvars_mat, quantiles)

    return xvars_fair_np, txt_report


def calculate_quantiles(data: NDArray[Any]) -> NDArray[Any]:
    """Calculate quantiles for each value in the dataset."""
    data_sort = np.sort(data)
    rank = np.searchsorted(data_sort, data, side='right')

    return rank / len(data)


def values_from_quantiles(data: NDArray[Any],
                          quantiles: NDArray[Any]
                          ) -> NDArray[Any]:
    """Get the values from the quantiles."""
    d_sorted = np.sort(data)  # Sort the empirical distribution
    obs = len(d_sorted)       # Number of data points
    cdf = np.arange(1, obs + 1) / obs
    indices = np.searchsorted(cdf, quantiles, side='left')

    return d_sorted[indices]


def residualisation(fair_cfg: 'FairCfg',
                    gen_cfg: 'GenCfg',
                    xvars_np: NDArray[Any],
                    xvars_name: list | tuple,
                    protect_np: NDArray[Any], *,
                    material_np: NDArray[Any] | None,
                    seed: int = 1246546,
                    title: str = '',
                    ) -> tuple[NDArray[Any], str]:
    """Adjust by residualisation."""
    if not np.issubdtype(xvars_np.dtype, np.floating):
        xvars_np = np.asarray(xvars_np, dtype=float)
    # Info and tuning parameters
    obs, no_of_xvars = xvars_np.shape
    boot = 1000
    cross_validation_k = min(5, obs)
    txt_report = ''

    # Define all conditioning variables for regressions below
    if material_np is None:
        x_cond_np = protect_np
        with_material_x = False
    else:
        x_cond_np = np.concatenate((protect_np, material_np), axis=1)
        with_material_x = True
    # Submethod: Adjust only mean, or mean and variance
    if fair_cfg.adj_type == 'Mean':
        adjustment_set = ('mean', )
    else:
        adjustment_set = ('mean', 'variance')

    # Numpy arrays to save conditional means and variance for each variable
    y_mean_cond_x_np = np.zeros_like(xvars_np)
    if with_material_x:
        y_mean_cond_mat_np = np.zeros_like(xvars_np)
    if fair_cfg.adj_type == 'MeanVar':
        y_var_cond_x_np = np.zeros_like(xvars_np)
        if with_material_x:
            y_var_cond_mat_np = np.zeros_like(xvars_np)
        else:
            y_var_cond_mat_np = None
    else:
        y_var_cond_x_np = y_var_cond_mat_np = None

    # Loop over variables to obtain prediction of conditonal expectation of y
    for idx in range(no_of_xvars):
        for mean_var in adjustment_set:
            if gen_cfg.with_output:
                print('\n' + title + f'Currently adjusting {mean_var} of {xvars_name[idx]}')

            # Define dependent variable in regression & check if regr. is needed

            match mean_var:
                case 'mean':
                    # Adjust conditional mean by residualisation
                    y_np = xvars_np[:, idx]  # Dependent variable in regression
                    # No regression if there is no variation in the variable
                    if np.std(y_np) < 1e-8:
                        y_mean_cond_x_np[:, idx] = y_np.copy()
                        if with_material_x:
                            y_mean_cond_mat_np[:, idx] = y_np.copy()
                        continue
                case 'variance':
                    # Adjust conditional variance by rescaling
                    # Law of total variance: Var(Y|X)=E(Y**2|X) - (EY|X)**2
                    y_np = xvars_np[:, idx]**2
                    y_mean_x_2 = y_mean_cond_x_np[:, idx]**2
                    if with_material_x:
                        y_mean_mat_2 = y_mean_cond_mat_np[:, idx]**2
                    # No regression if there is no variation in the variable
                    if np.std(y_np) < 1e-8:
                        y_var_cond_x_np[:, idx] = y_np - y_mean_x_2
                        if with_material_x:
                            y_var_cond_mat_np[:, idx] = y_np - y_mean_mat_2
                        continue
                case _:
                    raise ValueError('Wrong adjustement method.')

            # Find best estimator for specific variable (only using all covars)
            (estimator, params, best_label, _, transform_x, txt_mse) = mcf_gf.best_regression(
                x_cond_np, y_np.ravel(),
                estimator=fair_cfg.regression_method,
                boot=boot,
                seed=seed + 12435,
                max_workers=gen_cfg.mp_parallel,
                cross_validation_k=cross_validation_k,
                absolute_values_pred=mean_var == 'variance')

            if with_material_x:
                (estimator_m, params_m, best_label_m, _, transform_x_m, txt_mse_m
                 ) = mcf_gf.best_regression(
                    material_np,  y_np.ravel(),
                    estimator=fair_cfg.regression_method,
                    boot=boot,
                    seed=seed + 12435,
                    max_workers=gen_cfg.mp_parallel,
                    cross_validation_k=cross_validation_k,
                    absolute_values_pred=mean_var == 'variance')

            if gen_cfg.with_output:
                text = '\n' + title + f'Adjustment for {mean_var} of {xvars_name[idx]}:'
                txt_mse = text + txt_mse
                if with_material_x:
                    txt_mse += '\n' + title + 'Short regression:' + txt_mse_m
                mcf_ps.print_mcf(gen_cfg, txt_mse, summary=False)
                if mean_var == 'mean':
                    txt_report += '\n'
                txt_report += text + ' by ' + best_label
                if with_material_x:
                    txt_report += ('\n' + title + 'Short regression adjusted '
                                   + 'by ' + best_label_m
                                   )
            # Obtain out-of-sample prediction by k-fold cross-validation
            index = np.arange(obs)      # indices
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(index)
            index_folds = np.array_split(index, cross_validation_k)
            for fold_pred in range(cross_validation_k):
                fold_train = [x for indx, x in enumerate(index_folds)
                              if indx != fold_pred]
                index_train = np.hstack(fold_train)
                index_pred = index_folds[fold_pred]
                if transform_x:
                    _, x_train, x_pred = mcf_gf.scale(x_cond_np[index_train], x_cond_np[index_pred])
                else:
                    x_train = x_cond_np[index_train]
                    x_pred = x_cond_np[index_pred]
                if with_material_x:
                    if transform_x_m:
                        _, x_train_m, x_pred_m = mcf_gf.scale(material_np[index_train],
                                                              material_np[index_pred]
                                                              )
                    else:
                        x_train_m = material_np[index_train]
                        x_pred_m = material_np[index_pred]

                y_train = y_np[index_train].ravel()
                y_obj = mcf_gf.regress_instance(estimator, params)
                if y_obj is None:
                    if mean_var == 'mean':
                        y_mean_cond_x_np[index_pred, idx] = np.average(y_train)
                    else:
                        y_var_cond_x_np[index_pred, idx] = (np.average(y_train)
                                                            - y_mean_x_2[index_pred]
                                                            )
                else:
                    y_obj.fit(x_train, y_train)
                    if mean_var == 'mean':
                        y_mean_cond_x_np[index_pred, idx] = y_obj.predict(x_pred
                                                                          )
                    else:
                        y_var_cond_x_np[index_pred, idx] = (y_obj.predict(x_pred)
                                                            - y_mean_x_2[index_pred]
                                                            )
                if with_material_x:
                    y_obj_m = mcf_gf.regress_instance(estimator_m, params_m)
                    if y_obj_m is None:
                        if mean_var == 'mean':
                            y_mean_cond_mat_np[index_pred, idx] = np.average(y_train)
                        else:
                            y_var_cond_mat_np[index_pred, idx] = (np.average(y_train)
                                                                  - y_mean_mat_2[index_pred]
                                                                  )
                    else:
                        y_obj_m.fit(x_train_m, y_train)
                        if mean_var == 'mean':
                            y_mean_cond_mat_np[index_pred, idx] = y_obj_m.predict(x_pred_m)
                        else:
                            y_var_cond_mat_np[index_pred, idx] = (y_obj_m.predict(x_pred_m)
                                                                  - y_mean_mat_2[index_pred]
                                                                  )
    residuum_np = xvars_np - y_mean_cond_x_np

    # Adjust variance as well
    if fair_cfg.adj_type == 'MeanVar':

        # Conditional standard deviation (must be non-zero)
        bound_var = 1e-6
        y_std_cond_x_np = optp_fair_add.var_to_std(y_var_cond_x_np, bound_var)
        if with_material_x:
            y_std_cond_mat_np = optp_fair_add.var_to_std(y_var_cond_mat_np, bound_var)

        # Avoid too extreme values when scaling
        bound = 0.05
        y_std_cond_x_np = optp_fair_add.bound_std(y_std_cond_x_np, bound, no_of_xvars)

        # Remove predictability due to heteroscedasticity
        residuum_np /= y_std_cond_x_np

        # Rescale to retain about the variability of the original variabless
        std_resid = np.std(residuum_np, axis=0).reshape(-1, 1).T
        np.divide(residuum_np, std_resid, out=residuum_np, where=std_resid > 0)
        if with_material_x:
            residuum_np *= y_std_cond_mat_np
        else:
            residuum_np *= np.mean(
                y_std_cond_x_np, axis=0).reshape(-1, 1).T

    # Correct variables, but keep variable specific (conditional) mean
    if with_material_x:
        xvars_fair_np = residuum_np + y_mean_cond_mat_np
    else:
        xvars_fair_np = residuum_np + np.mean(
            y_mean_cond_x_np, axis=0).reshape(-1, 1).T

    return xvars_fair_np, txt_report


def test_for_consistency(fair_cfg: 'FairCfg',
                         gen_cfg: 'GenCfg',
                         var_cfg: 'VarCfg',
                         data_fair_df: pd.DataFrame,
                         score_fair_name: list | tuple, *,
                         seed: int = 124567,
                         title: str = 'Consistency test',
                         ) -> tuple[dict, str]:
    """Test for consistency.

    Compare difference of fair scores to difference made explicitly fair.
    Maybe use in descriptive part.
    Also define flag because it is computationally expensive since additional
    scores have to be made fair.
    """
    score_name, test_dic = var_cfg.polscore_name, {}
    score_fair_np = data_fair_df[score_fair_name].to_numpy(copy=True)
    scores_np, protect_np, material_np = vars_to_numpy(data_fair_df, var_cfg.polscore_name,
                                                       var_cfg.protected_name, var_cfg.material_name
                                                       )
    # Find valid combinations of scores that can be tested
    test_scores_np, test_fair_scores_np, test_scores_name, _ = optp_fair_add.score_combinations(
        score_fair_np, score_fair_name, scores_np, score_name
        )
    # Fairness adjust the score differences
    test_scores_adj_np, _, _, _ = fair_continuous_xvars_fct(fair_cfg, gen_cfg, test_scores_np,
                                                            test_scores_name, protect_np,
                                                            material_np=material_np, seed=seed,
                                                            title=title, strata_number_fair=False,
                                                            )
    test_scores_adj_name = [name + '_adj' for name in test_scores_name]

    test_diff_np = (np.mean(np.abs(test_fair_scores_np - test_scores_adj_np), axis=0)
                    / np.std(test_scores_np, axis=0)
                    )
    same_sign_np = np.mean(
        np.sign(test_fair_scores_np) == np.sign(test_scores_adj_np), axis=0)

    for idx, name in enumerate(test_scores_adj_name):
        correlation = np.corrcoef(test_fair_scores_np[:, idx], test_scores_adj_np[:, idx])[0, 1]
        test_dic[name] = [test_diff_np[idx], same_sign_np[idx], correlation]

    txt = ''
    if gen_cfg.with_output:
        txt1 = ('\nTest for consistency of different fairness normalisations:'
                '\n    - Compare difference of adjusted scores to adjusted difference of scores'
                )
        for key, value in test_dic.items():
            keys = key + ':'
            txt1 += (f'\n{keys:50} MAD: {value[0]:5.2%}, '
                     f'Same sign: {value[1]:5.2%}, Correlation: {value[2]:5.2%}'
                     )
        txt1 += (
            '\n\nMAD: Mean absolute differences of scores in % of standard '
            'deviation of absolute unadjusted score. Ideal value is 0%.'
            '\nSame sign: Share of scores with same sign. Ideal value is 100%.'
            '\nCorrelation: Share of scores with same sign. Ideal value is 100%.'
            )
        mcf_ps.print_mcf(gen_cfg, '\n' + '-' * 100 + txt + txt1, summary=True)
        txt += '\n' + txt1

    return test_dic, txt


def update_names_for_solve_in_self(var_cfg: 'VarCfg',
                                   fair_scores_names: list,
                                   fair_x_names_ord: list,
                                   fair_x_names_unord: list,
                                   ) -> dict[str, list]:
    """Update variables to be used in solve procedure."""
    # Remember original variables
    var_cfg = deepcopy(var_cfg)
    org_name_dic = {'polscore_name': var_cfg.polscore_name.copy(),
                    'x_ord_name': var_cfg.x_ord_name.copy(),
                    'x_unord_name': var_cfg.x_unord_name.copy()
                    }
    var_cfg.polscore_name = fair_scores_names
    var_cfg.x_ord_name = fair_x_names_ord
    var_cfg.x_unord_name = fair_x_names_unord

    return org_name_dic, var_cfg


def _align_prediction_dummies(data_df: pd.DataFrame,
                              unordered_names: list[str],
                              ordered_names: list[str],
                              training_names: list[str],
                              ) -> pd.DataFrame:
    """Align prediction dummies to dummy columns used in fairness training."""
    if not unordered_names:
        return data_df

    dummies_df = pd.get_dummies(data_df[unordered_names], columns=unordered_names, dtype=int)
    expected = [name for name in training_names if name not in ordered_names]
    unexpected = [name for name in dummies_df.columns if name not in expected]

    if unexpected:
        raise ValueError('Prediction data contain categorical values that were not observed '
                         f'during fairness training: {" ".join(unexpected)}'
                         )
    dummies_df = dummies_df.reindex(columns=expected, fill_value=0)

    for name in expected:
        data_df[name] = dummies_df[name].to_numpy(copy=False)

    return data_df


def fair_adjust_data_for_pred(fair_cfg: 'FairCfg',
                              gen_cfg: 'GenCfg',
                              var_cfg: 'VarCfg',
                              data_df: pd.DataFrame,
                              fair_adjust_decision_vars: bool,
                              ) -> tuple[pd.DataFrame, 'FairCfg', str]:
    """Add fairness-transformed data if needed for prediction."""
    txt = ''
    # Remove some information from training that will be updated
    if fair_cfg.fair_strata is not None:
        fair_cfg.fair_strata = None
    if fair_cfg.decision_vars_org_df is not None:
        fair_cfg.decision_vars_org_df = None
    if fair_cfg.decision_vars_fair_org_name is not None:
        fair_cfg.decision_vars_fair_org_name = None
    if fair_cfg.x_ord_org_name is not None:
        fair_cfg.x_ord_org_name = None

    fair_adjustvar = (gen_cfg.method != 'best_policy_score'
                      and fair_cfg.adjust_target in ('xvariables', 'scores_xvariables')
                      )
    fair_decision_var_na = any(elem not in data_df.columns
                               for elem in [*var_cfg.x_ord_name, *var_cfg.x_unord_name]
                               )
    fair_adjust_scores = (gen_cfg.method == 'best_policy_score'
                          and fair_cfg.adjust_target in ('scores', 'scores_xvariables')
                          )
    fair_score_na = any(name not in data_df.columns for name in var_cfg.polscore_name)
    needs_adjustment = (fair_adjust_decision_vars or (fair_adjustvar and fair_decision_var_na)
                        or (fair_adjust_scores and fair_score_na)
                        )
    if (needs_adjustment and var_cfg.protected_unord_name
            and any(name not in data_df.columns for name in var_cfg.protected_name)):
        data_df = _align_prediction_dummies(data_df,
                                            var_cfg.protected_unord_name,
                                            var_cfg.protected_ord_name,
                                            var_cfg.protected_name,
                                            )
    if (needs_adjustment and var_cfg.material_unord_name
            and any(name not in data_df.columns for name in var_cfg.material_name)):
        data_df = _align_prediction_dummies(data_df,
                                            var_cfg.material_unord_name,
                                            var_cfg.material_ord_name,
                                            var_cfg.material_name,
                                            )

    if fair_adjust_scores and fair_score_na:
        original_score_names = fair_cfg.org_name_dict['polscore_name']

        data_df, score_fair_names, fair_cfg, txt_scores = fair_adjust_cont_disc(
            deepcopy(fair_cfg),
            deepcopy(gen_cfg),
            deepcopy(var_cfg),
            data_df,
            original_score_names,
            x_are_continous=True,
            strata_number_fair=False,
            seed=1234567,
            with_output=True,
            decision_variable_adjust=False,
            )

        if score_fair_names != var_cfg.polscore_name:
            raise RuntimeError('Names of fairness-adjusted prediction scores are inconsistent '
                               'with the scores used for training.'
                               )
        txt += ('\nFairness-adjusted policy scores were not available in the '
                'prediction data. They were therefore computed from the original policy scores.'
                )
        txt += txt_scores

    if fair_adjustvar and fair_decision_var_na:
        # If adjusted variables are not available,they must be created and added to the data
        # Add explanations to report file
        txt += ('\n\nFairness adjusted variables were not available in the '
                'data provided to compute allocations (prediction data). '
                'Therefore, they are computed with the prediction data.'
                )
        fair_adjust_decision_vars = True

    if fair_adjust_decision_vars:
        data_train_df = fair_cfg.data_train_for_pred_df
        add_train_data = len(data_train_df) > len(data_df) * 2
        if add_train_data:
            txt += ('\nData used for training is more than twice as large as '
                    'data to be used for prediction. Therefore, for computing '
                    'variable fairness adjustments, training data will be '
                    'added to the prediction data.'
                    )
            data_temp_df = pd.concat((data_df[data_train_df.columns], data_train_df,), axis=0
                                     ).reset_index(drop=True)
        else:
            data_temp_df = data_df

        var_cfg_temp = deepcopy(var_cfg)
        var_cfg_temp.x_ord_name = fair_cfg.org_name_dict['x_ord_name']
        var_cfg_temp.x_unord_name = fair_cfg.org_name_dict['x_unord_name']

        data_temp_df, _, _, _, fair_cfg, _, = adjust_decision_variables(
            deepcopy(fair_cfg), deepcopy(gen_cfg), var_cfg_temp, data_temp_df.copy(),
            training=False, seed=1234567,
            )
        if add_train_data:
            data_pred_df = data_temp_df.iloc[:len(data_df), :]
            data_pred_rel_df = data_pred_df[[*var_cfg.x_ord_name, *var_cfg.x_unord_name]]
            data_df = data_df.reset_index(drop=True).copy()
            data_df[data_pred_rel_df.columns] = data_pred_rel_df.to_numpy(copy=False)
        else:
            data_df = data_temp_df

    return data_df, fair_cfg, txt
