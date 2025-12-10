"""Created on Wed May  1 16:35:19 2024.

Functions for correcting variables w.r.t. protected variables.

@author: MLechner

# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d

from mcf import mcf_estimation_generic_functions as mcf_gf
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import optpolicy_fair_add_functions as optp_fair_add


def adjust_decision_variables(
        fair_cfg: Any,
        gen_cfg: Any,
        var_cfg: Any,
        data_df: pd.DataFrame,
        training: bool = True,
        seed: int = 1246546
        ) -> tuple[pd.DataFrame, list[str], dict, dict, str]:
    """Make decision var's (conditionally) independent of protected features."""
    # 1.categorize decision variables into groups whether to be treated as
    # continuous or unordered
    (x_cont_name, x_disc_name, fair_info_dict
     ) = optp_fair_add.cont_or_discrete(data_df,
                                        var_cfg.x_ord_name,
                                        var_cfg.x_unord_name,
                                        fair_cfg.cont_min_values
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
                          and fair_cfg.adjust_target in ('xvariables',
                                                         'scores_xvariables',)
                          )
    if x_cont_name:
        (data_fair_df, x_fair_cont_name, fair_cfg, txt_report
         ) = fair_adjust_cont_disc(fair_cfg, gen_cfg, var_cfg, data_fair_df,
                                   x_cont_name, x_are_continous=True, seed=seed,
                                   with_output=True,
                                   strata_number_fair=strata_number_fair,
                                   decision_variable_adjust=True
                                   )
        for name_idx, name in enumerate(x_cont_name):
            if name in var_cfg.x_ord_name:
                x_fair_ord_name.append(x_fair_cont_name[name_idx])

    if x_disc_name:
        (data_fair_df, x_fair_disc_name, fair_cfg, txt_report
         ) = fair_adjust_cont_disc(fair_cfg, gen_cfg, var_cfg, data_fair_df,
                                   x_disc_name, x_are_continous=False,
                                   seed=seed, with_output=True,
                                   strata_number_fair=strata_number_fair,
                                   decision_variable_adjust=True
                                   )
        for name_idx, name in enumerate(x_disc_name):
            if name in var_cfg.x_ord_name:
                x_fair_ord_name.append(x_fair_disc_name[name_idx])
            else:
                x_fair_unord_name.append(x_fair_disc_name[name_idx])

    if training and fair_cfg.data_train_for_pred_df is None:
        x_name = [*var_cfg.x_ord_name, *var_cfg.x_unord_name,
                  *var_cfg.protected_name, *var_cfg.material_name
                  ]
        x_name_extend = [var for var in
                         (*var_cfg.material_unord_name,
                          *var_cfg.protected_unord_name
                          )
                         if var not in x_name
                         ]
        if x_name_extend:
            x_name.extend(x_name_extend)

        fair_cfg.data_train_for_pred_df = data_fair_df[x_name]
    if strata_number_fair:
        fair_cfg.protected_matrel = data_df[var_cfg.prot_mat_no_dummy_name]

    return (data_fair_df, x_fair_ord_name, x_fair_unord_name,
            fair_info_dict, fair_cfg, txt_report
            )


def adjust_scores(fair_cfg: Any,
                  gen_cfg: Any,
                  var_cfg: Any,
                  data_df: pd.DataFrame,
                  seed: int = 1246546,
                  ) -> tuple[pd.DataFrame, list[str], dict, dict, dict, str]:
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
        tests_dict, text = test_for_consistency(
            fair_cfg, gen_cfg, var_cfg,
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
    var_cfg = optp_fair_add.change_variable_names_fair(var_cfg.score_fair_name)

    return (data_fair_df, score_fair_name, tests_dict, fair_cfg, var_cfg,
            txt_report
            )


def fair_adjust_cont_disc(fair_cfg: Any,  # Rename
                          gen_cfg: Any,
                          var_cfg: Any,
                          data_df: pd.DataFrame,
                          x_name: list | tuple,
                          x_are_continous: bool = True,
                          strata_number_fair: bool = False,
                          seed: int = 1246546,
                          with_output: bool = True,
                          decision_variable_adjust: bool = False,
                          ) -> tuple[pd.DataFrame, list[str], dict, str]:
    """Fairness-adjust continuous and discrete variables."""
    fair_cfg = deepcopy(fair_cfg)

    x_np, protect_np, material_np = vars_to_numpy(
        data_df, x_name, var_cfg.protected_name, var_cfg.material_name
        )
    txt_report = fair_method_message(gen_cfg, fair_cfg.adj_type,
                                     continuous_vars=x_are_continous
                                     )
    # Get fairness adjusted variables
    if x_are_continous:
        x_fair_np, cluster_fair_np, fair_cfg, txt = fair_continuous_xvars_fct(
            fair_cfg, gen_cfg, x_np, x_name, protect_np, material_np, seed=seed,
            title='', strata_number_fair=strata_number_fair,
            )
    else:
        if fair_cfg.adj_type != 'Quantiled':
            txt = ('\nAdjustment method for discrete variables changed to '
                   "'Quantiled'."
                   )
            fair_cfg.adj_type = 'Quantiled'
        x_fair_np, cluster_fair_np, fair_cfg, txt = fair_discrete_xvars_fct(
            fair_cfg, gen_cfg, x_np, protect_np, material_np, seed=seed,
            title='', strata_number_fair=strata_number_fair,
            )
    txt_report += txt

    # Convert numpy array of fair variable to pandas dataframe and add
    data_fair_df, x_fair_name, fair_strata_df, x_org_df = update_fair_df(
        data_df, x_name, x_fair_np, cluster_fair_np,
        )

    if gen_cfg.with_output and with_output:
        txt_report += optp_fair_add.fair_stats(
            gen_cfg, var_cfg, data_fair_df, x_name, x_fair_name,
            var_continuous=x_are_continous
            )

    if decision_variable_adjust:
        fair_cfg.fair_strata = fair_strata_df

        if fair_cfg.decision_vars_org_df is None:
            fair_cfg.decision_vars_org_df = x_org_df
        else:
            fair_cfg.decision_vars_org_df = pd.concat(
                (fair_cfg.decision_vars_org_df, x_org_df), axis=1
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

    return data_fair_df, x_fair_name, fair_cfg, txt_report


def update_fair_df(data_df: pd.DataFrame,
                   x_name: list,
                   x_fair_np: NDArray[Any],
                   cluster_fair_np: NDArray[Any],
                   ) -> tuple[pd.DataFrame, list, pd.DataFrame]:
    """Update dataframe and names."""
    x_fair_name = [name + '_fair' for name in x_name]
    x_fair_df = pd.DataFrame(x_fair_np, columns=x_fair_name)
    if cluster_fair_np is None:
        fair_strata_df = x_org_df = None
    else:
        fair_strata_df = pd.DataFrame(cluster_fair_np, columns=('cluster',))
        x_org_df = data_df[x_name]

    x_org_df = None if x_name is None else data_df[x_name]

    data_df = data_df.reset_index(drop=True)
    data_fair_df = pd.concat((data_df, x_fair_df), axis=1)

    return data_fair_df, x_fair_name, fair_strata_df, x_org_df


def fair_method_message(gen_cfg: Any,
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


def vars_to_numpy(
        data_df: pd.DataFrame,
        vars_name: list[str] | tuple[str, ...] | None,
        protected_name: list[str] | tuple[str, ...] | None,
        material_name:  list[str] | tuple[str, ...] | None,
        ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Transfer some dataframes to numpy."""
    vars_np = data_df[vars_name].to_numpy() if vars_name else None
    protect_np = data_df[protected_name].to_numpy() if protected_name else None
    material_np = data_df[material_name].to_numpy() if material_name else None

    return vars_np, protect_np, material_np


def fair_continuous_xvars_fct(
        fair_cfg: Any,
        gen_cfg: Any,
        xvars_np: NDArray[Any],
        xvars_name: list,
        protect_np: NDArray[Any],
        material_np: NDArray[Any] | None,
        strata_number_fair: bool = False,
        seed: int = 1234567,
        title: str = '',
        ) -> tuple[NDArray[Any], NDArray[Any], str, dict]:
    """Use one of the different fairness adjustment methods."""
    adj = getattr(fair_cfg, 'adj_type')
    match adj:
        case 'Mean' | 'MeanVar':
            xvars_fair_np, txt = residualisation(
                fair_cfg, gen_cfg, xvars_np, xvars_name, protect_np,
                material_np, seed=seed, title=title
                )
            cluster_fair_np = None
        case 'Quantiled':
            xvars_fair_np, cluster_fair_np, txt, fair_cfg = quantalisation(
                fair_cfg, gen_cfg, xvars_np, protect_np, material_np, seed=seed,
                title=title, continuous=True,
                strata_number_fair=strata_number_fair
                )
        case _:
            raise ValueError('Invalid method selected for fairness adjustment.')

    return xvars_fair_np, cluster_fair_np, fair_cfg, txt


def fair_discrete_xvars_fct(
        fair_cfg: Any,
        gen_cfg: Any,
        xvars_np: NDArray[Any],
        protect_np: NDArray[Any],
        material_np: NDArray[Any] | None,
        strata_number_fair: bool = False,
        seed: int = 1234567,
        title: str = '',
        ) -> tuple[NDArray[Any], NDArray[Any], str, dict]:
    """Use one of the different fairness adjustment methods."""
    # 1. Add a little noise, so that obs with same value appear to be ordered.
    #    Importantly, this ordering is random.
    xvars_noise_np, unique_vals = optp_fair_add.add_noise_discrete(
        xvars_np.copy()
        )
    # 2. Switch the values of the (semi-random) ranks from the conditional
    #    distribution to the (semi-random) ranks of the unconditional
    #    distribution
    xvars_fair_noise_np, cluster_fair_np, txt, fair_cfg = quantalisation(
        fair_cfg, gen_cfg, xvars_noise_np, protect_np, material_np, seed=seed,
        title=title, continuous=False, strata_number_fair=strata_number_fair,
        )
    # 3. Remove the noise so that the original scaling is retained (rounding).
    xvars_fair_np = optp_fair_add.remove_noise_discrete(
        xvars_fair_noise_np.copy(), unique_vals
        )

    return xvars_fair_np, cluster_fair_np, fair_cfg, txt


def quantalisation(fair_cfg: Any,
                   gen_cfg: Any,
                   xvars_np: NDArray[Any],
                   protect_np: NDArray[Any],
                   material_np: NDArray[Any] | None,
                   strata_number_fair: bool = False,
                   seed: int = 1246546,
                   title: str = '',
                   continuous: bool = True,
                   ) -> tuple[NDArray[Any], NDArray[Any], str,  dict]:
    """Adjust by quantalisation similar to Strack & Yang (2024)."""
    txt_report = ''
    fair_cfg = deepcopy(fair_cfg)
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
    if optp_fair_add.no_discretization(material_np, fair_cfg.material_max_groups
                                       ):
        fair_cfg.material_disc_method = 'NoDiscretization'
        txt_report += ('\nMaterial relevant features have no or only a few '
                       'values. Discretization is not used.'
                       )
    # Check if protected features should be treated as discrete
    if optp_fair_add.no_discretization(protect_np, fair_cfg.protected_max_groups
                                       ):
        fair_cfg.protected_disc_method = 'NoDiscretization'
        txt_report += ('\nProtected features have no or only a few values. '
                       'Discretization is not used.'
                       )

    # Discretize if needed, otherwise no change of data
    protect_np, material_np, txt_add = optp_fair_add.data_quantilized(
        fair_cfg, protect_np, material_np, seed)
    txt_report += txt_add
    # Get clusternumber for later use in Policy Trees
    if strata_number_fair:
        strata_fair_np, txt_add = cluster_number_within_cell_quantilization(
            protect_np, material_np
            )
        txt_report += txt_add
    else:
        strata_fair_np = None

    if ((fair_cfg.protected_disc_method in disc_methods)
        and ((material_np is None)
             or (fair_cfg.material_disc_method in disc_methods))):
        # No density estimation needed
        xvars_fair_np, txt_add = within_cell_quantilization(
            xvars_np, protect_np, material_np
            )
    else:
        xvars_fair_np, txt_add = kernel_quantilization(
            fair_cfg, xvars_np, protect_np, material_np)

    txt_report += txt_add
    if gen_cfg.with_output and title == '':
        mcf_ps.print_mcf(gen_cfg, txt_report, summary=True)

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
                data_cond_np = np.concatenate((protected_np, material_np),
                                              axis=1)
            quantiles_z, _ = calculate_quantiles_kde(xvars, data_cond_np,
                                                     xvars_all_grid)

        # Translate quantile to values of distribution conditional on
        # materially relevant variables only
        if fair_cfg.material_disc_method in disc_methods:
            if material_np is None:
                xvars_fair_np[:, idx_xvars] = values_from_quantiles(
                    xvars, quantiles_z)
            else:
                for val in vals_material:
                    mask = material_np.reshape(-1) == val
                    xvars_fair_np[mask, idx_xvars] = values_from_quantiles(
                        xvars[mask], quantiles_z[mask])
        else:
            _, xvars_fair_np[:, idx_xvars] = calculate_quantiles_kde(
                xvars, material_np, xvars_all_grid, quantile_data=quantiles_z)

    return xvars_fair_np, txt_report


def calculate_quantiles_kde(
        xvars: NDArray[Any],
        data_cond: NDArray[Any],
        xvars_grid: NDArray[Any],
        quantile_data: NDArray[Any] | None = None,
        ) -> tuple[NDArray[Any] | None, NDArray[Any] | None]:
    """Calculate the quantiles using Kernel density estimation."""
    quantile_values = quantile_data is not None
    if quantile_values:
        y_at_quantile = np.zeros_like(xvars)
    else:
        quantile_at_y = np.zeros_like(xvars)

    num_points = len(xvars_grid)

    # Fit KDE for the joint density p(x, y)
    if data_cond.shape[1] == 1:
        data_cond = data_cond.reshape(-1, 1)
    if data_cond is None:
        joint_data = xvars.reshape(-1, 1)
    else:
        joint_data = np.concatenate((data_cond, xvars.reshape(-1, 1)), axis=1)
    kde_joint = KernelDensity(kernel='epanechnikov', bandwidth='silverman'
                              ).fit(joint_data)
    kde_marg = KernelDensity(kernel='epanechnikov', bandwidth='silverman'
                             ).fit(data_cond)

    kde_marg_data = np.exp(kde_marg.score_samples(data_cond))  # All datapoints

    for idx, xvars_i in enumerate(xvars):
        xy_grid = np.hstack([np.tile(data_cond[idx, :], (num_points, 1)),
                             xvars_grid])
        joint_density = np.exp(kde_joint.score_samples(xy_grid))
        cond_density_idx = joint_density / kde_marg_data[idx]
        cond_density_idx /= np.sum(cond_density_idx)
        # Calculate the cumulative distribution function (CDF)
        cdf_idx = np.cumsum(cond_density_idx)
        # Normalise it (again, as a safeguard, should be unnecessary)
        cdf_idx /= cdf_idx[-1]

        if quantile_values:
            # Function to interpolate the quantile function (inverse CDF)
            quantile_func = interp1d(
                cdf_idx, xvars_grid.flatten(), bounds_error=False,
                fill_value=(xvars_grid[0, 0], xvars_grid[-1, 0]))
            y_at_quantile[idx] = quantile_func(quantile_data[idx])
            quantile_at_y = None
        else:
            cdf_func = interp1d(xvars_grid.flatten(), cdf_idx,
                                bounds_error=False, fill_value=(0, 1))
            quantile_at_y[idx] = cdf_func(xvars_i)
            y_at_quantile = None

    return quantile_at_y, y_at_quantile


def get_grid(data: np.array, no_eval_point: int | float) -> NDArray[Any]:
    """Get evaluation grid for densities."""
    grid = np.linspace(data.min(), data.max(), no_eval_point).reshape(-1, 1)

    return grid


def cluster_number_within_cell_quantilization(
        protected_np: NDArray[Any],
        material_np: NDArray[Any] | None,
        ) -> tuple[NDArray[Any], str]:
    """Do within cell quantilization for discrete univariate features."""
    if material_np is None:
        material_np = np.ones((protected_np.shape[0], 1))
    material_values = np.unique(material_np, return_counts=False)
    cluster_fair_np = np.zeros(len(protected_np))
    cluster_no = 0
    for mat_val in material_values:
        indices_mat = np.where(material_np == mat_val)[0]
        if indices_mat.size == 0:  # Empty cluster
            continue

        prot_mat = protected_np[indices_mat].reshape(-1)
        vals_prot_mat = np.unique(prot_mat, return_counts=False)
        if len(vals_prot_mat) == 1:
            cluster_no += 1
            continue
        for val in vals_prot_mat:
            mask = prot_mat == val
            index_mask = indices_mat[mask]
            cluster_fair_np[index_mask] = cluster_no
            cluster_no += 1

    txt_report = ('\nNumber of clusters (protected x materially relevant): '
                  f'{cluster_no}'
                  )

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
    cluster_no = 0
    for mat_val in material_values:
        indices_mat = np.where(material_np == mat_val)[0]
        if indices_mat.size == 0:
            continue

        prot_mat = protected_np[indices_mat].reshape(-1)
        vals_prot_mat = np.unique(prot_mat, return_counts=False)
        if len(vals_prot_mat) == 1:
            cluster_no += 1
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
            xvars_fair_np[indices_mat, idx_xvars] = values_from_quantiles(
                xvars_mat, quantiles)

    return xvars_fair_np, txt_report


def calculate_quantiles(data: NDArray[Any]) -> NDArray[Any]:
    """Calculate quantiles for each value in the dataset."""
    data_sort = np.sort(data)  # Sort the data
    rank = np.empty_like(data)
    for idx, value in enumerate(data):
        rank[idx] = np.searchsorted(data_sort, value, side='right')  # Find rank

    return rank / len(data)


def values_from_quantiles(data: NDArray[Any],
                          quantiles: NDArray[Any]
                          ) -> NDArray[Any]:
    """Get the values from the quantiles."""
    d_sorted = np.sort(data)  # Sort the empirical distribution
    obs = len(d_sorted)       # Number of data points
    indices = np.int64(np.round(quantiles * (obs - 1)))  # Compute the indices

    return d_sorted[indices]


def residualisation(fair_cfg: Any,
                    gen_cfg: Any,
                    xvars_np: NDArray[Any],
                    xvars_name: list | tuple,
                    protect_np: NDArray[Any],
                    material_np: NDArray[Any] | None,
                    seed: int = 1246546,
                    title: str = '',
                    ) -> tuple[NDArray[Any], str]:
    """Adjust by residualisation."""
    # Info and tuning parameters
    obs, no_of_xvars = xvars_np.shape
    boot, cross_validation_k, txt_report = 5, 1000, ''

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
                print('\n' + title + f'Currently adjusting {mean_var} of '
                      f'{xvars_name[idx]}')

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
            (estimator, params, best_label, _, transform_x, txt_mse
             ) = mcf_gf.best_regression(
                x_cond_np, y_np.ravel(),
                estimator=fair_cfg.regression_method,
                boot=boot,
                seed=seed + 12435,
                max_workers=gen_cfg.mp_parallel,
                cross_validation_k=cross_validation_k,
                absolute_values_pred=mean_var == 'variance')

            if with_material_x:
                (estimator_m, params_m, best_label_m, _, transform_x_m,
                 txt_mse_m) = mcf_gf.best_regression(
                    material_np,  y_np.ravel(),
                    estimator=fair_cfg.regression_method,
                    boot=boot,
                    seed=seed + 12435,
                    max_workers=gen_cfg.mp_parallel,
                    cross_validation_k=cross_validation_k,
                    absolute_values_pred=mean_var == 'variance')

            if gen_cfg.with_output:
                text = ('\n' + title + f'Adjustment for {mean_var} of '
                        f'{xvars_name[idx]}:')
                txt_mse = text + txt_mse
                if with_material_x:
                    txt_mse += '\n' + title + 'Short regression:' + txt_mse_m
                mcf_ps.print_mcf(gen_cfg, txt_mse, summary=False)
                if mean_var == 'mean':
                    txt_report += '\n'
                txt_report += text + ' by ' + best_label
                if with_material_x:
                    txt_report += ('\n' + title + 'Short regression adjusted '
                                   + 'by ' + best_label_m)

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
                    _, x_train, x_pred = mcf_gf.scale(x_cond_np[index_train],
                                                      x_cond_np[index_pred])
                else:
                    x_train = x_cond_np[index_train]
                    x_pred = x_cond_np[index_pred]
                if with_material_x:
                    if transform_x_m:
                        _, x_train_m, x_pred_m = mcf_gf.scale(
                            material_np[index_train], material_np[index_pred])
                    else:
                        x_train_m = material_np[index_train]
                        x_pred_m = material_np[index_pred]

                y_train = y_np[index_train].ravel()
                y_obj = mcf_gf.regress_instance(estimator, params)
                if y_obj is None:
                    if mean_var == 'mean':
                        y_mean_cond_x_np[index_pred, idx] = np.average(y_train)
                    else:
                        y_var_cond_x_np[index_pred, idx] = (
                            np.average(y_train) - y_mean_x_2[index_pred])
                else:
                    y_obj.fit(x_train, y_train)
                    if mean_var == 'mean':
                        y_mean_cond_x_np[index_pred, idx] = y_obj.predict(x_pred
                                                                          )
                    else:
                        y_var_cond_x_np[index_pred, idx] = (
                            y_obj.predict(x_pred) - y_mean_x_2[index_pred])

                if with_material_x:
                    y_obj_m = mcf_gf.regress_instance(estimator_m, params_m)
                    if y_obj_m is None:
                        if mean_var == 'mean':
                            y_mean_cond_mat_np[index_pred, idx] = np.average(
                                y_train)
                        else:
                            y_var_cond_mat_np[index_pred, idx] = (
                                np.average(y_train) - y_mean_mat_2[index_pred])
                    else:
                        y_obj_m.fit(x_train_m, y_train)
                        if mean_var == 'mean':
                            y_mean_cond_mat_np[index_pred, idx] = (
                                y_obj_m.predict(x_pred_m))
                        else:
                            y_var_cond_mat_np[index_pred, idx] = (
                                y_obj_m.predict(x_pred_m)
                                - y_mean_mat_2[index_pred])

    residuum_np = xvars_np - y_mean_cond_x_np

    # Adjust variance as well
    if fair_cfg.adj_type == 'MeanVar':

        # Conditional standard deviation (must be non-zero)
        bound_var = 1e-6
        y_std_cond_x_np = optp_fair_add.var_to_std(y_var_cond_x_np, bound_var)
        if with_material_x:
            y_std_cond_mat_np = optp_fair_add.var_to_std(y_var_cond_mat_np,
                                                         bound_var)

        # Avoid too extreme values when scaling
        bound = 0.05
        y_std_cond_x_np = optp_fair_add.bound_std(y_std_cond_x_np, bound,
                                                  no_of_xvars)

        # Remove predictability due to heteroscedasticity
        residuum_np /= y_std_cond_x_np

        # Rescale to retain about the variability of the original variabless
        std_resid = np.std(residuum_np, axis=0).reshape(-1, 1).T
        residuum_np /= std_resid
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


def test_for_consistency(fair_cfg: Any,
                         gen_cfg: Any,
                         var_cfg: Any,
                         data_fair_df: pd.DataFrame,
                         score_fair_name: list | tuple,
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
    score_fair_np = data_fair_df[score_fair_name].to_numpy()
    scores_np, protect_np, material_np = vars_to_numpy(
        data_fair_df, var_cfg.polscore_name, var_cfg.protected_name,
        var_cfg.material_name
        )
    # Find valid combinations of scores that can be tested
    (test_scores_np, test_fair_scores_np, test_scores_name, _
     ) = optp_fair_add.score_combinations(score_fair_np, score_fair_name,
                                          scores_np, score_name)

    # Fairness adjust the score differences
    test_scores_adj_np, _, _, _ = fair_continuous_xvars_fct(
        fair_cfg, gen_cfg, test_scores_np, test_scores_name, protect_np,
        material_np, seed=seed, title=title, strata_number_fair=False,
        )
    test_scores_adj_name = [name + '_adj' for name in test_scores_name]

    test_diff_np = (np.mean(np.abs(test_fair_scores_np - test_scores_adj_np),
                            axis=0)
                    / np.std(test_scores_np, axis=0)
                    )
    same_sign_np = np.mean(
        np.sign(test_fair_scores_np) == np.sign(test_scores_adj_np), axis=0)

    for idx, name in enumerate(test_scores_adj_name):
        correlation = np.corrcoef(test_fair_scores_np[:, idx],
                                  test_scores_adj_np[:, idx])[0, 1]
        test_dic[name] = [test_diff_np[idx], same_sign_np[idx], correlation]

    txt = ''
    if gen_cfg.with_output:
        txt1 = ('\nTest for consistency of different fairness normalisations:'
                '\n    - Compare difference of adjusted scores to '
                'adjusted difference of scores'
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
            '\nCorrelation: Share of scores with same sign. '
            'Ideal value is 100%.'
            )
        mcf_ps.print_mcf(gen_cfg, '\n' + '-' * 100 + txt + txt1, summary=True)
        txt += '\n' + txt1

    return test_dic, txt


def update_names_for_solve_in_self(var_cfg: Any,
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


def fair_adjust_data_for_pred(fair_cfg: Any,
                              gen_cfg: Any,
                              var_cfg: Any,
                              data_df: pd.DataFrame,
                              fair_adjust_decision_vars: bool,
                              ) -> tuple[pd.DataFrame, str]:
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

    fair_adjustvar = (fair_cfg.adjust_target in ('xvariables',
                                                 'scores_xvariables',))

    fair_decision_var_na = any(elem not in data_df.columns
                               for elem in [*var_cfg.x_ord_name,
                                            *var_cfg.x_unord_name]
                               )
    if (var_cfg.protected_unord_name
        and any(elem not in data_df.columns
                for elem in var_cfg.protected_name)):
        dummies_df = pd.get_dummies(data_df[var_cfg.protected_unord_name],
                                    columns=var_cfg.protected_unord_name,
                                    dtype=int
                                    )
        data_df = pd.concat((data_df, dummies_df), axis=1)

    # if (var_cfg.material_name
    #     and any(elem not in data_df.columns
    #             for elem in var_cfg.material_unord_name)):
    if (var_cfg.material_unord_name
        and any(elem not in data_df.columns
                for elem in var_cfg.material_name)):
        dummies_df = pd.get_dummies(data_df[var_cfg.material_unord_name],
                                    columns=var_cfg.material_unord_name,
                                    dtype=int
                                    )
        data_df = pd.concat((data_df, dummies_df), axis=1)

    if fair_adjustvar and fair_decision_var_na:
        # If adjusted variables are not available,they must be created and
        # added to the data
        # Add explanations to report file
        txt += (
            '\n\nFairness adjusted variables were not available in the '
            'data provided to compute allocations (prediction data). '
            'Therefore, they are computed with the prediction data.'
            )
        fair_adjust_decision_vars = True

    if fair_adjust_decision_vars:
        data_train_df = fair_cfg.data_train_for_pred_df
        add_train_data = len(data_train_df) > len(data_df) * 2
        if add_train_data:
            txt += (
                '\nData used for training is more than twice as large as '
                'data to be used for prediction. Therefore, for computing '
                'variable fairness adjustments, training data will be '
                'added to the prediction data.'
                )
            data_temp_df = pd.concat(
                (data_df[data_train_df.columns], data_train_df,), axis=0
                ).reset_index(drop=True)
        else:
            data_temp_df = data_df

        var_cfg_temp = deepcopy(var_cfg)
        var_cfg_temp.x_ord_name = fair_cfg.org_name_dict['x_ord_name']
        var_cfg_temp.x_unord_name = fair_cfg.org_name_dict['x_unord_name']

        data_temp_df, _, _, _, fair_cfg, _, = adjust_decision_variables(
            deepcopy(fair_cfg), deepcopy(gen_cfg), var_cfg_temp,
            data_temp_df.copy(), training=False, seed=1234567,
            )
        if add_train_data:
            data_pred_df = data_temp_df.iloc[:len(data_df), :]
            data_pred_rel_df = data_pred_df[
                [*var_cfg.x_ord_name, *var_cfg.x_unord_name]
                ]
            data_df = pd.concat((data_df, data_pred_rel_df,), axis=1
                                ).reset_index(drop=True)
        else:
            data_df = data_temp_df

    return data_df, fair_cfg, txt
