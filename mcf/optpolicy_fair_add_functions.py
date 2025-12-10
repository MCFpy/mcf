"""Created on Wed May  1 16:35:19 2024.

Functions for correcting scores w.r.t. protected variables.

@author: MLechner

# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.cluster import KMeans

from mcf import mcf_print_stats_functions as mcf_ps
from mcf.mcf_general import remove_duplicates, unique_list
from mcf.mcf_estimation_generic_functions import cramers_v


def cont_or_discrete(data_df: pd.DataFrame,
                     x_ord_name: list | tuple,
                     x_unord_name: list | tuple,
                     cont_min_values: int
                     ) -> tuple[list, list, tuple]:
    """Find continuous variable and start dictionary with fairness info."""
    info_dict = {}
    x_cont_names, x_discr_names = [], []

    for var in x_ord_name:
        info_var_dic = {}
        unique_count = data_df[var].nunique()
        if unique_count < cont_min_values:
            info_var_dic['type'] = 'discr'
            info_var_dic['fair_name'] = var + '_fair'
            x_discr_names.append(var)
        else:
            info_var_dic['type'] = 'cont'
            info_var_dic['fair_name'] = var + '_fair'
            x_cont_names.append(var)
        info_dict[var] = info_var_dic.copy()

    for var in x_unord_name:
        info_var_dic = {}
        info_var_dic['type'] = 'discr'
        x_discr_names.append(var)
        info_dict[var] = info_var_dic.copy()

    return x_cont_names, x_discr_names, info_dict


def score_combinations(scores_fair_np: NDArray[Any],
                       scores_fair_name: list | tuple,
                       scores_np: NDArray[Any],
                       scores_name: list | tuple
                       ) -> tuple[NDArray[Any], NDArray[Any], list, list]:
    """Find and create combinations of scores that can be tested."""
    fs_np, fs_name, s_np, s_name = check_if_variation(
        scores_fair_np, scores_fair_name, scores_np, scores_name)

    obs, no_scores = s_np.shape
    no_of_new_scores = int(no_scores * (no_scores - 1) / 2)
    test_np = np.zeros((obs, no_of_new_scores))
    test_fair_np = np.zeros_like(test_np)
    test_name, test_fair_name = [], []
    index = 0
    for idx in range(0, no_scores):
        for jdx in range(idx+1, no_scores):
            test_np[:, index] = s_np[:, jdx] - s_np[:, idx]
            test_fair_np[:, index] = fs_np[:, jdx] - fs_np[:, idx]
            test_name.append(s_name[jdx] + '_m_' + s_name[idx])
            test_fair_name.append(fs_name[jdx] + '_m_' + fs_name[idx])
            index += 1

    return test_np, test_fair_np, test_name, test_fair_name


def check_if_variation(fair_scores_np: NDArray[Any],
                       fairscores_name: list | tuple,
                       scores_np: NDArray[Any],
                       scores_name: list | tuple,
                       ) -> tuple[NDArray[Any], list, NDArray[Any], list]:
    """Delete scores without variation."""
    const = 1e-8
    score_const = (np.std(fair_scores_np, axis=0)
                   + np.std(scores_np, axis=0)) > const

    positions_0 = np.where(score_const)[0]
    if not score_const.all():
        fs_np = fair_scores_np[:, positions_0].copy()
        s_np = scores_np[:, positions_0].copy()
        fs_name = [fairscores_name[idx] for idx in positions_0]
        s_name = [scores_name[idx] for idx in positions_0]
    else:
        fs_np, fs_name = fair_scores_np.copy(), fairscores_name[:]
        s_np, s_name = scores_np.copy(), scores_name[:]

    return fs_np, fs_name, s_np, s_name


def change_variable_names_fair(var_cfg: Any,
                               fairscore_name: list | tuple,
                               ) -> dict:
    """Change variable names to account for fair scores."""
    var_cfg = deepcopy(var_cfg)
    if var_cfg.polscore_desc_name is not None:
        var_cfg.polscore_desc_name.extend(var_cfg.polscore_name.copy())
    else:
        var_cfg.polscore_desc_name = var_cfg.polscore_name.copy()
    # Remove duplicates from list without changing order
    var_cfg.polscore_desc_name = remove_duplicates(var_cfg.polscore_desc_name)

    if var_cfg.vi_x_name is not None and var_cfg.protected_ord_name is not None:
        var_cfg.vi_x_name.extend(var_cfg.protected_ord_name)
    elif var_cfg.vi_x_name is None and var_cfg.protected_ord_name is not None:
        var_cfg.vi_x_name = var_cfg.protected_ord_name

    if (var_cfg.vi_to_dummy_name is not None
            and var_cfg.protected_unord_name is not None):
        var_cfg.vi_to_dummy_name.extend(var_cfg.protected_unord_name)
    elif (var_cfg.vi_to_dummy_name is None
          and var_cfg.protected_ord_name is not None):
        var_cfg.vi_to_dummy_name = var_cfg.protected_unord_name

    if var_cfg.vi_x_name is not None:
        var_cfg.vi_x_name = unique_list(var_cfg.vi_x_name)
    if var_cfg.vi_to_dummy_name is not None:
        var_cfg.vi_to_dummy_name = unique_list(var_cfg.vi_to_dummy_name)

    var_cfg.polscore_name = fairscore_name

    return var_cfg


def fair_stats(gen_cfg: Any,
               var_cfg: Any,
               data_df: pd.DataFrame,
               var_name: list | tuple,
               var_fair_name: list | tuple,
               var_continuous: bool = True,
               cont_min_values: int | float = 20,
               ) -> str:
    """Compute descriptive statistics of fairness adjustments."""
    prot_names = var_cfg.protected_name
    if var_continuous:
        measure = 'Correlation'
        var_type = 'continuous'
    else:
        measure = 'Cramers V'
        var_type = 'discrete'

    txt = ('\n\nDescriptive statistics of adjusted and unadjusted '
           f'{var_type} variables'
           )
    all_variables_all = [*var_name, *var_fair_name]
    seen = set()
    all_variables = [x for x in all_variables_all
                     if not (x in seen or seen.add(x))]
    stats = np.round(data_df[all_variables].describe().T, 3)

    data_df = data_df.loc[:, ~data_df.columns.duplicated()]
    if var_continuous:
        corr = np.round(data_df[all_variables].corr() * 100, 1)
    else:
        corr_np = np.zeros((len(all_variables), len(all_variables)))
        corr = pd.DataFrame(corr_np, index=all_variables, columns=all_variables)
        for var1 in all_variables:
            for var2 in all_variables:
                corr.loc[var1, var2] = np.round(
                    cramers_v(data_df[var1], data_df[var2]) * 100,
                    1)

    corr = corr.dropna(how='all', axis=0)
    corr = corr.dropna(how='all', axis=1)
    txt += '\n' + str(stats) + '\n\n'
    txt += f'{measure} of adjusted and unadjusted variables in %'
    txt += '\n' + str(corr)
    correlations_np = np.zeros((len(all_variables), len(prot_names)))
    correlations_df = pd.DataFrame(correlations_np,
                                   index=all_variables, columns=prot_names)
    # Determine which protected variables are continuous (correlation or CramV)
    unique_count = data_df[prot_names].nunique(axis=0)
    prot_continuous = [count >= cont_min_values for count in unique_count]
    labels = ['    Corr' if (c or var_continuous) else '   CramV'
              for c in prot_continuous]
    longest_var = max(len(v) for v in all_variables)
    label = ' ' * longest_var + ' '.join(labels)
    for var in all_variables:
        for prot_idx, prot in enumerate(prot_names):

            if var_continuous or prot_continuous[prot_idx]:
                corr = data_df[[var, prot]].corr().iloc[1, 0]
            else:
                corr = cramers_v(data_df[var], data_df[prot])
            correlations_df.loc[var, prot] = round(corr * 100, 1)

    correlations_df = correlations_df.dropna(how='all', axis=0)
    correlations_df = correlations_df.dropna(how='all', axis=1)
    txt += ('\n' * 2
            + 'Bivariate dependence of (modified) decision and protected '
              'variables in %'
            + '\n' + label + '\n' + str(correlations_df)
            )
    mcf_ps.print_mcf(gen_cfg, txt, summary=True)

    return txt


def var_to_std(var: NDArray[Any], constant: NDArray[Any]) -> NDArray[Any]:
    """Change variance to standard deviation."""
    var_neu = var.copy()
    mask_var = var < constant
    if mask_var.any():
        var_neu = np.where(mask_var, constant, var)

    return np.sqrt(var_neu)


def bound_std(std: NDArray[Any],
              bound: float,
              no_of_scores: int
              ) -> NDArray[Any]:
    """Lower bound for standard deviation to avoid explosion of values."""
    std_neu = std.copy()
    for idx in range(no_of_scores):
        lower = bound * np.mean(std[:, idx])
        mask_var = std[:, idx] < lower
        if mask_var.any():
            std_neu[:, idx] = np.where(mask_var, lower, std[:, idx])
    return std_neu


def data_quantilized(fair_cfg: Any,
                     protected_in_np: NDArray[Any],
                     material_in_np: NDArray[Any],
                     seed: int = 12345
                     ) -> tuple[NDArray[Any], NDArray[Any], str]:
    """Prepare the data (discretize) for fairness quantilization."""
    txt = ''
    # Materially relevant variables if they exist, and if relevant
    if material_in_np is not None:
        mat_max_groups = fair_cfg.material_max_groups
        method = getattr(fair_cfg, 'material_disc_method')

        match method:
            case 'EqualCell':
                material_np = discretize_equalcell(material_in_np,
                                                   mat_max_groups
                                                   )
            case 'Kmeans':
                material_np = discretize_kmeans(material_in_np,
                                                mat_max_groups, seed=seed
                                                )
            case 'NoDiscretization':
                material_np = recode_no_discretization(material_in_np)
            case _:
                raise ValueError(f'Unknown discretization method: {method!r}')

        if fair_cfg.material_disc_method != 'NoDiscretization':
            txt += ('\nMaterially relevant features discretized: '
                    f'{len(np.unique(material_np))} cells.'
                    )
    else:
        material_np = None

    method = getattr(fair_cfg, 'protected_disc_method')
    k = fair_cfg.protected_max_groups
    match method:
        case 'EqualCell':
            protected_np = discretize_equalcell(protected_in_np, k)
        case 'Kmeans':
            protected_np = discretize_kmeans(protected_in_np, k,
                                             seed=seed + 124335
                                             )
        case 'NoDiscretization':
            protected_np = recode_no_discretization(protected_in_np)
        case _:
            raise ValueError(f'Unknown discretization method: {method!r}')

    if fair_cfg.protected_disc_method != 'NoDiscretization':
        txt += ('\nProtected features discretized: '
                f'{len(np.unique(protected_np))} cells.'
                )
    return protected_np, material_np, txt


def discretize_kmeans(data_in_np: NDArray[Any],
                      max_groups: int | float,
                      seed: int = 12345
                      ) -> NDArray[Any]:
    """Discretize using cells of similar size (for cont. features)."""
    data_np = KMeans(
        n_clusters=max_groups,
        init='k-means++',
        n_init='auto',
        max_iter=1000,
        algorithm='lloyd',
        random_state=seed,
        tol=1e-5,
        verbose=0,
        copy_x=True
        ).fit_predict(data_in_np.copy())

    return data_np.reshape(-1, 1)


def discretize_equalcell(data_in_np: NDArray[Any],
                         max_groups: int
                         ) -> NDArray[Any]:
    """Discretize using cells of similar size (for cont. features)."""
    data_np = data_in_np.copy()

    cols = data_in_np.shape[1]
    for col_idx in range(cols):
        unique_val = np.unique(data_in_np[:, col_idx])
        if len(unique_val) > max_groups:
            data_np[:, col_idx] = pd.qcut(data_in_np[:, col_idx],
                                          q=max_groups,
                                          labels=False)
    if cols > 1:
        _, data_np = np.unique(data_np, axis=0, return_inverse=True)

    return data_np.reshape(-1, 1)


def no_discretization(data_np: NDArray[Any] | None, threshold: int) -> bool:
    """Check if unique values summed up over columns is > than threshold."""
    if data_np is None:
        return True

    total = 0
    for col in range(data_np.shape[1]):
        n_unique = len(np.unique(data_np[:, col]))
        total += n_unique
        if total > threshold:
            return False

    return True


def recode_no_discretization(data_in_np: NDArray[Any]) -> NDArray[Any]:
    """Recode unique values starting with 0."""
    _, ids = np.unique(data_in_np, axis=0, return_inverse=True)

    return ids.reshape(-1, 1)


def reshuffle_share_rows(data_org_np: NDArray[Any],
                         share: float,
                         seed: int = None,
                         ) -> NDArray[Any]:
    """Reshuffle a share of rows in a numpy array."""
    data_np = data_org_np.copy()
    if share == 0:
        return data_np

    rng = np.random.default_rng(seed=seed)
    n_rows = data_np.shape[0]
    n_sample = int(np.floor(share * n_rows))

    # Select random row indices to reshuffle
    selected_indices = rng.choice(n_rows, size=n_sample, replace=False)

    # Extract and reshuffle the selected rows
    selected_rows = data_np[selected_indices]
    shuffled_rows = selected_rows.copy()
    rng.shuffle(shuffled_rows)

    # Create a copy of the original array and replace with shuffled rows
    result = data_np.copy()
    result[selected_indices] = shuffled_rows

    return result


def add_noise_discrete(xvars_in_np: NDArray[Any],
                       seed: int = 1234567,
                       ) -> tuple[NDArray[Any], list[float], list[list[float]]]:
    """Add a little noise to discrete variables."""
    rng = np.random.default_rng(seed=seed)
    noise = rng.uniform(-0.05, 0.05, size=(xvars_in_np.shape[0],))
    xvars_noise_np = xvars_in_np.copy()
    unique_values_all = []
    for col in range(xvars_in_np.shape[1]):
        unique_col = np.unique(xvars_in_np[:, col])  # values are sorted
        unique_values_all.append(unique_col)
        min_dist = np.abs(np.min(np.diff(unique_col)))
        if min_dist < 1:
            xvars_noise_np[:, col] = (xvars_in_np[:, col]
                                      + min_dist * noise)
        else:
            xvars_noise_np[:, col] = xvars_in_np[:, col] + noise

    return xvars_noise_np, unique_values_all


def remove_noise_discrete(xvars_noise_np: NDArray[Any],
                          unique_values_all: list[list[float]],
                          ) -> NDArray[Any]:
    """Remove the noise from discrete variables."""
    xvars_np = np.zeros_like(xvars_noise_np)

    for k in range(xvars_noise_np.shape[1]):
        valid_values = np.array(unique_values_all[k])
        diffs = np.abs(xvars_noise_np[:, k, None] - valid_values[None, :])
        # shape: (N, len(valid_values))
        closest_indices = np.argmin(diffs, axis=1)
        xvars_np[:, k] = valid_values[closest_indices]

    return xvars_np
