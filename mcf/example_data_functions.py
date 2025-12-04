"""
Created on Wed Dec 21 15:37:16 2022.

# -*- coding: utf-8 -*-
@author: MLechner
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz
from scipy.stats import logistic, norm


def example_data(obs_y_d_x_iate: int = 1000,
                 obs_x_iate: int = 1000,
                 no_features: int = 20,
                 no_treatments: int = 3,
                 type_of_heterogeneity: str = 'WagerAthey',
                 seed: int = 12345,
                 descr_stats: bool = True,
                 strength_iv: int = 1,
                 correlation_x: str = 'middle',
                 no_effect=False,
                 ):
    """
    Create example data to be used with mcf estimation and optimal policy.

    Parameters
    ----------
    obs_y_d_x_iate : Integer, optional
        Number of observations for training data. The default is 1000.
    obs_x_iate : Integer, optional
        Number of observations for prediction data. The default is 1000.
    no_features : Integer, optional
        Number of features of different type. The default is 20.
    no_treatments : Integer, optional
        Number of treatments (all non-zero treatments have same IATEs).
        The default is 3.
    type_of_heterogeneity : String, optional
        Different types of heterogeneity broadly (but not exactly) following
        the specifications used in the simulations of Lechner and Mareckova
        (Comprehensive Causal Machine Learning, arXiv, 2024). Possible
        types are 'linear', 'nonlinear', 'quadratic', 'WagerAthey'.
    seed : Integer, optional
        Seed of numpy random number generator object. The default is 12345.
    descr_stats :  Boolean, optional
        Show descriptive statistics. The default is True.
    strength_iv : Integer or Float, optional.
        The larger this number is, the stronger the instrument will be.
        Default is 1.
    correlation_x : str, optinal
        Allows three different levels of dependence between features ('low',
        'middle', 'high'). Default is 'middle'.
    no_effect : Boolean, optional
        All IATEs are set to 0 if True.

    Returns
    -------
    train_df : DataFrame
        Contains outcome, treatment, features, potential outcomes, IATEs, ITEs,
        and zero column (for convenience to be used with OptimalPolicy).
    pred_df : DataFrame
        Contains features, potential outcomes, IATEs, ITEs.
    name_dict : Dictionary
        Contains the names of the variable groups.

    """
    if obs_y_d_x_iate is None:
        obs_y_d_x_iate = 1000

    if obs_x_iate is None:
        obs_x_iate = 1000

    if correlation_x not in ('low', 'middle', 'high'):
        raise ValueError('Illegal correlation level of features '
                         f'{correlation_x} specified. Allowed are only "low", '
                         '"middle", and "high".')
    if type_of_heterogeneity not in ('linear', 'nonlinear', 'quadratic',
                                     'WagerAthey'):
        raise ValueError(f'Illegal heterogeneity: {type_of_heterogeneity} '
                         'specified. Allowed are only "linear", "nonlinear", '
                         '"quadratic", and "WagerAthey"')
    k_cont = round(no_features/3)
    k_ord_cat = round(no_features/3)
    k_unord = no_features - k_cont - k_ord_cat
    k_all = (k_cont, k_ord_cat, k_unord)
    name_dict = {}
    name_dict['d_name'] = 'treat'
    name_dict['y_name'] = 'outcome'
    name_dict['y_pot_name'] = ['y_pot' + str(i) for i in range(no_treatments)]
    name_dict['y_pot_se_name'] = ['y_pot' + str(i) + '_se'
                                  for i in range(no_treatments)]
    name_dict['iate_name'] = ['iate' + str(i+1) + 'vs0'
                              for i in range(no_treatments-1)]
    name_dict['ite_name'] = ['ite' + str(i+1) + 'vs0'
                             for i in range(no_treatments-1)]
    name_dict['zero_name'] = 'zero'
    name_dict['ite0_name'] = 'ite0vs0'
    name_dict['iate0_name'] = 'iate0vs0'
    name_dict['id_name'] = 'id'
    name_dict['cluster_name'] = 'cluster'
    name_dict['weight_name'] = 'weight'
    name_dict['x_name_ord'] = ['x_cont' + str(i) for i in range(k_cont)]
    name_dict['x_name_ord'].extend(['x_ord' + str(i) for i in range(k_ord_cat)])
    name_dict['x_name_unord'] = ['x_unord' + str(i) for i in range(k_unord)]
    name_dict['inst_bin_name'] = ['binary_instrument',]
    x_name = [*name_dict['x_name_ord'], *name_dict['x_name_unord']]
    rng = np.random.default_rng(seed=seed)

    # Features
    x_train = covariates_x(rng, k_all, obs_y_d_x_iate, correlation=correlation_x
                           )
    x_pred = covariates_x(rng, k_all, obs_x_iate, correlation=correlation_x)

    # Instrument
    inst_train = instrument(rng, obs_y_d_x_iate)
    inst_pred = instrument(rng, obs_x_iate)
    # Treatment
    d_train = treatment_d(rng, no_treatments, k_all, x_train, inst_train,
                          strength_iv=strength_iv, correlation_x=correlation_x)
    d_pred = treatment_d(rng, no_treatments, k_all, x_pred, inst_pred,
                         strength_iv=strength_iv, correlation_x=correlation_x)

    # Potential and observed outcomes
    y_pot_train, y_pot_se_train, iate_train, ite_train, y_train = get_outcomes(
        rng, x_train, d_train, k_all, type_of_heterogeneity,
        plot_iate=descr_stats, no_effect=no_effect)
    y_pot_pred, y_pot_se_pred, iate_pred, ite_pred, _ = get_outcomes(
        rng, x_pred, d_pred, k_all, type_of_heterogeneity, plot_iate=False,
        no_effect=no_effect)

    id_train = np.arange(obs_y_d_x_iate).reshape(-1, 1)
    id_pred = np.arange(obs_x_iate).reshape(-1, 1)

    cluster_train = rng.integers(0, 100, size=obs_y_d_x_iate).reshape(-1, 1)
    cluster_pred = rng.integers(0, 100, size=obs_x_iate).reshape(-1, 1)
    weight_train = rng.uniform(0.5, 1.5, size=obs_y_d_x_iate).reshape(-1, 1)
    weight_pred = rng.uniform(0.5, 1.5, size=obs_x_iate).reshape(-1, 1)

    train_np = np.concatenate(
        (y_train, d_train, x_train, inst_train, y_pot_train, y_pot_se_train,
         iate_train, ite_train, id_train, cluster_train, weight_train,
         np.zeros((len(ite_train), 3))),
        axis=1)
    train_df = pd.DataFrame(data=train_np,
                            columns=(name_dict['y_name'],
                                     name_dict['d_name'],
                                     *x_name,
                                     *name_dict['inst_bin_name'],
                                     *name_dict['y_pot_name'],
                                     *name_dict['y_pot_se_name'],
                                     *name_dict['iate_name'],
                                     *name_dict['ite_name'],
                                     name_dict['id_name'],
                                     name_dict['cluster_name'],
                                     name_dict['weight_name'],
                                     name_dict['zero_name'],
                                     name_dict['ite0_name'],
                                     name_dict['iate0_name'],
                                     )
                            )

    pred = np.concatenate((d_pred, x_pred, inst_pred, y_pot_pred, y_pot_se_pred,
                           iate_pred, ite_pred, id_pred, cluster_pred,
                           weight_pred, np.zeros((len(ite_pred), 3))),
                          axis=1)
    pred_df = pd.DataFrame(data=pred, columns=(name_dict['d_name'],
                                               *x_name,
                                               *name_dict['inst_bin_name'],
                                               *name_dict['y_pot_name'],
                                               *name_dict['y_pot_se_name'],
                                               *name_dict['iate_name'],
                                               *name_dict['ite_name'],
                                               name_dict['id_name'],
                                               name_dict['cluster_name'],
                                               name_dict['weight_name'],
                                               name_dict['zero_name'],
                                               name_dict['ite0_name'],
                                               name_dict['iate0_name'],
                                               )
                           )
    if descr_stats:
        descriptive_stats(train_df, pred_df, name_dict, x_name=x_name)
    return train_df, pred_df, name_dict


def get_observed_outcome(y_pot, d_np):
    """Get the observed values from the potentials."""
    y_np = y_pot[np.arange(len(y_pot)), d_np[:, 0]].reshape(-1, 1)
    return y_np


def instrument(rng, obs):
    """Create instrument."""
    instr_train = np.int8(rng.uniform(low=0, high=1, size=(obs, 1)) > 0.5)
    return instr_train


def get_outcomes(rng, x_np, d_np, k_all, iate_type='WagerAthey',
                 plot_iate=True, no_effect=False):
    """Simulate the outcome data."""
    k_cont, k_ord_cat, k_unord = k_all
    obs, treat = len(x_np), len(np.unique(d_np))

    noise_y0 = rng.normal(loc=0, scale=1, size=(obs, 1))
    noise_y1 = rng.normal(loc=0, scale=1, size=(obs, treat-1))

    # Need type specific coefficeint
    x_cont = x_np[:, :k_cont]
    x_ord = x_np[:, k_cont:k_ord_cat+k_cont]
    x_unord = x_np[:, k_ord_cat+k_cont:k_ord_cat+k_cont+k_unord]

    coeff_cont, coeff_ord, coeff_unord = coefficients(rng, *k_all)

    xb_np = (x_cont @ coeff_cont + x_ord @ coeff_ord + x_unord @ coeff_unord
             ).reshape(-1, 1)

    xb_np = np.sin(xb_np).reshape(-1, 1)
    xb_np /= np.std(xb_np)

    iate_all = get_iate(rng, x_np[:, :k_cont], iate_type, no_effect=no_effect)
    iate = np.repeat(iate_all, treat-1, axis=1)

    y_0 = xb_np + noise_y0
    y_0_se = np.abs(noise_y0) / 2
    y_1_se = np.abs(noise_y1) / 2
    y_1 = xb_np + iate + noise_y1  # Same expected effects for all treatments
    ite = y_1 - y_0
    y_pot = np.concatenate((y_0, y_1), axis=1)
    y_pot_se = np.concatenate((y_0_se, y_1_se), axis=1)

    # Selecting the corresponding column values for the observed outcomes
    y_obs = y_pot[np.arange(obs), d_np.flatten()].reshape(-1, 1)

    if plot_iate:
        plot_pot_iate(xb_np, iate, (y_0, y_1), (noise_y0, noise_y1))
    return y_pot, y_pot_se, iate, ite, y_obs


def plot_pot_iate(x_indx, iate, y_pot, noise):
    """Plot heterogeneity."""
    y_0, y_1 = y_pot
    noise_y0, noise_y1 = noise
    labels = (('IATE', 'IATE_Noise'), ('Y0', 'Y1', 'EY0|X', 'EY1|X'))
    dotsize = 3
    colors = ('b', 'r', 'g', 'black')
    label_x = 'X * beta'
    label_y = ('Effect', 'Potential outcomes')
    titel = ('Effects with and without noise', 'Potential outcomes')
    ind_sort = np.argsort(x_indx, axis=0).flatten()
    xb_s = x_indx[ind_sort, :]
    iates = iate[ind_sort, 0]
    y_0s, noise_y0s = y_0[ind_sort, 0], noise_y0[ind_sort, 0]
    y_1s, noise_y1s = y_1[ind_sort, 0], noise_y1[ind_sort, 0]
    _, ax1 = plt.subplots()
    _, ax2 = plt.subplots()
    ax1.plot(xb_s, iates, label=labels[0][0], c=colors[0], linewidth=1)
    ax1.scatter(xb_s, y_1s-y_0s, label=labels[0][1], c=colors[1],
                s=dotsize)
    ax2.scatter(xb_s, y_0s, label=labels[1][0], c=colors[0], s=dotsize)
    ax2.scatter(xb_s, y_1s, label=labels[1][1], c=colors[1], s=dotsize)
    ax2.plot(xb_s, y_0s-noise_y0s, label=labels[1][2], c=colors[2],
             linewidth=1)
    ax2.plot(xb_s, y_1s-noise_y1s, label=labels[1][3], c=colors[3],
             linewidth=1)
    ax1.legend()
    ax2.legend()
    ax1.set_ylabel(label_y[0])
    ax1.set_xlabel(label_x)
    ax1.set_title(titel[0])
    ax2.set_ylabel(label_y[0])
    ax2.set_xlabel(label_x)
    ax2.set_title(titel[1])
    plt.show()
    plt.close()


def get_iate(rng, x_np, iate_type, no_effect=False):
    """Compute IATE."""
    def awsinglefunct(x_0):
        """Compute the single function of WA transformation."""
        return (1 + 1 / (1 + np.exp(-20 * (x_0 - 1/3))))

    cols = x_np.shape[1]
    coeff, _, _ = coefficients(rng, cols)
    index = (x_np @ coeff).reshape(-1, 1)
    if iate_type == 'linear':
        iate = index
    elif iate_type == 'nonlinear':
        iate = logistic.cdf(index, loc=0, scale=1) - 0.5
    elif iate_type == 'quadratic':
        iate = (index**2 - 1.25) / np.sqrt(3)
    elif iate_type == 'WagerAthey':
        x_0 = (x_np[:, 0] + np.sqrt(12)/2) / (np.sqrt(12))  # 1st X usually
        x_1 = (x_np[:, 1] + np.sqrt(12)/2) / (np.sqrt(12))  # uniform
        iate = (awsinglefunct(x_0) * awsinglefunct(x_1)).reshape(-1, 1)
        iate -= 2.8
    iate *= 1
    iate += 1
    if no_effect:
        iate *= 0

    return iate


def coefficients(rng, k_cont, k_ord_cat=None, k_unord=None):
    """Make coefficients for different variable types."""
    coeff_cont = np.linspace(1, 1/k_cont, k_cont)
    coeff_ord = None if k_ord_cat is None else np.linspace(1, 1/k_ord_cat,
                                                           k_ord_cat)
    coeff_unord = None if k_unord is None else rng.normal(loc=0, scale=1,
                                                          size=k_unord)
    return coeff_cont, coeff_ord, coeff_unord


def treatment_d(rng, no_treat, k_all, x_np, inst_np, strength_iv=1,
                correlation_x='middle'):
    """Create treatment variable."""
    noise = (rng.normal(loc=0, scale=1, size=(len(x_np), 1))
             + (inst_np - 0.5) * strength_iv)
    k_cont, k_ord_cat, k_unord = k_all
    x_cont = x_np[:, :k_cont]
    x_ord = x_np[:, k_cont:k_ord_cat+k_cont]
    x_unord = x_np[:, k_ord_cat+k_cont:k_ord_cat+k_cont+k_unord]

    coeff_cont, coeff_ord, coeff_unord = coefficients(rng, *k_all)

    xb_np = (x_cont @ coeff_cont + x_ord @ coeff_ord + x_unord @ coeff_unord
             ).reshape(-1, 1)
    if correlation_x == 'high':
        d_index = xb_np / np.std(x_np) + noise * 1.5
    else:
        d_index = xb_np / np.std(x_np) * 2 + noise

    cuts = np.quantile(
        d_index, np.linspace(1-1/no_treat, 1/no_treat, no_treat-1))
    d_np = np.digitize(d_index, cuts)

    return d_np


def covariates_x(rng: np.random.default_rng,
                 k_all: int,
                 obs: int,
                 correlation: str = 'middle') -> np.array:
    """Randomly sample the covariates."""
    k_cont, k_ord_cat, k_unord = k_all
    k_uni = round(k_cont/2)
    k_norm = k_cont - k_uni
    k_dumm = round(k_ord_cat / 2)
    k_ord = k_ord_cat - k_dumm

    if correlation == 'low':   # all features uncorrelated
        x_uniform = rng.uniform(low=-np.sqrt(12)/2, high=np.sqrt(12)/2,
                                size=(obs, k_uni))
        x_normal = rng.normal(loc=0, scale=1, size=(obs, k_norm))

    elif correlation == 'high':  # All continuous features correlated
        first_column = np.linspace(1, 0, k_uni+k_norm)
        toeplitz_covariance = toeplitz(first_column)
        x_uninorm = rng.multivariate_normal(mean=np.zeros(k_uni+k_norm),
                                            cov=toeplitz_covariance,
                                            size=obs)
        x_uniform = (norm.cdf(x_uninorm[:, :k_uni]) - 0.5) * np.sqrt(12)
        x_normal = x_uninorm[:, k_uni:]
    else:  # 'middle'     All normal features correlated
        x_uniform = rng.uniform(low=-np.sqrt(12)/2, high=np.sqrt(12)/2,
                                size=(obs, k_uni))
        first_column = np.linspace(1, 0, k_norm)
        toeplitz_covariance = toeplitz(first_column)
        x_normal = rng.multivariate_normal(mean=np.zeros(k_norm),
                                           cov=toeplitz_covariance,
                                           size=obs)

    grid_ord = (0.1, 0.3, 0.5, 0.7, 0.9)
    grid_unord = (0.1, 0.2, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9)
    uniform_dumm = rng.uniform(low=0, high=1, size=(obs, k_dumm))
    uniform_ord = rng.uniform(low=0, high=1, size=(obs, k_ord))
    uniform_unord = rng.uniform(low=0, high=1, size=(obs, k_unord))
    if correlation in ('low', 'middle',):
        x_dumm = np.int8(uniform_dumm > 0.5)
        x_ord = np.digitize(uniform_ord, grid_ord)
        x_unord = np.digitize(uniform_unord, grid_unord)
    else:
        add_var = (x_uniform[:, 1] + x_normal[:, 1] + x_normal[:, 2]
                   ).reshape(-1, 1)
        x_dumm = np.int8(uniform_dumm + add_var / 3 > 0.5)
        x_ord = np.digitize(uniform_ord + add_var / 6, grid_ord)
        x_unord = np.digitize(uniform_unord + add_var / 6, grid_unord)

    x_np = np.concatenate((x_uniform, x_normal, x_dumm, x_ord, x_unord), axis=1)
    return x_np


def descriptive_stats(train_df: pd.DataFrame, pred_df: pd.DataFrame,
                      name_dict: dict, x_name: (str, None) = None):
    """Get descriptive statistics of data generating process."""
    width = 100
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', width)
    print('Training data')
    print(round(train_df.describe().transpose(), 2), '\n', width * '-', '\n')
    if x_name:
        print('\nCorrelation matrix of features')
        print((train_df[x_name].corr() * 100).round().astype(int))

    print('\nPrediction data')
    print(round(pred_df.describe().transpose(), 2), '\n', width * '-', '\n')
    if x_name:
        print('\nCorrelation matrix of features')
        print((pred_df[x_name].corr() * 100).round().astype(int))

    iate_np = train_df[name_dict['iate_name']].to_numpy()
    str1 = f'True ATE (prediction) {np.mean(iate_np):.3f} '
    mean_y = train_df[[name_dict['y_name'], *name_dict['y_pot_name'],
                       *name_dict['ite_name'], *name_dict['iate_name'],
                       name_dict['d_name']]].groupby(
                           [name_dict['d_name']]).mean()
    count_d = train_df[name_dict['d_name']].value_counts(sort=False)
    str2 = 'Mean of outcome in different treatment groups:\n'
    print(str1, '\n')
    print(str2, round(mean_y, 3), '\n\nObservations: ', count_d, '\n')
    print('-' * width, '\n')
