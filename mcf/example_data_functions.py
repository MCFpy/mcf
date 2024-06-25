"""
Created on Wed Dec 21 15:37:16 2022.

# -*- coding: utf-8 -*-
@author: MLechner
"""

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

from scipy.stats import logistic


def example_data(obs_y_d_x_iate=1000, obs_x_iate=1000, no_features=20,
                 no_treatments=3, type_of_heterogeneity='WagerAthey',
                 seed=12345, descr_stats=True):
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
    name_dict['iate_name'] = ['iate' + str(i+1) + 'vs0'
                              for i in range(no_treatments-1)]
    name_dict['ite_name'] = ['ite' + str(i+1) + 'vs0'
                             for i in range(no_treatments-1)]
    name_dict['zero_name'] = 'zero'
    name_dict['id_name'] = 'id'
    name_dict['cluster_name'] = 'cluster'
    name_dict['weight_name'] = 'weight'
    name_dict['x_name_ord'] = ['x_cont' + str(i) for i in range(k_cont)]
    name_dict['x_name_ord'].extend(['x_ord' + str(i) for i in range(k_ord_cat)])
    name_dict['x_name_unord'] = ['x_unord' + str(i) for i in range(k_unord)]
    x_name = [*name_dict['x_name_ord'], *name_dict['x_name_unord']]
    rng = np.random.default_rng(seed=seed)

    # Features
    x_train = covariates_x(rng, k_all, obs_y_d_x_iate)
    x_pred = covariates_x(rng, k_all, obs_x_iate)

    # Treatment
    d_train = treatment_d(rng, no_treatments, k_all, x_train)
    d_pred = treatment_d(rng, no_treatments, k_all, x_pred)

    # Potential and observed outcomes
    y_pot_train, iate_train, ite_train, y_train = get_outcomes(
        rng, x_train, d_train, k_all, type_of_heterogeneity,
        plot_iate=descr_stats)
    y_pot_pred, iate_pred, ite_pred, _ = get_outcomes(
        rng, x_pred, d_pred, k_all, type_of_heterogeneity, plot_iate=False)

    id_train = np.arange(obs_y_d_x_iate).reshape(-1, 1)
    id_pred = np.arange(obs_x_iate).reshape(-1, 1)

    cluster_train = rng.integers(0, 100, size=obs_y_d_x_iate).reshape(-1, 1)
    cluster_pred = rng.integers(0, 100, size=obs_x_iate).reshape(-1, 1)
    weight_train = rng.uniform(0.5, 1.5, size=obs_y_d_x_iate).reshape(-1, 1)
    weight_pred = rng.uniform(0.5, 1.5, size=obs_x_iate).reshape(-1, 1)

    train_np = np.concatenate(
        (y_train, d_train, x_train, y_pot_train, iate_train, ite_train,
         id_train, cluster_train, weight_train,
         np.zeros((len(ite_train), 1))), axis=1)
    train_df = pd.DataFrame(data=train_np,
                            columns=(name_dict['y_name'],
                                     name_dict['d_name'],
                                     *x_name,
                                     *name_dict['y_pot_name'],
                                     *name_dict['iate_name'],
                                     *name_dict['ite_name'],
                                     name_dict['id_name'],
                                     name_dict['cluster_name'],
                                     name_dict['weight_name'],
                                     name_dict['zero_name']))
    pred = np.concatenate((d_pred, x_pred, y_pot_pred, iate_pred, ite_pred,
                          id_pred, cluster_pred, weight_pred,
                          np.zeros((len(ite_pred), 1))), axis=1)
    pred_df = pd.DataFrame(data=pred, columns=(name_dict['d_name'],
                                               *x_name,
                                               *name_dict['y_pot_name'],
                                               *name_dict['iate_name'],
                                               *name_dict['ite_name'],
                                               name_dict['id_name'],
                                               name_dict['cluster_name'],
                                               name_dict['weight_name'],
                                               name_dict['zero_name']))
    if descr_stats:
        descriptive_stats(train_df, pred_df, name_dict)
    return train_df, pred_df, name_dict


def get_observed_outcome(y_pot, d_np):
    """Get the observed values from the potentials."""
    y_np = y_pot[np.arange(len(y_pot)), d_np[:, 0]].reshape(-1, 1)
    return y_np


def get_outcomes(rng, x_np, d_np, k_all, iate_type='WagerAthey',
                 plot_iate=True):
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

    iate_all = get_iate(rng, x_np[:, :k_cont], iate_type)
    iate = np.repeat(iate_all, treat-1, axis=1)

    y_0 = xb_np + noise_y0
    y_1 = xb_np + iate + noise_y1  # Same expected effects for all treatments
    ite = y_1 - y_0
    y_pot = np.concatenate((y_0, y_1), axis=1)

    # Selecting the corresponding column values for the observed outcomes
    y_obs = y_pot[np.arange(obs), d_np.flatten()].reshape(-1, 1)

    if plot_iate:
        plot_pot_iate(xb_np, iate, (y_0, y_1), (noise_y0, noise_y1))
    return y_pot, iate, ite, y_obs


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


def get_iate(rng, x_np, iate_type):
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
    return iate


def coefficients(rng, k_cont, k_ord_cat=None, k_unord=None):
    """Make coefficients for different variable types."""
    coeff_cont = np.linspace(1, 1/k_cont, k_cont)
    coeff_ord = None if k_ord_cat is None else np.linspace(1, 1/k_ord_cat,
                                                           k_ord_cat)
    coeff_unord = None if k_unord is None else rng.normal(loc=0, scale=1,
                                                          size=k_unord)
    return coeff_cont, coeff_ord, coeff_unord


def treatment_d(rng, no_treat, k_all, x_np):
    """Create treatment variable."""
    noise = rng.normal(loc=0, scale=1, size=(len(x_np), 1))
    k_cont, k_ord_cat, k_unord = k_all
    x_cont = x_np[:, :k_cont]
    x_ord = x_np[:, k_cont:k_ord_cat+k_cont]
    x_unord = x_np[:, k_ord_cat+k_cont:k_ord_cat+k_cont+k_unord]

    coeff_cont, coeff_ord, coeff_unord = coefficients(rng, *k_all)

    xb_np = (x_cont @ coeff_cont + x_ord @ coeff_ord + x_unord @ coeff_unord
             ).reshape(-1, 1)
    d_index = xb_np / np.std(x_np) * 2 + noise

    cuts = np.quantile(
        d_index, np.linspace(1-1/no_treat, 1/no_treat, no_treat-1))
    d_np = np.digitize(d_index, cuts)

    return d_np


def covariates_x(rng, k_all, obs):
    """Randomly sample the covariates."""
    k_cont, k_ord_cat, k_unord = k_all
    k_uni = round(k_cont/2)
    k_norm = k_cont - k_uni
    k_dumm = round(k_ord_cat / 2)
    k_ord = k_ord_cat - k_dumm

    x_uniform = rng.uniform(low=-np.sqrt(12)/2, high=np.sqrt(12)/2,
                            size=(obs, k_uni))
    # x_normal = rng.normal(loc=0, scale=1, size=(obs, k_norm))
    first_column = np.linspace(1, 0, k_norm)
    toeplitz_covariance = toeplitz(first_column)
    x_normal = rng.multivariate_normal(mean=np.zeros(k_norm),
                                       cov=toeplitz_covariance,
                                       size=obs)
    x_dumm = np.int8(rng.uniform(low=0, high=1, size=(obs, k_dumm)) > 0.5)
    grid = (0.1, 0.3, 0.5, 0.7, 0.9)
    x_ord = np.digitize(rng.uniform(low=0, high=1, size=(obs, k_ord)), grid)
    grid = (0.1, 0.2, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9)
    x_unord = np.digitize(rng.uniform(low=0, high=1, size=(obs, k_unord)), grid)
    x_np = np.concatenate((x_uniform, x_normal, x_dumm, x_ord, x_unord), axis=1)
    return x_np


def descriptive_stats(train_df, pred_df, name_dict):
    """Get descriptive statistics of data generating process."""
    width = 100
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', width)
    print('Training data')
    print(round(train_df.describe().transpose(), 2), '\n', width * '-', '\n')
    print('Prediction data')
    print(round(pred_df.describe().transpose(), 2), '\n', width * '-', '\n')
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
