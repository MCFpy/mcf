"""Created on Wed May  1 16:35:19 2024.

Functions for correcting scores w.r.t. protected variables.

@author: MLechner

# -*- coding: utf-8 -*-
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d

from mcf import mcf_estimation_generic_functions as mcf_gf
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import optpolicy_fair_add_functions as optp_fair_add


def adjust_scores(optp_, data_df, seed=1246546):
    """Remove effect of protected variables from policy score."""
    gen_dic, fair_dic = optp_.gen_dict, optp_.fair_dict
    # Change data to numpy arrays
    scores_np = data_df[optp_.var_dict['polscore_name']].to_numpy()
    protect_np = data_df[optp_.var_dict['protected_name']].to_numpy()
    if optp_.var_dict['material_name']:
        material_np = data_df[optp_.var_dict['material_name']].to_numpy()
    else:
        material_np = None

    if gen_dic['with_output']:
        txt_report = ('\nMethod selected for Fairness adjustment: '
                      f'{fair_dic["adj_type"]}')
        if gen_dic['with_output']:
            mcf_ps.print_mcf(gen_dic, txt_report, summary=True)
    else:
        txt_report = ''
    # Get fairness adjusted scores
    fair_score_np, txt = fair_score_fct(
        optp_, scores_np, optp_.var_dict['polscore_name'], protect_np,
        material_np, seed=seed, title='')

    txt_report += txt

    # Convert numpy array of fair scores to pandas dataframe
    fairscore_name = [name + '_fair'
                      for name in optp_.var_dict['polscore_name']]
    fair_score_df = pd.DataFrame(fair_score_np, columns=fairscore_name)
    data_df = data_df.reset_index(drop=True)
    data_fair_df = pd.concat((data_df, fair_score_df), axis=1)

    # Descriptive statistics
    if gen_dic['with_output']:
        txt_report += optp_fair_add.fair_stats(optp_, data_fair_df,
                                               fairscore_name)

    if fair_dic['consistency_test']:
        tests_dict, text = test_for_consistency(
            optp_, fair_score_np, fairscore_name, scores_np, protect_np,
            material_np, seed=seed+124567, title='Consistency test - ')
        txt_report += text
    else:
        tests_dict = {}

    optp_.report['fairscores_build_stats'] = txt_report

    # Change the names of variables (in particular scores) to be used for
    # policy learning.
    optp_fair_add.change_variable_names_fair(optp_, fairscore_name)

    return data_fair_df, fairscore_name, tests_dict


def fair_score_fct(optp_, scores_np, scores_name, protect_np, material_np,
                   seed=12345567, title=''):
    """Use one of the different fairness adjustment methods."""
    if optp_.fair_dict['adj_type'] in ('Mean', 'MeanVar',):
        fair_score_np, txt = residualisation(
            optp_, scores_np, scores_name, protect_np, material_np, seed=seed,
            title=title)
    elif optp_.fair_dict['adj_type'] == 'Quantiled':
        fair_score_np, txt = quantalisation(
            optp_, scores_np, protect_np, material_np, seed=seed, title=title)
    else:
        raise ValueError('Invalid method selected for fairness adjustment.')
    return fair_score_np, txt


def quantalisation(optp_, scores_np, protect_np, material_np, seed=1246546,
                   title=''):
    """Adjust by quantalisation similar to Strack & Yang (2024)."""
    txt = ''
    disc_methods = optp_.fair_dict['discretization_methods']
    if optp_.gen_dict['with_output']:
        if title is None or title == '':
            print('\nComputing fairness adjustments')
        else:
            print('\n' + title + 'computing fairness adjustments')
    # Check if materially relevant features should be treated as discrete
    if optp_.fair_dict['material_disc_method'] == 'NoDiscretization':
        if (material_np is None or (len(np.unique(material_np))
                                    < optp_.fair_dict['material_max_groups'])):
            optp_.fair_dict['material_disc_method'] = optp_.fair_dict[
                'default_disc_method']
            txt += ('\nMaterial relevant features have no or only a few '
                    'values. Discretization method changed to '
                    f'{optp_.fair_dict["material_disc_method"]}'
                    )

    # Check if protected features should be treated as discrete
    if optp_.fair_dict['protected_disc_method'] == 'NoDiscretization':
        if (len(np.unique(protect_np)) < optp_.fair_dict[
                'protected_max_groups']):
            optp_.fair_dict['protected_disc_method'] = optp_.fair_dict[
                'default_disc_method']
            txt += ('\nProtected features have no or only a few '
                    'values. Discretization method changed to '
                    f'{optp_.fair_dict["protected_disc_method"]}'
                    )

    # Discretize if needed, otherwise no change of data
    protect_np, material_np, txt_report = optp_fair_add.data_quantilized(
        optp_, protect_np, material_np, seed)
    txt_report += txt

    if ((optp_.fair_dict['protected_disc_method'] in disc_methods)
        and ((material_np is None) or (optp_.fair_dict['material_disc_method']
                                       in disc_methods))):
        # No density estimation needed
        fair_score_np, txt = within_cell_quantilization(scores_np, protect_np,
                                                        material_np)
    else:
        fair_score_np, txt = kernel_quantilization(
            optp_, scores_np, protect_np, material_np)

    txt_report += txt
    if optp_.gen_dict['with_output'] and title == '':
        mcf_ps.print_mcf(optp_.gen_dict, txt_report, summary=True)
    return fair_score_np, txt_report


def kernel_quantilization(optp_, scores_np, protected_np, material_np):
    """Do within cell quantilization for arbitrary materially rel. features."""
    disc_methods = optp_.fair_dict['discretization_methods']
    txt_report = ('\nQuantile based method by Strack & Yang (2024) used for '
                  'materially relevant features with many variables.')

    no_of_scores = scores_np.shape[1]
    fair_score_np = scores_np.copy()
    no_eval_point = 2000

    # Case of discrete protected and discrete materially relevant is dealt with
    # in other procedure

    if optp_.fair_dict['protected_disc_method']:
        vals_prot = np.unique(protected_np, return_counts=False)
    if material_np is not None and optp_.fair_dict['material_disc_method']:
        vals_material = np.unique(material_np, return_counts=False)

    if protected_np.shape[1] == 1:
        protected_np = protected_np.reshape(-1, 1)
    if material_np is not None and material_np.shape[1] == 1:
        material_np = material_np.reshape(-1, 1)

    for idx_score in range(no_of_scores):
        score = scores_np[:, idx_score]
        if np.std(score) < 1e-8:
            continue
        quantiles_z = np.zeros_like(score)

        # Kernel density estimation conditional on protected variables
        score_all_grid = get_grid(score, no_eval_point)
        if optp_.fair_dict['material_disc_method'] in disc_methods:
            if material_np is None:
                quantiles_z, _ = calculate_quantiles_kde(
                    score, protected_np, score_all_grid)
            else:
                for val in vals_material:
                    mask = material_np.reshape(-1) == val
                    score_grid = get_grid(score[mask], no_eval_point)
                    quantiles_z[mask], _ = calculate_quantiles_kde(
                        score[mask], protected_np[mask], score_grid)
        elif optp_.fair_dict['protected_disc_method'] in disc_methods:
            for val in vals_prot:
                mask = protected_np.reshape(-1) == val
                # Find quantile in conditional data
                if material_np is None:
                    data_cond_np = None
                else:
                    data_cond_np = material_np[mask].copy()
                score_grid = get_grid(score[mask], no_eval_point)
                quantiles_z[mask], _ = calculate_quantiles_kde(
                    score[mask], data_cond_np, score_grid)

        else:  # Both groups of variables treated as continuous
            if material_np is None:
                data_cond_np = protected_np
            else:
                data_cond_np = np.concatenate((protected_np, material_np),
                                              axis=1)
            quantiles_z, _ = calculate_quantiles_kde(score, data_cond_np,
                                                     score_all_grid)

        # Translate quantile to values of distribution conditional on
        # materially relevant variables only
        if optp_.fair_dict['material_disc_method'] in disc_methods:
            if material_np is None:
                fair_score_np[:, idx_score] = values_from_quantiles(
                    score, quantiles_z)
            else:
                for val in vals_material:
                    mask = material_np.reshape(-1) == val
                    fair_score_np[mask, idx_score] = values_from_quantiles(
                        score[mask], quantiles_z[mask])
        else:
            _, fair_score_np[:, idx_score] = calculate_quantiles_kde(
                score, material_np, score_all_grid, quantile_data=quantiles_z)

    return fair_score_np, txt_report


def calculate_quantiles_kde(score, data_cond, score_grid, quantile_data=None):
    """Calculate the quantiles using Kernel density estimation."""
    quantile_values = quantile_data is not None
    if quantile_values:
        y_at_quantile = np.zeros_like(score)
    else:
        quantile_at_y = np.zeros_like(score)

    num_points = len(score_grid)

    # Fit KDE for the joint density p(x, y)
    if data_cond.shape[1] == 1:
        data_cond = data_cond.reshape(-1, 1)
    if data_cond is None:
        joint_data = score.reshape(-1, 1)
    else:
        joint_data = np.concatenate((data_cond, score.reshape(-1, 1)), axis=1)
    kde_joint = KernelDensity(kernel='epanechnikov', bandwidth='silverman'
                              ).fit(joint_data)
    kde_marg = KernelDensity(kernel='epanechnikov', bandwidth='silverman'
                             ).fit(data_cond)

    kde_marg_data = np.exp(kde_marg.score_samples(data_cond))  # All datapoints

    for idx, score_i in enumerate(score):
        xy_grid = np.hstack([np.tile(data_cond[idx, :], (num_points, 1)),
                             score_grid])
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
                cdf_idx, score_grid.flatten(), bounds_error=False,
                fill_value=(score_grid[0, 0], score_grid[-1, 0]))
            y_at_quantile[idx] = quantile_func(quantile_data[idx])
            quantile_at_y = None
        else:
            cdf_func = interp1d(score_grid.flatten(), cdf_idx,
                                bounds_error=False, fill_value=(0, 1))
            quantile_at_y[idx] = cdf_func(score_i)
            y_at_quantile = None

    return quantile_at_y, y_at_quantile


def get_grid(data, no_eval_point):
    """Get evaluation grid for densities."""
    data_min, data_max = data.min(), data.max()
    grid = np.linspace(data_min, data_max, no_eval_point).reshape(-1, 1)
    return grid


def within_cell_quantilization(scores_np, protected_np, material_np):
    """Do within cell quantilization for discrete univariate features."""
    txt_report = ('\nQuantile based method by Strack & Yang (2024) used for '
                  'discrete features.')
    no_of_scores = scores_np.shape[1]
    if material_np is None:
        material_np = np.ones((protected_np.shape[0], 1))
    material_values = np.unique(material_np, return_counts=False)
    fair_score_np = scores_np.copy()

    for mat_val in material_values:
        indices_mat = np.where(material_np == mat_val)[0]
        if indices_mat.size == 0:
            continue
        prot_mat = protected_np[indices_mat].reshape(-1)
        vals_prot_mat = np.unique(prot_mat, return_counts=False)
        if len(vals_prot_mat) == 1:
            continue

        for idx_score in range(no_of_scores):
            score_mat = scores_np[indices_mat, idx_score]
            if np.std(score_mat) < 1e-8:
                continue
            quantiles = np.zeros_like(score_mat)
            for val in vals_prot_mat:
                mask = prot_mat == val
                # Find quantile in conditional data
                quantiles[mask] = calculate_quantiles(score_mat[mask])
                # Translate quantile to values of distribution conditional on
                # materially relevant variables
            fair_score_np[indices_mat, idx_score] = values_from_quantiles(
                score_mat, quantiles)

    return fair_score_np, txt_report


def calculate_quantiles(data):
    """Calculate quantiles for each value in the dataset."""
    data_sort = sorted(data)  # Sort the data
    rank = np.empty_like(data)
    for idx, value in enumerate(data):
        rank[idx] = np.searchsorted(data_sort, value, side='right')  # Find rank
    return rank / len(data)


def values_from_quantiles(data, quantiles):
    """Get the values from the quantiles."""
    d_sorted = np.sort(data)  # Sort the empirical distribution
    obs = len(d_sorted)       # Number of data points
    indices = np.int64(np.round(quantiles * (obs - 1)))  # Compute the indices
    return d_sorted[indices]


def residualisation(optp_, scores_np, scores_name, protect_np, material_np,
                    seed=1246546, title=''):
    """Adjust by residualisation."""
    # Info and tuning parameters
    obs, no_of_scores = scores_np.shape
    boot, cross_validation_k, txt_report = 5, 1000, ''

    # Define all conditioning variables for regressions below
    if material_np is None:
        x_cond_np = protect_np
        with_material_x = False
    else:
        x_cond_np = np.concatenate((protect_np, material_np), axis=1)
        with_material_x = True
    # Submethod: Adjust only mean, or mean and variance
    if optp_.fair_dict['adj_type'] == 'Mean':
        adjust_ment_set = ('mean', )
    else:
        adjust_ment_set = ('mean', 'variance')

    # Numpy arrays to save conditional means and variance for each score
    y_mean_cond_x_np = np.zeros_like(scores_np)
    if with_material_x:
        y_mean_cond_mat_np = np.zeros_like(scores_np)
    if optp_.fair_dict['adj_type'] == 'MeanVar':
        y_var_cond_x_np = np.zeros_like(scores_np)
        if with_material_x:
            y_var_cond_mat_np = np.zeros_like(scores_np)
    # Loop over scores to obtain prediction of conditonal expectation of y
    for idx in range(no_of_scores):
        for mean_var in adjust_ment_set:
            if optp_.gen_dict['with_output']:
                print('\n' + title + f'Currently adjusting {mean_var} of '
                      f'{scores_name[idx]}')

            # Define dependent variable in regression & check if regr. is needed
            if mean_var == 'mean':
                # Adjust conditional mean by residualisation
                y_np = scores_np[:, idx]  # Dependent variable in regression

                # No regression if there is no variation in the score
                if np.std(y_np) < 1-8:
                    y_mean_cond_x_np[:, idx] = y_np.copy()
                    if with_material_x:
                        y_mean_cond_mat_np[:, idx] = y_np.copy()
                    continue
            elif mean_var == 'variance':
                # Adjust conditional variance by rescaling
                # Low of total variance: Var(Y|X)=E(Y**2|X)+(EY|X)**2
                y_np = scores_np[:, idx]**2
                y_mean_x_2 = y_mean_cond_x_np[:, idx]**2
                if with_material_x:
                    y_mean_mat_2 = y_mean_cond_mat_np[:, idx]**2
                # No regression if there is no variation in the score
                if np.std(y_np) < 1-8:
                    y_var_cond_x_np[:, idx] = y_np - y_mean_x_2
                    if with_material_x:
                        y_var_cond_mat_np[:, idx] = y_np - y_mean_mat_2
                    continue
            else:
                raise ValueError('Wrong adjustement method.')

            # Find best estimator for specific score (only using all covars)
            (estimator, params, best_label, _, transform_x, txt_mse
             ) = mcf_gf.best_regression(
                x_cond_np,  y_np.ravel(),
                estimator=optp_.fair_dict['regression_method'],
                boot=boot, seed=seed+12435,
                max_workers=optp_.int_dict['mp_parallel'],
                cross_validation_k=cross_validation_k,
                absolute_values_pred=mean_var == 'variance')
            if with_material_x:
                (estimator_m, params_m, best_label_m, _, transform_x_m,
                 txt_mse_m) = mcf_gf.best_regression(
                    material_np,  y_np.ravel(),
                    estimator=optp_.fair_dict['regression_method'],
                    boot=boot, seed=seed+12435,
                    max_workers=optp_.int_dict['mp_parallel'],
                    cross_validation_k=cross_validation_k,
                    absolute_values_pred=mean_var == 'variance')

            if optp_.gen_dict['with_output']:
                text = ('\n' + title + f'Adjustment for {mean_var} of '
                        f'{scores_name[idx]}:')
                txt_mse = text + txt_mse
                if with_material_x:
                    txt_mse += '\n' + title + 'Short regression:' + txt_mse_m
                mcf_ps.print_mcf(optp_.gen_dict, txt_mse, summary=False)
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
    residuum_np = scores_np - y_mean_cond_x_np

    # Adjust variance as well
    if optp_.fair_dict['adj_type'] == 'MeanVar':

        # Conditional standard deviation (must be non-zero)
        bound_var = 1e-6
        y_std_cond_x_np = optp_fair_add.var_to_std(y_var_cond_x_np, bound_var)
        if with_material_x:
            y_std_cond_mat_np = optp_fair_add.var_to_std(y_var_cond_mat_np,
                                                         bound_var)

        # Avoid too extreme values when scaling
        bound = 0.05
        y_std_cond_x_np = optp_fair_add.bound_std(y_std_cond_x_np, bound,
                                                  no_of_scores)

        # Remove predictability due to heteroscedasticity
        residuum_np /= y_std_cond_x_np

        # Rescale to retain about the variability of the original scores
        std_resid = np.std(residuum_np, axis=0).reshape(-1, 1).T
        residuum_np /= std_resid
        if with_material_x:
            residuum_np *= y_std_cond_mat_np
        else:
            residuum_np *= np.mean(
                y_std_cond_x_np, axis=0).reshape(-1, 1).T

    # Correct scores, but keep score specific (conditional) mean
    if with_material_x:
        fair_score_np = residuum_np + y_mean_cond_mat_np
    else:
        fair_score_np = residuum_np + np.mean(
            y_mean_cond_x_np, axis=0).reshape(-1, 1).T

    return fair_score_np, txt_report


def test_for_consistency(optp_, fair_score_np, fairscore_name, scores_np,
                         protect_np, material_np, seed=124567,
                         title='Consistency test'):
    """Test for consistency.

    Compare difference of fair scores to difference made explicitly fair.
    Maybe use in descriptive part.
    Also define flag because it is computationally expensive since additional
    scores have to be made fair.
    """
    score_name, test_dic = optp_.var_dict['polscore_name'], {}

    # Find valid combinations of scores that can be tested
    (test_scores_np, test_fair_scores_np, test_scores_name, _
     ) = optp_fair_add.score_combinations(fair_score_np, fairscore_name,
                                          scores_np, score_name)

    # Fairness adjust the score differences
    test_scores_adj_np, _ = fair_score_fct(
        optp_, test_scores_np, test_scores_name, protect_np, material_np,
        seed=seed, title=title
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
    if optp_.gen_dict['with_output']:
        txt1 = ('\nTest for consistency of different fairness normalisations:'
                '\n    - Compare difference of adjusted scores to '
                'adjusted difference of scores')
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
            'Ideal value is 100%.')
        mcf_ps.print_mcf(optp_.gen_dict, '\n' + '-' * 100 + txt + txt1,
                         summary=True)
        txt += '\n' + txt1
    return test_dic, txt
