"""
Procedures needed for IATE estimation.

Created on Thu Mar 17 09:52:10 2022.

@author: MLechner

# -*- coding: utf-8 -*-
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as sct
import matplotlib.pyplot as plt
from matplotlib import cm

from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est


def print_iate(iate, iate_se, iate_p, effect_list, v_dict, c_dict,
               reg_round=True):
    """Print statistics for the two types of IATEs.

    Parameters
    ----------
    iate : 4D Numpy array. Effects. (obs x outcome x effects x type_of_effect)
    iate_se : 4D Numpy array. Standard errors.
    iate_t : 4D Numpy array.
    iate_p : 4D Numpy array.
    effect_list : List. Names of effects.
    v_dict : Dict. Variables.
    c_dict : Dict. Control paramaters.
    reg_round : Boolean. True if regular estimation round. Default is True.

    Returns
    -------
    None.

    """
    no_outcomes = np.size(iate, axis=1)
    n_obs = len(iate)
    str_f, str_m, str_l = '=' * 80, '-' * 80, '- ' * 40
    print_str = '\n' + str_f + '\nDescriptives for IATE estimation\n' + str_m
    if not reg_round:
        print_str += ('\n' + str_m
                      + '\nSecond round of estimation of Eff. IATE\n' + str_l)
    for types in range(2):
        if types == 0:
            print_str += 'IATE with corresponding statistics\n' + str_l
        else:
            print_str += ('IATE minus ATE with corresponding statistics '
                          + '(weights not censored)\n' + str_l)
        for o_idx in range(no_outcomes):
            print_str += (f'\nOutcome variable: {v_dict["y_name"][o_idx]}\n'
                          + str_l)
            str1 = '        Comparison          Mean       Median      Std'
            if c_dict['iate_se_flag']:
                print_str += ('\n' + str1 + '  Effect > 0 mean(SE)  sig 10%'
                              + '  sig 5%   sig 1%')
            else:
                print_str += '\n' + str1 + '  Effect > 0'
            for jdx, effects in enumerate(effect_list):
                fdstring = (f'{effects[0]:>9.5f} vs {effects[1]:>9.5f}'
                            if c_dict['d_type'] == 'continuous' else
                            f'{effects[0]:<9} vs {effects[1]:<9} ')
                print_str += '\n' + fdstring
                est = iate[:, o_idx, jdx, types].reshape(-1)
                if c_dict['iate_se_flag']:
                    stderr = iate_se[:, o_idx, jdx, types].reshape(-1)
                    p_val = iate_p[:, o_idx, jdx, types].reshape(-1)
                print_str += (f'{np.mean(est):10.5f} {np.median(est):10.5f}'
                              + f' {np.std(est):10.5f} ')
                if c_dict['iate_se_flag']:
                    print_str += (
                        '\n' + f'{np.count_nonzero(est > 1e-15) / n_obs:6.2%}'
                        + f' {np.mean(stderr):10.5f}'
                        + f' {np.count_nonzero(p_val < 0.1)/n_obs:6.2%}'
                        + f' {np.count_nonzero(p_val < 0.05)/n_obs:6.2%}'
                        + f' {np.count_nonzero(p_val < 0.01)/n_obs:6.2%}')
                else:
                    print_str += '\n'
        print_str += '\n' + str_m + '\n'
    print_str = '\n' + str_m
    if c_dict['iate_se_flag']:
        print_str += ('\n' + gp_est.print_se_info(c_dict['cluster_std'],
                                                  c_dict['se_boot_iate']))
        print_str += gp_est.print_minus_ate_info(c_dict['w_yes'],
                                                 gate_or_iate='IATE')
    print(print_str)
    gp.print_f(c_dict['outfilesummary'], print_str)


def post_estimation_iate(file_name, iate_pot_all_name, ate_all, ate_all_se,
                         effect_list, v_dict, c_dict, v_x_type):
    """Do post-estimation analysis: correlations, k-means, sorted effects.

    Parameters
    ----------
    file_name : String. Name of file with potential outcomes and effects.
    iate_pot_all_name : Dict. Name of potential outcomes and effects.
    ate_all : 3D Numpy array. ATEs.
    ate_all_se : 3D Numpy array. Std.errors of ATEs.
    effect_list : List of list. Explanation of effects related to ATEs.
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.
    v_x_type : Dict. Variable information.

    Returns
    -------
    None.

    """

    def delete_x_with_catv(names_with_catv):
        """Delete variables which end with CATV from list."""
        return [x_name for x_name in names_with_catv
                if not x_name.endswith('CATV')]

    if c_dict['with_output'] and c_dict['verbose']:
        print_str = '\n' + '=' * 80 + '\nPost estimation analysis\n' + '-' * 80
        print(print_str)
        gp.print_f(c_dict['outfilesummary'], print_str)
    if c_dict['d_type'] == 'continuous':
        d_values = c_dict['ct_d_values_dr_np']
        no_of_treat = len(d_values)
    else:
        no_of_treat, d_values = c_dict['no_of_treat'], c_dict['d_values']
    if (c_dict['relative_to_first_group_only']
            or c_dict['d_type'] == 'continuous'):
        iate_pot_name = iate_pot_all_name[1]
        dim_all = (len(ate_all), no_of_treat-1)
        ate, ate_se = np.empty(dim_all), np.empty(dim_all)
        jdx = 0
        for idx, i_lab in enumerate(effect_list):
            if i_lab[1] == d_values[0]:  # compare to 1st treat only
                ate[:, jdx] = ate_all[:, 0, idx]
                ate_se[:, jdx] = ate_all_se[:, 0, idx]
                jdx += 1
    else:
        iate_pot_name = iate_pot_all_name[0]
        dim_all = (np.size(ate_all, axis=0), np.size(ate_all, axis=2))
        ate, ate_se = ate_all[:, 0, :], ate_all_se[:, 0, :]
    ate, ate_se = ate.reshape(-1), ate_se.reshape(-1)
    data = pd.read_csv(file_name)
    pot_y = data[iate_pot_name['names_pot_y']]      # deep copies
    iate = data[iate_pot_name['names_iate']]
    x_name = delete_x_with_catv(v_x_type.keys())
    x_dat = data[x_name]
    cint = sct.norm.ppf(c_dict['fig_ci_level'] +
                        0.5 * (1 - c_dict['fig_ci_level']))
    if c_dict['bin_corr_yes']:
        print('\n' + ('=' * 80), '\nCorrelations of effects with ... in %')
        print('-' * 80)
    label_ci = f'{c_dict["fig_ci_level"]:2.0%}-CI'
    iterator = range(2) if c_dict['iate_se_flag'] else range(1)
    no_of_names = len(iate_pot_name['names_iate'])
    eva_points = eva_points_fct(no_of_names, len(v_dict['y_name']))
    for idx in range(no_of_names):
        for imate in iterator:
            if imate == 0:
                name_eff = 'names_iate'
                ate_t, ate_se_t = ate[idx].copy(), ate_se[idx].copy()
            else:
                name_eff = 'names_iate_mate'
                ate_t = 0
            name_iate_t = iate_pot_name[name_eff][idx]
            if c_dict['iate_se_flag']:
                name_se = name_eff + '_se'
                name_iate_se_t = iate_pot_name[name_se][idx]
            else:
                name_se = name_iate_se_t = None
            titel = 'Sorted ' + name_iate_t
            # Add correlation analyis of IATEs
            if c_dict['d_type'] == 'discrete' or idx in eva_points:
                if c_dict['bin_corr_yes'] and imate == 0:
                    print('Effect:', name_iate_t, '\n' + ('-' * 80))
                    if c_dict['d_type'] == 'discrete':
                        corr = iate.corrwith(data[name_iate_t])
                        for jdx in corr.keys():
                            print(f'{jdx:<20} {corr[jdx]*100:>8.2f}')
                        print('-' * 80)
                        corr = pot_y.corrwith(data[name_iate_t])
                        for jdx in corr.keys():
                            print(f'{jdx:<20} {corr[jdx] * 100:>8.2f}')
                        print('-' * 80)
                    corr = x_dat.corrwith(data[name_iate_t])
                    corr = corr.sort_values()
                    for jdx in corr.keys():
                        if np.abs(corr[jdx].item()
                                  ) > c_dict['bin_corr_thresh']:
                            print(f'{jdx:<20} {corr[jdx] * 100:>8.2f}')
                    print('-' * 80)
                iate_temp = data[name_iate_t].to_numpy()
                iate_se_temp = (data[name_iate_se_t].to_numpy()
                                if c_dict['iate_se_flag'] else None)
                sorted_ind = np.argsort(iate_temp)
                iate_temp = iate_temp[sorted_ind]
                if c_dict['iate_se_flag']:
                    iate_se_temp = iate_se_temp[sorted_ind]
                x_values = (np.arange(len(iate_temp)) + 1)
                x_values = np.around(x_values / x_values[-1] * 100, decimals=1) 
                k = np.round(c_dict['knn_const'] * np.sqrt(len(iate_temp)) * 2)
                iate_temp = gp_est.moving_avg_mean_var(iate_temp, k, False)[0]
                if c_dict['iate_se_flag']:
                    iate_se_temp = gp_est.moving_avg_mean_var(
                        iate_se_temp, k, False)[0]
                titel_f = titel.replace(' ', '')
                file_name_jpeg = (c_dict['cs_ate_iate_fig_pfad_jpeg']
                                  + '/' + titel_f + '.jpeg')
                file_name_pdf = (c_dict['cs_ate_iate_fig_pfad_pdf']
                                 + '/' + titel_f + '.pdf')
                file_name_csv = (c_dict['cs_ate_iate_fig_pfad_csv']
                                 + '/' + titel_f + 'plotdat.csv')
                if c_dict['iate_se_flag']:
                    upper = iate_temp + iate_se_temp * cint
                    lower = iate_temp - iate_se_temp * cint
                ate_t = ate_t * np.ones(len(iate_temp))
                if imate == 0:
                    ate_upper = ate_t + (
                        ate_se_t * cint * np.ones(len(iate_temp)))
                    ate_lower = ate_t - (
                        ate_se_t * cint * np.ones(len(iate_temp)))
                line_ate, line_iate = '_-r', '-b'
                fig, axe = plt.subplots()
                if imate == 0:
                    label_t, label_r, label_y = 'IATE', 'ATE', 'Effect'
                else:
                    label_t, label_r = 'IATE-ATE', '_nolegend_'
                    label_y = 'Effect - average'
                axe.plot(x_values, iate_temp, line_iate, label=label_t)
                axe.set_ylabel(label_y)
                axe.plot(x_values, ate_t, line_ate, label=label_r)
                if imate == 0:
                    axe.fill_between(x_values, ate_upper, ate_lower,
                                     alpha=0.3, color='r', label=label_ci)
                titel_tmp = titel.replace('_iate','')
                if 'mate' in titel:
                   titel_tmp = titel_tmp.replace('mate','')
                titel_tmp = titel_tmp[:-4] + ' ' + titel_tmp[-4:]
                titel_tmp = titel_tmp.replace('vs', ' vs ')
                if 'mate' in titel:
                    titel_tmp += ' (- ATE)'
                axe.set_title(titel_tmp)
                axe.set_xlabel('Quantile of sorted IATEs')
                if c_dict['iate_se_flag']:
                    axe.fill_between(x_values, upper, lower, alpha=0.3,
                                     color='b', label=label_ci)
                axe.legend(loc=c_dict['fig_legend_loc'], shadow=True,
                           fontsize=c_dict['fig_fontsize'])
                if c_dict['post_plots']:
                    gp.delete_file_if_exists(file_name_jpeg)
                    gp.delete_file_if_exists(file_name_pdf)
                    fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
                    fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
                if c_dict['show_plots']:
                    plt.show()
                else:
                    plt.close()
                iate_temp = iate_temp.reshape(-1, 1)
                if c_dict['iate_se_flag']:
                    upper, lower = upper.reshape(-1, 1), lower.reshape(-1, 1)
                ate_t = ate_t.reshape(-1, 1)
                iate_temp = iate_temp.reshape(-1, 1)
                if imate == 0:
                    ate_upper = ate_upper.reshape(-1, 1)
                    ate_lower = ate_lower.reshape(-1, 1)
                    if c_dict['iate_se_flag']:
                        effects_et_al = np.concatenate(
                            (upper, iate_temp, lower, ate_t, ate_upper,
                             ate_lower), axis=1)
                        cols = ['upper', 'effects', 'lower', 'ate', 'ate_l',
                                'ate_u']
                    else:
                        effects_et_al = np.concatenate(
                            (iate_temp, ate_t, ate_upper, ate_lower), axis=1)
                        cols = ['effects', 'ate', 'ate_l', 'ate_u']
                else:
                    effects_et_al = np.concatenate(
                        (upper, iate_temp, lower, ate_t), axis=1)
                    cols = ['upper', 'effects', 'lower', 'ate']
                datasave = pd.DataFrame(data=effects_et_al, columns=cols)
                gp.delete_file_if_exists(file_name_csv)
                datasave.to_csv(file_name_csv, index=False)
                # density plots
                if imate == 0:
                    titel = 'Density ' + iate_pot_name['names_iate'][idx]
                    titel_f = titel.replace(' ','')
                    file_name_jpeg = (c_dict['cs_ate_iate_fig_pfad_jpeg']
                                      + '/' + titel_f + '.jpeg')
                    file_name_pdf = (c_dict['cs_ate_iate_fig_pfad_pdf']
                                     + '/' + titel_f + '.pdf')
                    file_name_csv = (c_dict['cs_ate_iate_fig_pfad_csv']
                                     + '/' + titel_f + 'plotdat.csv')
                    iate_temp = data[name_iate_t].to_numpy()
                    bandwidth = gp_est.bandwidth_silverman(iate_temp, 1)
                    dist = np.abs(iate_temp.max() - iate_temp.min())
                    low_b = iate_temp.min() - 0.1 * dist
                    up_b = iate_temp.max() + 0.1 * dist
                    grid = np.linspace(low_b, up_b, 1000)
                    density = gp_est.kernel_density(iate_temp, grid, 1,
                                                    bandwidth)
                    fig, axe = plt.subplots()
                    titel_tmp = titel.replace('_iate','')
                    titel_tmp = titel_tmp[:-4] + ' ' + titel_tmp[-4:]
                    titel_tmp = titel_tmp.replace('vs', ' vs ')
                    axe.set_title(titel_tmp)
                    axe.set_ylabel('Estimated density')
                    axe.set_xlabel('IATE')
                    axe.plot(grid, density, '-b')
                    axe.fill_between(grid, density, alpha=0.3, color='b')
                    if c_dict['post_plots']:
                        gp.delete_file_if_exists(file_name_jpeg)
                        gp.delete_file_if_exists(file_name_pdf)
                        fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
                        fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
                    if c_dict['show_plots']:
                        plt.show()
                    else:
                        plt.close()
                    density = density.reshape(-1, 1)
                    cols = ['grid', 'density']
                    grid = grid.reshape(-1, 1)
                    density = density.reshape(-1, 1)
                    effects_et_al = np.concatenate((grid, density), axis=1)
                    datasave = pd.DataFrame(data=effects_et_al, columns=cols)
                    gp.delete_file_if_exists(file_name_csv)
                    datasave.to_csv(file_name_csv, index=False)
    if c_dict['d_type'] == 'continuous':
        no_of_y = len(v_dict['y_name'])
        no_of_iate_y = round(len(iate_pot_name['names_iate']) / no_of_y)
        index_0 = range(no_of_iate_y)
        for idx_y in range(no_of_y):  # In case there are several outcomes
            for imate in iterator:
                if imate == 0:
                    name_eff, iate_label = 'names_iate', 'IATE'
                else:
                    name_eff, iate_label = 'names_iate_mate', 'IATE-ATE'
                titel = ('Dose response relative to 0 ' + iate_label + ' '
                         + v_dict['y_name'][idx_y])
                index_t = [i + no_of_iate_y * idx_y for i in index_0]
                name_iate_t = [iate_pot_name[name_eff][idx] for idx in index_t]
                iate_temp = data[name_iate_t].to_numpy()
                indices_sort = np.argsort(np.mean(iate_temp, axis=1))
                iate_temp = iate_temp[indices_sort]
                z_plt = np.transpose(iate_temp)
                fig, axe = plt.subplots(subplot_kw={"projection": "3d"})
                x_plt, y_plt = np.meshgrid(np.arange(z_plt.shape[1]) + 1,
                                           d_values[1:])
                surf = axe.plot_surface(x_plt, y_plt, z_plt, cmap=cm.coolwarm,
                                        linewidth=0, antialiased=False)
                plt.title(titel)
                axe.set_ylabel('Treatment levels')
                axe.set_zlabel(iate_label)
                axe.set_xlabel('Index of sorted IATEs')
                fig.colorbar(surf, shrink=0.5, aspect=5)
                ttt = titel.replace(' ', '')
                file_name_jpeg = (c_dict['cs_ate_iate_fig_pfad_jpeg']
                                  + '/' + ttt + '.jpeg')
                file_name_pdf = (c_dict['cs_ate_iate_fig_pfad_pdf']
                                 + '/' + ttt + '.pdf')
                if c_dict['post_plots']:
                    gp.delete_file_if_exists(file_name_jpeg)
                    gp.delete_file_if_exists(file_name_pdf)
                    fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
                    fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
                if c_dict['show_plots']:
                    plt.show()
                else:
                    plt.close()
    # k-means clustering
    if c_dict['post_km']:
        pd.set_option('display.max_rows', 1000, 'display.max_columns', 100)
        iate_np = iate.to_numpy()
        silhouette_avg_prev = -1
        print_str = '\n' + '=' * 80 + '\nK-Means++ clustering\n' + '-' * 80
        for cluster_no in c_dict['post_km_no_of_groups']:
            cluster_lab_tmp = KMeans(
                n_clusters=cluster_no,
                n_init=c_dict['post_km_replications'], init='k-means++',
                max_iter=c_dict['post_kmeans_max_tries'], algorithm='lloyd',
                random_state=42, tol=1e-5, verbose=0, copy_x=True
                ).fit_predict(iate_np)
            silhouette_avg = silhouette_score(iate_np, cluster_lab_tmp)
            print('Number of clusters: ', cluster_no,
                  'Average silhouette score:', silhouette_avg)
            if silhouette_avg > silhouette_avg_prev:
                cluster_lab_np = np.copy(cluster_lab_tmp)
                silhouette_avg_prev = np.copy(silhouette_avg)
        print_str += ('\nBest value of average silhouette score:'
                      + f' {silhouette_avg_prev}')
        del iate_np
        # Reorder labels for better visible inspection of results
        iate_name = iate_pot_name['names_iate']
        namesfirsty = iate_name[0:round(len(iate_name)/len(v_dict['y_name']))]
        cl_means = iate[namesfirsty].groupby(by=cluster_lab_np).mean(
            numeric_only=True)
        cl_means_np = cl_means.to_numpy()
        cl_means_np = np.mean(cl_means_np, axis=1)
        sort_ind = np.argsort(cl_means_np)
        cl_group = cluster_lab_np.copy()
        for cl_j, cl_old in enumerate(sort_ind):
            cl_group[cluster_lab_np == cl_old] = cl_j
        print_str += ('\nEffects are ordered w.r.t. to the size of the effects'
                      + ' for the first outcome.')
        cl_values, cl_obs = np.unique(cl_group, return_counts=True)
        print_str += '\n' + '-' * 80 + '\nNumber of observations\n' + '-' * 80
        for idx, val in enumerate(cl_values):
            print_str += f'\nGroup {val:2}: {cl_obs[idx]:6} '
        print_str += '\n' + '-' * 80 + '\nEffects\n' + '-' * 80
        daten_neu = data.copy()
        daten_neu['IATE_Cluster'] = cl_group
        gp.delete_file_if_exists(file_name)
        print('\nSaving cluster indicator from k-means clustering.')
        daten_neu.to_csv(file_name)
        del daten_neu
        cl_means = iate.groupby(by=cl_group).mean(numeric_only=True)
        print_str += '\n' + cl_means.transpose().to_string()
        print_str += '\n' + '-' * 80 + '\nPotential outcomes\n' + '-' * 80
        cl_means = pot_y.groupby(by=cl_group).mean(numeric_only=True)
        print_str += '\n' + cl_means.transpose().to_string()
        print_str += '\n' + '-' * 80 + '\nCovariates\n' + '-' * 80
        names_unordered = [xn for xn in v_x_type.keys() if v_x_type[xn] > 0]
        if names_unordered:  # List is not empty
            x_dummies = pd.get_dummies(x_dat, columns=names_unordered)
            x_km = pd.concat([x_dat[names_unordered], x_dummies], axis=1)
        else:
            x_km = x_dat
        cl_means = x_km.groupby(by=cl_group).mean(numeric_only=True)
        print_str += '\n' + cl_means.transpose().to_string() + '\n' + '-' * 80
        print(print_str)
        gp.print_f(c_dict['outfilesummary'], print_str)
        pd.set_option('display.max_rows', None, 'display.max_columns', None)
    if c_dict['post_random_forest_vi'] and c_dict['with_output']:
        names_unordered = [xn for xn in v_x_type.keys() if v_x_type[xn] > 0]
        x_name = x_dat.columns.tolist()
        dummy_group_names = []
        if names_unordered:  # List is not empty
            dummy_names = []
            replace_dict = dict(zip(gp.primes_list(1000), list(range(1000))))
            for name in names_unordered:
                x_t_d = x_dat[name].replace(replace_dict)
                x_t_d = pd.get_dummies(x_t_d, prefix=name)
                this_dummy_names = x_t_d.columns.tolist()
                dummy_names.extend(this_dummy_names[:])
                this_dummy_names.append(name)
                dummy_group_names.append(this_dummy_names[:])
                x_dat = pd.concat([x_dat, x_t_d], axis=1)
            x_name.extend(dummy_names)
            if c_dict['with_output'] and c_dict['verbose']:
                print('The following dummy variables have been created',
                      dummy_names)
        x_train = x_dat.to_numpy(copy=True)
        if c_dict['with_output'] and c_dict['verbose']:
            print('Features used to build random forest')
            print(x_dat.describe())
            print()
        for idx, y_name in enumerate(iate_pot_name['names_iate']):
            if c_dict['d_type'] == 'continuous' and idx not in eva_points:
                continue
            print('Computing post estimation random forests for ', y_name)
            y_train = iate[y_name].to_numpy(copy=True)
            gp_est.random_forest_scikit(
                x_train, y_train, None, x_name=x_name, y_name=y_name,
                boot=c_dict['boot'], n_min=2, no_features='sqrt',
                max_depth=None, workers=c_dict['no_parallel'], alpha=0,
                var_im_groups=dummy_group_names,
                max_leaf_nodes=None, pred_p_flag=False, pred_t_flag=True,
                pred_oob_flag=True, with_output=True, variable_importance=True,
                pred_uncertainty=False, pu_ci_level=0.9, pu_skew_sym=0.5,
                var_im_with_output=True)


def eva_points_fct(no_of_names, no_of_y_name):
    """Get evaluation points."""
    no_of_names_y = round(no_of_names / no_of_y_name)
    eva_points_y = [round(no_of_names_y / 3), round(2 * no_of_names_y / 3)]
    eva_points = []
    for idx in range(no_of_y_name):
        eva_points_t = [i + no_of_names_y * idx for i in eva_points_y]
        eva_points.extend(eva_points_t)
    return eva_points
