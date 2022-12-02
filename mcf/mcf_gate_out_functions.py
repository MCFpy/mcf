"""
Output procedures needed for GATEs estimation.

Created on Wed March 23 15:48:57 2022.

@author: MLechner

# -*- coding: utf-8 -*-
"""
from itertools import chain, repeat

import numpy as np
import pandas as pd
import scipy.stats as sct
import matplotlib.pyplot as plt
from matplotlib import cm

from mcf import mcf_general_purpose as mcf_gp
from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est


def plot_marginal(pred, pred_se, names_pred, x_name, x_values_in, x_type,
                  c_dict, regrf, minus_ate=False):
    """Show the plots, similar to GATE and IATE for discrete treatments.

    Parameters
    ----------
    pred : Numpy array. Predictions.
    pred_se : Numpy array. Standard errors.
    names_pred : List of strings.
    v_name : Str.
    x_values : List of float or int. Values to evaluate the predictions.
    x_type : Type of variable.
    c_dict : Dict. Parameters.
    regrf : Boolean. Regression forest.

    Returns
    -------
    None.

    """
    conf_int = sct.norm.ppf(c_dict['fig_ci_level'] +
                            0.5 * (1 - c_dict['fig_ci_level']))
    if x_type > 0:  # categorical variables
        x_values = gp.primeposition(x_values_in, start_with_1=True)
    else:
        x_values = x_values_in
    for idx, _ in enumerate(names_pred):
        if regrf:
            titel = ('Marginal predictive plot ' + x_name + ' '
                     + names_pred[idx])
        else:
            if minus_ate:
                titel = ('MGATE - avg.(MGATE) ' + x_name + ' '
                         + names_pred[idx][:-9])
            else:
                titel = 'MGATE ' + x_name + ' ' + names_pred[idx][:-5]
        pred_temp, pred_se_temp = pred[:, idx], pred_se[:, idx]
        if x_type == 0 and len(x_values) > c_dict['no_filled_plot']:
            pred_temp = gp_est.moving_avg_mean_var(pred_temp, 3, False)[0]
            pred_se_temp = gp_est.moving_avg_mean_var(
                pred_se_temp, 3, False)[0]
        file_titel = titel.replace(" ", "")
        file_titel = file_titel.replace("-", "M")
        file_titel = file_titel.replace(".", "")
        file_name_jpeg = (c_dict['mgate_fig_pfad_jpeg'] + '/' + file_titel
                          + '.jpeg')
        file_name_pdf = (c_dict['mgate_fig_pfad_pdf'] + '/' + file_titel
                         + '.pdf')
        file_name_csv = (c_dict['mgate_fig_pfad_csv'] + '/' + file_titel
                         + 'plotdat.csv')
        upper = pred_temp + pred_se_temp * conf_int
        lower = pred_temp - pred_se_temp * conf_int
        fig, axs = plt.subplots()
        label_u = f'Upper {c_dict["fig_ci_level"]:2.0%} %-CI'
        label_l = f'Lower {c_dict["fig_ci_level"]:2.0%} %-CI'
        label_ci = str(c_dict['fig_ci_level'] * 100) + '%-CI'
        if regrf:
            label_m = 'Conditional prediction'
        else:
            if minus_ate:
                label_m = 'MGATE - avg.(MGATE)'
                label_y = 'Effect - average'
                label_0, line_0 = '_nolegend_', '_-r'
                zeros = np.zeros_like(pred_temp)
            else:
                label_m, label_y = 'MGATE', 'Effect'
        if x_type == 0 and len(x_values) > c_dict['no_filled_plot']:
            axs.plot(x_values, pred_temp, label=label_m, color='b')
            axs.fill_between(x_values, upper, lower, alpha=0.3, color='b',
                             label=label_ci)
        else:
            u_line, l_line, middle = 'v', '^', 'o'
            connect_y, connect_x = np.empty(2), np.empty(2)
            for ind, i_lab in enumerate(x_values):
                connect_y[0] = upper[ind].copy()
                connect_y[1] = lower[ind].copy()
                connect_x[0], connect_x[1] = i_lab, i_lab
                axs.plot(connect_x, connect_y, 'b-', linewidth=0.7)
            axs.plot(x_values, pred_temp, middle + 'b', label=label_m)
            axs.plot(x_values, upper, u_line + 'b', label=label_u)
            axs.plot(x_values, lower, l_line + 'b', label=label_l)
            if minus_ate:
                axs.plot(x_values, zeros, line_0, label=label_0)
        if c_dict['with_output']:
            print_str = print_mgate(pred_temp, pred_se_temp, titel, x_values)
            print_str += '\n' + gp_est.print_se_info(c_dict['cluster_std'],
                                                     c_dict['se_boot_gate'])
            if minus_ate:
                if not c_dict['gates_minus_previous']:
                    print_str += gp_est.print_minus_ate_info(c_dict['w_yes'],
                                                             print_it=False)
            print(print_str)
            gp.print_f(c_dict['outfilesummary'], print_str)
        axs.set_ylabel(label_y)
        axs.legend(loc=c_dict['fig_legend_loc'], shadow=True,
                   fontsize=c_dict['fig_fontsize'])
        titel_tmp = titel[:-4] + ' ' + titel[-4:]
        titel_tmp = titel_tmp.replace('vs', ' vs ')
        axs.set_title(titel_tmp)
        axs.set_xlabel(x_name)
        gp.delete_file_if_exists(file_name_jpeg)
        gp.delete_file_if_exists(file_name_pdf)
        fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
        fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
        if c_dict['show_plots']:
            plt.show()
        else:
            plt.close()
        upper, lower = upper.reshape(-1, 1), lower.reshape(-1, 1)
        pred_temp = pred_temp.reshape(-1, 1)
        x_values_np = np.array(x_values, copy=True).reshape(-1, 1)
        effects_et_al = np.concatenate(
            (upper, pred_temp, lower, x_values_np), axis=1)
        cols = ['upper', 'effects', 'lower', 'x_values']
        datasave = pd.DataFrame(data=effects_et_al, columns=cols)
        gp.delete_file_if_exists(file_name_csv)
        datasave.to_csv(file_name_csv, index=False)


def print_mgate(est, stderr, titel, z_values):
    """
    Print various GATEs (MGATE, AMGATE, ...).

    Parameters
    ----------
    est : 1-dim Numpy array. Effects.
    stderr : 1-dim Numpy array. Standard errors.
    titel : Str. Titel of Table
    z_values : List of floats/int. Values of z.

    Returns
    -------
    print_str : String. Printable output.

    """
    t_val = np.abs(est / stderr)
    p_val = sct.t.sf(np.abs(t_val), 1000000) * 2
    if isinstance(z_values[0], (float, np.float32, np.float64)):
        z_values_p, z_is_float = np.around(z_values, 2), True
    else:
        z_values_p, z_is_float = z_values, False
    print_str = '- ' * 40 + f'\n{titel}'
    print_str += '\nValue of Z       Est        SE    t-val   p-val'
    print_str += '\n' + '- ' * 40 + '\n'
    for z_ind, z_val in enumerate(z_values_p):
        if z_is_float:
            print_str += f'{z_val:>6.2f}         '
        else:
            print_str += f'{z_val:>6.0f}         '
        print_str += f'{est[z_ind]:>8.5f}  {stderr[z_ind]:>8.5f} '
        print_str += f'{t_val[z_ind]:>5.2f}  {p_val[z_ind]*100:>5.2f}% '
        print_str += print_stars(p_val[z_ind]) + '\n'
    print_str += '-' * 80 + '\n'
    print_str += 'Values of Z may have been recoded into primes.\n' + '-' * 80
    return print_str


def print_stars(p_val):
    """Print stars."""
    if p_val < 0.001:
        return '****'
    if p_val < 0.01:
        return ' ***'
    if p_val < 0.05:
        return '  **'
    if p_val < 0.1:
        return '   *'
    return '    '


def print_mgate_cont(est, stderr, titel, z_values, d_values):
    """
    Print various GATEs (MGATE, AMGATE, ...).

    Parameters
    ----------
    est : 1-dim Numpy array. Effects.
    stderr : 1-dim Numpy array. Standard errors.
    titel : Str. Titel of Table
    z_values : List of floats/int. Values of z.

    Returns
    -------
    None.

    """
    t_val = np.abs(est / stderr)
    p_val = sct.t.sf(np.abs(t_val), 1000000) * 2
    if isinstance(z_values[0], (float, np.float32, np.float64)):
        z_values_p, z_is_float = np.around(z_values, 2), True
    else:
        z_is_float, z_values_p = False, z_values
    print()
    print('- ' * 40)
    print(titel)
    print('Value of Z         Value of D            Est        SE     t-val',
          ' p-val')
    print('- ' * 40)
    for z_idx, z_val in enumerate(z_values_p):
        for d_idx, d_val in enumerate(d_values):
            if z_is_float:
                print(f'{z_val:>10.4f}        ', end=' ')
            else:
                print(f'{z_val:>10.0f}        ', end=' ')
            print(f'{d_val:>10.4f}        ', end=' ')
            print(f'{est[z_idx, d_idx]:>8.5f}  {stderr[z_idx, d_idx]:>8.5f}',
                  end=' ')
            print(f'{t_val[z_idx, d_idx]:>6.2f}',
                  f'{p_val[z_idx, d_idx]*100:>6.2f}%', end=' ')
            print_stars(p_val[z_idx, d_idx])
        print('- ' * 40)
    print('-' * 80)
    print('Values of Z may have been recoded into primes.')
    print('-' * 80)


def make_gate_figures_discr(
        titel, z_name, z_vals, z_type, effects, stderr, c_dict, ate=0,
        ate_se=None, am_gate=False, z_smooth=False, gatet_yes=False):
    """Generate the figures for GATE results (discrete outcomes).

    Parameters
    ----------
    titel : String. (Messy) title of plot and basis for files.
    z_values : List. Values of z-variables.
    z_type : Int. Type of variable (ordered or unordered)
    effects : 1D Numpy array. Effects for all z-values.
    stderr : 1D Numpy array. Standard errors for all effects.
    c : Dict. Parameters.

    Returns
    -------
    None.

    """
    z_values = z_vals.copy()
    if mcf_gp.find_precision(z_values) == 0:
        z_values = gp.recode_if_all_prime(z_values)
    titel_f = titel.replace(' ', '')
    titel_f = titel_f.replace('-', 'M')
    titel_f = titel_f.replace('.', '')
    if am_gate:
        file_name_jpeg = (c_dict['amgate_fig_pfad_jpeg'] + '/' + titel_f
                          + '.jpeg')
        file_name_pdf = c_dict['amgate_fig_pfad_pdf'] + '/' + titel_f + '.pdf'
        file_name_csv = (c_dict['amgate_fig_pfad_csv'] + '/' + titel_f
                         + 'plotdat.csv')
    else:
        file_name_jpeg = c_dict['gate_fig_pfad_jpeg'] + '/' + titel_f + '.jpeg'
        file_name_pdf = c_dict['gate_fig_pfad_pdf'] + '/' + titel_f + '.pdf'
        file_name_csv = (c_dict['gate_fig_pfad_csv'] + '/' + titel_f
                         + 'plotdat.csv')
    if ate_se is None:
        gate_str = 'AMGATE-avg.(AMGATE)' if am_gate else 'GATE-ATE'
        gate_str_y = 'Effect - average'
        if am_gate:
            label_m = 'AMGATE-avg.(AMATE)'
        else:
            label_m = 'GATET-ATET' if gatet_yes else 'GATE-ATE'
        label_y = 'Effect - average'
        ate_label = '_nolegend_'
    else:
        if am_gate:
            label_m = 'AMGATE'
            ate_label = 'avg.(AMGATE)'
        else:
            label_m = 'GATET' if gatet_yes else 'GATE'
            ate_label = 'ATE'
        label_y = 'Effect'
        gate_str = 'AMGATE' if am_gate else 'GATE'
        gate_str_y = 'Effect'
    ate = ate * np.ones((len(z_values), 1))
    if isinstance(z_type, (list, tuple, np.ndarray)):
        z_type = z_type[0]
    cint = sct.norm.ppf(
        c_dict['fig_ci_level'] + 0.5 * (1 - c_dict['fig_ci_level']))
    upper, lower = effects + stderr * cint, effects - stderr * cint
    if ate_se is not None:
        ate_upper, ate_lower = ate + ate_se * cint, ate - ate_se * cint
    label_ci = f'{c_dict["fig_ci_level"]:2.0%}-CI'
    if (z_type == 0) and (len(z_values) > c_dict['no_filled_plot']):
        if am_gate or z_smooth:
            if am_gate:
                file_name_f_jpeg = (c_dict['amgate_fig_pfad_jpeg']
                                    + '/' + titel_f + 'fill.jpeg')
                file_name_f_pdf = (c_dict['amgate_fig_pfad_pdf']
                                   + '/' + titel_f + 'fill.pdf')
            else:
                file_name_f_jpeg = (c_dict['gate_fig_pfad_jpeg']
                                    + '/' + titel_f + 'fill.jpeg')
                file_name_f_pdf = (c_dict['gate_fig_pfad_pdf']
                                   + '/' + titel_f + 'fill.pdf')
            figs, axs = plt.subplots()
            axs.plot(z_values, effects, label=label_m, color='b')
            axs.fill_between(z_values, upper, lower, alpha=0.3, color='b',
                             label=label_ci)
            line_ate = '-r'
            if ate_se is not None:
                axs.fill_between(
                    z_values, ate_upper.reshape(-1), ate_lower.reshape(-1),
                    alpha=0.3, color='r', label=label_ci)
                label_ate = 'ATE'
            else:
                label_ate = '_nolegend_'            
            axs.plot(z_values, ate, line_ate, label=label_ate)
            axs.set_ylabel(label_y)
            axs.legend(loc=c_dict['fig_legend_loc'], shadow=True,
                       fontsize=c_dict['fig_fontsize'])
            titel_tmp = titel[:-4] + ' ' + titel[-4:]
            titel_tmp = titel_tmp.replace('vs', ' vs ')
            axs.set_title(titel_tmp)
            axs.set_xlabel(z_name)
            gp.delete_file_if_exists(file_name_f_jpeg)
            gp.delete_file_if_exists(file_name_f_pdf)
            figs.savefig(file_name_f_jpeg, dpi=c_dict['fig_dpi'])
            figs.savefig(file_name_f_pdf, dpi=c_dict['fig_dpi'])
            if c_dict['show_plots']:
                plt.show()
            else:
                plt.close()
        e_line, u_line, l_line = '_-', 'v-', '^-'
    else:
        e_line, u_line, l_line = 'o', 'v', '^'
    connect_y, connect_x = np.empty(2), np.empty(2)
    fig, axe = plt.subplots()
    for idx, i_lab in enumerate(z_values):
        connect_y[0], connect_y[1] = upper[idx], lower[idx]
        connect_x[0], connect_x[1] = i_lab, i_lab
        axe.plot(connect_x, connect_y, 'b-', linewidth=0.7)
    axe.plot(z_values, effects, e_line + 'b', label=gate_str)
    axe.set_ylabel(gate_str_y)
    label_u = f'Upper {c_dict["fig_ci_level"]:2.0%}-CI'
    label_l = f'Lower {c_dict["fig_ci_level"]:2.0%}-CI'
    axe.plot(z_values, upper, u_line + 'b', label=label_u)
    axe.plot(z_values, lower, l_line + 'b', label=label_l)
    axe.plot(z_values, ate, '-' + 'r', label=ate_label)
    if ate_se is not None:
        axe.plot(z_values, ate_upper, '--' + 'r', label=label_u)
        axe.plot(z_values, ate_lower, '--' + 'r', label=label_l)
    axe.legend(loc=c_dict['fig_legend_loc'], shadow=True,
               fontsize=c_dict['fig_fontsize'])
    titel_tmp = titel[:-4] + ' ' + titel[-4:]
    titel_tmp = titel_tmp.replace('vs', ' vs ')
    axe.set_title(titel_tmp)
    axe.set_xlabel(z_name)
    gp.delete_file_if_exists(file_name_jpeg)
    gp.delete_file_if_exists(file_name_pdf)
    fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
    fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
    if c_dict['show_plots']:
        plt.show()
    else:
        plt.close()
    effects = effects.reshape(-1, 1)
    upper, lower = upper.reshape(-1, 1), lower.reshape(-1, 1)
    z_values_np = np.array(z_values, copy=True).reshape(-1, 1)
    if ate_se is not None:
        ate_upper = ate_upper.reshape(-1, 1)
        ate_lower = ate_lower.reshape(-1, 1)
        effects_et_al = np.concatenate(
            (upper, effects, lower, ate, ate_upper, ate_lower, z_values_np),
            axis=1)
        cols = ['upper', 'effects', 'lower', 'ate', 'ate_upper', 'ate_lower',
                'z_values']
    else:
        cols = ['upper', 'effects', 'lower', 'ate', 'z_values']
        effects_et_al = np.concatenate(
            (upper, effects, lower, ate, z_values_np), axis=1)
    datasave = pd.DataFrame(data=effects_et_al, columns=cols)
    gp.delete_file_if_exists(file_name_csv)
    datasave.to_csv(file_name_csv, index=False)


def make_gate_figures_cont(titel, z_name, z_values, effects, c_dict,
                           ate=None, am_gate=False, d_values=None):
    """Generate the figures for GATE results.

    Parameters
    ----------
    titel : String. (Messy) title of plot and basis for files.
    z_values : List. Values of z-variables.
    effects : 1D Numpy array. Effects for all z-values.
    c : Dict. Parameters.

    Returns
    -------
    None.

    """
    titel = 'Dose response ' + titel
    titel_f = titel.replace(' ', '')
    titel_f = titel_f.replace('-', 'M')
    titel_f = titel_f.replace('.', '')
    if am_gate:
        file_name_jpeg = (c_dict['amgate_fig_pfad_jpeg'] + '/' + titel_f
                          + '.jpeg')
        file_name_pdf = c_dict['amgate_fig_pfad_pdf'] + '/' + titel_f + '.pdf'
    else:
        file_name_jpeg = c_dict['gate_fig_pfad_jpeg'] + '/' + titel_f + '.jpeg'
        file_name_pdf = c_dict['gate_fig_pfad_pdf'] + '/' + titel_f + '.pdf'
    if ate is not None:
        gate_str = 'AMGATE-A(AMATE)' if am_gate else 'GATE-ATE'
    else:
        gate_str = 'AMGATE' if am_gate else 'GATE'
    fig, axe = plt.subplots(subplot_kw={"projection": "3d"})
    z_plt = np.transpose(effects)
    x_plt, y_plt = np.meshgrid(z_values, d_values[1:])
    surf = axe.plot_surface(x_plt, y_plt, z_plt, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    plt.title(titel)
    axe.set_ylabel('Treatment levels')
    axe.set_zlabel(gate_str)
    axe.set_xlabel(z_name)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    gp.delete_file_if_exists(file_name_jpeg)
    gp.delete_file_if_exists(file_name_pdf)
    fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
    fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
    if c_dict['show_plots']:
        plt.show()
    else:
        plt.close()


def plot_marginal_cont(pred, pred_se, x_name, x_values_in, x_type,
                       c_dict, minus_ate=False):
    """Show the plots, similar to GATE and IATE for continuous treatments.

    Parameters
    ----------
    pred : Numpy array. Predictions.
    names_pred : List of strings.
    v_name : Str.
    x_values : List of float or int. Values to evaluate the predictions.
    x_type : Type of variable.
    c_dict : Dict. Parameters.

    Returns
    -------
    None.

    """
    if x_type > 0:  # categorical variables
        x_values = gp.primeposition(x_values_in, start_with_1=True)
    else:
        x_values = x_values_in
    d_values = c_dict['ct_d_values_dr_np'][1:]
    titel = 'MGATE - MATE ' + x_name if minus_ate else 'MGATE ' + x_name
    titel = 'Dose response ' + titel
    file_titel = titel.replace(" ", "")
    file_titel = titel.replace("-", "M")
    file_name_jpeg = c_dict['mgate_fig_pfad_jpeg'] + '/' + file_titel + '.jpeg'
    file_name_pdf = c_dict['mgate_fig_pfad_pdf'] + '/' + file_titel + '.pdf'
    fig, axe = plt.subplots(subplot_kw={"projection": "3d"})
    mgate_str = 'MGATE-A(AMATE)' if minus_ate else 'MGATE'
    z_plt = np.transpose(pred)
    x_plt, y_plt = np.meshgrid(x_values, d_values)
    surf = axe.plot_surface(x_plt, y_plt, z_plt, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    plt.title(titel)
    axe.set_ylabel('Treatment levels')
    axe.set_zlabel(mgate_str)
    axe.set_xlabel(x_name)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    gp.delete_file_if_exists(file_name_jpeg)
    gp.delete_file_if_exists(file_name_pdf)
    fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
    fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
    if c_dict['show_plots']:
        plt.show()
    else:
        plt.close()
    print_mgate_cont(pred, pred_se, titel, x_values, d_values)
    gp_est.print_se_info(c_dict['cluster_std'], c_dict['se_boot_gate'])
    gp_est.print_minus_ate_info(c_dict['w_yes'])


def wald_test(z_name, no_of_zval, w_gate, y_dat, w_dat, cl_dat, a_idx, o_idx,
              w_ate, c_dict, gate_str='GATE', no_of_treat=None, d_values=None,
              print_output=True):
    """Compute Wald tests for GATE(T).

    Parameters
    ----------
    z_name : String.
    no_of_zval : Int.
    w_gate : Numpy array. Weights.
    y : Numpy array. Outcomes.
    w : Numpy array. Sampling weights (if needed).
    cl : Numpy array. Cluster indicator.
    a_idx : Int. Reference population indicator.
    o_idx : Int. Outcome indicator.
    w_ate : Numpy array. Weights.
    c : Dict. Parameters.
    gate_str : Str. Str for label of test. Default is 'GATE'.
    print_output : Bool. Print output in this function. Default is True.

    Returns
    -------
    print_str : String. String with printable output.

    """
    print_str = f'\nWald tests:  {gate_str} for {z_name}'
    print_str += '\nComparison     Chi2 stat   df   p-value (%)'
    w_gate_sum, w_ate_sum = np.sum(w_gate, axis=3), np.sum(w_ate, axis=2)
    for t1_idx in range(no_of_treat):
        if not (1-1e-10 < w_ate_sum[a_idx, t1_idx] < 1+1e-10):
            w_ate[a_idx, t1_idx, :] = w_ate[
                a_idx, t1_idx, :] / w_ate_sum[a_idx, t1_idx]
        for zj1 in range(no_of_zval):
            if not ((-1e-15 < w_gate_sum[zj1, a_idx, t1_idx] < 1e-15)
                    or (1-1e-10 < w_gate_sum[zj1, a_idx, t1_idx] < 1+1e-10)):
                w_gate[zj1, a_idx, t1_idx, :] = w_gate[
                    zj1, a_idx, t1_idx, :] / w_gate_sum[zj1, a_idx, t1_idx]
    not_computed = False
    for t1_idx in range(no_of_treat):
        for t2_idx in range(t1_idx+1, no_of_treat):
            diff_w = np.empty(no_of_zval-1)
            var_w = np.empty((no_of_zval-1, no_of_zval-1))
            for zj1 in range(no_of_zval-1):
                ret1 = gp_est.weight_var(
                    w_gate[zj1, a_idx, t1_idx, :] - w_ate[a_idx, t1_idx, :],
                    y_dat[:, o_idx], cl_dat, c_dict, False, weights=w_dat,
                    bootstrap=c_dict['se_boot_gate'])
                ret2 = gp_est.weight_var(
                    w_gate[zj1, a_idx, t2_idx, :] - w_ate[a_idx, t2_idx, :],
                    y_dat[:, o_idx], cl_dat, c_dict, False, weights=w_dat,
                    bootstrap=c_dict['se_boot_gate'])
                diff_w[zj1] = ret2[0] - ret1[0]
                var_w[zj1, zj1] = ret1[1] + ret2[1]
                if no_of_zval > 2:
                    for zj2 in range(zj1+1, no_of_zval-1):
                        if c_dict['cluster_std']:
                            ret1 = gp_est.aggregate_cluster_pos_w(
                                cl_dat, w_gate[zj1, a_idx, t1_idx, :]
                                - w_ate[a_idx, t1_idx, :], y_dat[:, o_idx],
                                False, w2_dat=w_gate[zj2, a_idx, t1_idx, :]
                                - w_ate[a_idx, t1_idx, :], sweights=w_dat,
                                y2_compute=True)
                            ret2 = gp_est.aggregate_cluster_pos_w(
                                cl_dat, w_gate[zj1, a_idx, t2_idx, :]
                                - w_ate[a_idx, t2_idx, :], y_dat[:, o_idx],
                                False, w2_dat=w_gate[zj2, a_idx, t2_idx, :]
                                - w_ate[a_idx, t2_idx, :], sweights=w_dat,
                                y2_compute=True)
                            w11, w12 = ret1[0], ret1[2]
                            w21, w22 = ret2[0], ret2[2]
                            y_t11, y_t12 = ret1[1], ret1[3]
                            y_t21, y_t22 = ret2[1], ret2[3]
                        else:
                            w11 = (w_gate[zj1, a_idx, t1_idx, :]
                                   - w_ate[a_idx, t1_idx, :])
                            w12 = (w_gate[zj2, a_idx, t1_idx, :]
                                   - w_ate[a_idx, t1_idx, :])
                            w21 = (w_gate[zj1, a_idx, t2_idx, :]
                                   - w_ate[a_idx, t2_idx, :])
                            w22 = (w_gate[zj2, a_idx, t2_idx, :]
                                   - w_ate[a_idx, t2_idx, :])
                            y_t11 = y_dat[:, o_idx]
                            y_t12, y_t21, y_t22 = y_t11, y_t11, y_t11
                        cv1 = (np.mean(w11 * w12 * (y_t11 * y_t12)) -
                               (np.mean(w11 * y_t11) * np.mean(w12 * y_t12)))
                        cv2 = (np.mean(w21 * w22 * y_t21 * y_t22) -
                               (np.mean(w21 * y_t21) * np.mean(w22 * y_t22)))
                        cv12 = len(w11) * cv1 + len(w21) * cv2
                        var_w[zj1, zj2] = cv12
                        var_w[zj2, zj1] = var_w[zj1, zj2]
            stat, dfr, pval = gp.waldtest(diff_w, var_w)
            print_str += (f'\n{d_values[t2_idx]:<3} vs. '
                          + f' {d_values[t1_idx]:<3}:'
                          + f' {stat:>8.3f}  {dfr:>4}   {pval*100:>7.3f}%')
            if stat < 0:
                not_computed = True
    if not_computed:
        print_str += '\n' + '- ' * 40 + '\n'
        print_str += ('Negative values imply that statistic has not been'
                      + ' computed because covariance matrix of test is not'
                      + ' of full rank.')
        print_str += '\n' + '- ' * 40
    if print_output:
        print(print_str)
    return print_str


def ref_file_marg_plot_amgate(in_csv_file, z_name, c_dict, eva_values):
    """
    Create reference samples for covariates (AMGATE).

    Revised procedure by Hugo, 5.1.2021. Should be faster using less memory.

    Parameters
    ----------
    in_csv_file : String. File for which predictions will be obtained
    z_name : String. Name of variable that is varying.
    c_dict : Dict. Parameters.
    eva_values : List. Values at which z is evaluated.

    Returns
    -------
    out_csv_file : String. csv-file with reference values.
    z_values : List of float/Int. List of values

    """
    eva_values = eva_values[z_name]
    data_df = pd.read_csv(in_csv_file)
    no_eval = len(eva_values)
    obs = len(data_df)
    if obs/no_eval > 10:  # Save computation time by using random samples
        share = c_dict['gmate_sample_share'] / no_eval
        if 0 < share < 1:
            seed = 9324561
            rng = np.random.default_rng(seed)
            idx = rng.choice(obs, int(np.floor(obs * share)), replace=False)
            obs = len(idx)
            if c_dict['with_output'] and c_dict['verbose']:
                print(f'{share * 100:5.2f}% random sample drawn')
        else:
            idx = np.arange(obs)
    else:
        idx = np.arange(obs)
    new_idx_dataframe = list(chain.from_iterable(repeat(idx, no_eval)))
    data_all_df = data_df.loc[new_idx_dataframe, :]
    new_values_z = list(chain.from_iterable(
        [[i] * obs for i in eva_values]))
    data_all_df.loc[:, z_name] = new_values_z
    out_csv_file = c_dict['temporary_file']
    if c_dict['with_output'] and c_dict['verbose']:
        print()
        print('AMGATEs minus ATE are evaluated at fixed z-feature values',
              '(equally weighted).')
    gp.delete_file_if_exists(out_csv_file)
    data_all_df.to_csv(out_csv_file, index=False)
    return out_csv_file, eva_values


def ref_file_marg_plot(plotx_name, c_dict, var_x_type, reference_values,
                       eva_values):
    """
    Create reference values for covariates (MAGE).

    Parameters
    ----------
    plotx_name : String. Name of variable that is varying.
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.
    var_x_type : Dict. Name and type of variable.

    Returns
    -------
    out_csv_file : String. csv-file with reference values.

    """
    names = []
    eva_values = eva_values[plotx_name]
    no_rows = len(eva_values)
    x_vals = np.empty((no_rows, len(var_x_type.keys())))
    for c_idx, vname in enumerate(var_x_type.keys()):
        names.append(vname)
        for r_idx in range(no_rows):
            if vname == plotx_name:
                x_vals[r_idx, c_idx] = eva_values[r_idx]
            else:
                x_vals[r_idx, c_idx] = reference_values[vname]
    out_csv_file = c_dict['temporary_file']
    datasave = pd.DataFrame(data=x_vals, columns=names)
    datasave.to_csv(out_csv_file, index=False)
    return out_csv_file


def ref_vals_margplot(in_csv_file, var_x_type, var_x_values,
                      with_output=False, ref_values_needed=True,
                      no_eva_values=50):
    """Compute reference values for marginal plots.

    Parameters
    ----------
    in_csv_file : String. Data.
    var_x_type : Dict. Variable names and types (0,1,2)
    var_x_values: Dict. Variable names and values.
    with_output : Bool. Print output. Default is False.
    ref_values_needed : Bool. Compute reference values. Default is True.
    no_eva_values : Bool. Number of evaluation points (cont. var).
                          Default is 50.

    Returns
    -------
    reference_values : Dict. Variable names and reference values.
    evaluation_values : Dict. Variable names and evaluation values.

    """
    if with_output and ref_values_needed:
        print('Effects are evaluated at these values for other features')
        print("Variable                  Type     Reference value")
    reference_values = {}
    evaluation_values = {}
    data_df = pd.read_csv(in_csv_file)
    obs = len(data_df)
    for vname in var_x_type.keys():
        ddf = data_df[vname]
        if var_x_type[vname] == 0:
            if ref_values_needed:
                ref_val = ddf.median()
            type_str = '  ordered'
        else:
            type_str = 'unordered'
            if ref_values_needed:
                ref_val = ddf.mode()
                ref_val = ref_val.to_numpy()
                if np.size(ref_val) > 1:
                    ref_val = ref_val[0]
                ref_val = int(ref_val)
        if not var_x_values[vname]:
            no_eva_values = min(no_eva_values, obs)
            quas = np.linspace(0.01, 0.99, no_eva_values)
            eva_val = ddf.quantile(quas)
            eva_val = np.unique(eva_val).tolist()
        else:
            eva_val = var_x_values[vname].copy()
        if ref_values_needed:
            reference_values.update({vname: ref_val})
        evaluation_values.update({vname: eva_val})
        if with_output and ref_values_needed:
            print(f'{vname:20}: ', type_str, end=' ')
            if isinstance(ref_val, float):
                print(f' {ref_val:8.4f}')
            else:
                print(f' {float(ref_val):8.0f}')
    return reference_values, evaluation_values
