"""Created on Wed May  1 16:35:19 2024.

Functions for correcting scores w.r.t. protected variables.

@author: MLechner

# -*- coding: utf-8 -*-
"""
import numpy as np
import pandas as pd

from mcf import mcf_print_stats_functions as mcf_ps
from mcf.mcf_general import remove_duplicates


def adjust_scores_for_estimation_risk(optp_, data_df):
    """Remove effect of protected variables from policy score."""
    gen_dic, estrisk_dic = optp_.gen_dict, optp_.estrisk_dict

    if gen_dic['with_output']:
        txt_report = ('\nAdjusting for estimation uncertainty by adjusting '
                      'the policy score as \nAdjusted Score = Unadjusted Score '
                      f'- {estrisk_dic["value"]} x Standard Error of '
                      'Unadjusted Score'
                      )
        mcf_ps.print_mcf(gen_dic, txt_report, summary=True)
    else:
        txt_report = ''

    # Change data to numpy arrays
    scores_np = data_df[optp_.var_dict['polscore_name']].to_numpy()
    scores_se_np = data_df[optp_.var_dict['polscore_se_name']].to_numpy()

    # Adjusted scores
    scores_estrisk_np = scores_np - estrisk_dic['value'] * scores_se_np

    # Convert numpy array of fair scores to pandas dataframe
    score_estrisk_name = [name + '_estrisk'
                          for name in optp_.var_dict['polscore_name']]
    score_estrisk_df = pd.DataFrame(scores_estrisk_np,
                                    columns=score_estrisk_name)
    data_df = data_df.reset_index(drop=True)
    data_estrisk_df = pd.concat((data_df, score_estrisk_df), axis=1)

    # Descriptive statistics
    if gen_dic['with_output']:
        txt_report += estrisk_stats(optp_, data_estrisk_df, score_estrisk_name)

    # Change the names of variables (in particular scores) to be used for
    # policy learning.
    change_variable_names_estrisk(optp_, score_estrisk_name)
    optp_.report['estrisk_scores_stats'] = txt_report

    return data_estrisk_df, score_estrisk_name


def estrisk_stats(optp_, data_df, score_estrisk_name):
    """Compute descriptive statistics of fairness adjustments."""
    score_names = optp_.var_dict['polscore_name']
    txt = '\n\nDescriptive statistics of adjusted and unadjusted scores'
    all_scores = [*score_names, *score_estrisk_name]
    stats = np.round(data_df[all_scores].describe().T, 3)
    corr = np.round(data_df[all_scores].corr() * 100)
    corr = corr.dropna(how='all', axis=0)
    corr = corr.dropna(how='all', axis=1)
    txt += '\n' + str(stats)
    txt += '\n\nCorrelation of adjusted and unadjusted scores in %'
    txt += '\n' + str(corr)

    mcf_ps.print_mcf(optp_.gen_dict, txt, summary=True)

    return txt


def change_variable_names_estrisk(optp_, score_estrisk_name):
    """Change variable names to account for risk adjusted scores."""
    if optp_.var_dict['polscore_desc_name'] is not None:
        optp_.var_dict['polscore_desc_name'].extend(
            optp_.var_dict['polscore_name'].copy())
    else:
        optp_.var_dict['polscore_desc_name'] = optp_.var_dict[
            'polscore_name'].copy()
    # Remove duplicates from list without changing order
    optp_.var_dict['polscore_desc_name'] = remove_duplicates(
        optp_.var_dict['polscore_desc_name'])

    optp_.var_dict['polscore_name'] = score_estrisk_name
