"""Created on Fri Dec  8 13:42:19 2023.

Contains classes and methods needed for mcf and optimal policy reporting.

# -*- coding: utf-8 -*-

@author: MLechner
"""
import numpy as np
import pandas as pd


def opt_fairness(opt_o, empty_lines_end):
    """Text content for the fairness (protected variables) adjustment."""
    prot_vars = [*opt_o.var_dict['protected_ord_name'],
                 *opt_o.var_dict['protected_unord_name']]
    mat_vars = [*opt_o.var_dict['material_ord_name'],
                *opt_o.var_dict['material_unord_name']]
    txt = (
        '\nBACKGROUND'
        '\nThis fairness module is experimental. It is a preview of what '
        'will be analysed in the paper by Bearth, Lechner, Mareckova, and '
        'Muny (2024): Explainable Optimal Policy with Protected Variables. '
        'The main idea is to adjust the policy scores in a way such that '
        'the resulting optimal allocation will not depend on the protected '
        'features. '
        '\nCurrently the following methods are available for the '
        'adjustments: '
        '\n(1)  Mean: Mean dependence of the policy score with protected '
        'features is removed by residualisation.'
        '\n(2)  MeanVar: In addition to the mean dependence, '
        'heteroscedasticity of the policy scores related to the '
        'protected features is removed by rescaling.'
        '\n(3) Quantiled: insprired by the quantile based '
        'approach suggested by Strack & Yang (2024), but extended to address '
        'multiple '
        'treatments and possibly continuous protected and / or materially '
        'important features.'
        'If there are many values of the materially relevant '
        'features, features may be either discretized or treated as '
        'continuous. If treated as continuous, the respective densities '
        'needed to obtain the quantile '
        'positions are estimated by kernel density estimation (scikit-learn) '
        'within cells defined by the values of the protected features, if '
        'protected features are discrete or discretized. If the protected '
        'features are treated as continuous, respective conditional '
        'densities are estimated. Since '
        'the quality of the nonparametric estimators of these multivariate '
        'densities is rapidly declining with the dimension of the '
        'materially relevant variables, their dimension should be kept '
        'as small as possible (or they should be discretized).'
        '\n\nReference'
        '\n-Strack, Philipp, and Kai Hao Yang (2024): Privacy Preserving '
        'Signals, arXiv.'
           )

    txt += ('\n\nIMPLEMENTATION '
            f'\nProtected features: {", ".join(prot_vars)}. ')
    txt += '\nProtected features are '
    if opt_o.fair_dict['protected_disc_method'] == 'NoDiscretization':
        txt += ('are treated as continous, i.e., their joint densities are '
                'estimated. ')
    else:
        txt += 'discretized using '
        if opt_o.fair_dict['protected_disc_method'] == 'EqualCell':
            txt += ('equal cell sizes for each discrete feature of not more '
                    f'than {opt_o.fair_dict["protected_disc_method"]} '
                    'observations.'
                    )
        else:
            txt += ('k-means++ clustering with '
                    f'{opt_o.fair_dict["protected_disc_method"]} clusters. '
                    )
    if mat_vars:
        txt += (
            f'\nMaterially relevant features: {", ".join(mat_vars)}. '
            '\nMaterially relevant features '
            'may be correlated with the protected features. Therefore, '
            'the adjusted scores may still show some correlation with the '
            'protected features due to their correlation with the materially '
            'relevant features. '
            )
        if opt_o.fair_dict['material_disc_method'] == 'NoDiscretization':
            txt += ('are treated as continous, i.e., their joint densities are '
                    'estimated. ')
        else:
            txt += 'discretized using '
            if opt_o.fair_dict['material_disc_method'] == 'EqualCell':
                txt += ('equal cell sizes for each discrete feature of not '
                        f'more than {opt_o.fair_dict["material_disc_method"]} '
                        'observations.'
                        )
            else:
                txt += ('k-means++ clustering with '
                        f'{opt_o.fair_dict["material_disc_method"]} clusters. '
                        )
    txt += '\n' + opt_o.report['fairscores_delete_x_vars_txt']
    if 'fairscores_build_stats' in opt_o.report:
        txt += '\n' + opt_o.report['fairscores_build_stats']
    if 'fair_decision_vars_build_stats' in opt_o.report:
        txt += '\n' + opt_o.report['fair_decision_vars_build_stats']
    txt += ('\n\nTo see whether the resulting decision rules are still '
            'dependent on the protected variable, check the variable '
            'importance statistics in the output files as well.')

    return txt + '\n' * empty_lines_end


def opt_estimationrisk(opt_o, empty_lines_end):
    """Text content for the estimation risk adjustment."""
    txt = (
        '\nBACKGROUND'
        '\nThis module adjusting the policy score is experimental. '
        'Generally, the idea implemented follows the paper by '
        'Chernozhukov, Lee, Rosen, and Sun (2025). However, since several '
        'approximations are used in the algorithm, the methods will not have '
        'the direct confidence-level-related interpretations suggested by '
        'these authors. \nThe scores are adjusted by substracting a multiple '
        'of their standard errors. '
        '\n\nReference'
        '\n-Chernozhukov, Lee, Rosen, and Sun (2025), Policy Learning With '
        'Confidence, arXiv.'
           )

    txt += ('\n\nIMPLEMENTATION '
            '\nStandard errors of policy scores: '
            f'{", ".join(opt_o.var_dict["polscore_se_name"])}. '
            '\nMultiplier used for adjustment: '
            f'{opt_o.estrisk_dict["value"]}. '
            )
    return txt + '\n' * empty_lines_end


def opt_allocation(opt_o, empty_lines_end):
    """Text allocation part of description."""
    txt = ''
    for idx, alloc in enumerate(opt_o.report['alloc_list']):
        txt += alloc
        if idx < len(opt_o.report['alloc_list'])-1:
            txt += '\n' * 3 + 'NEXT ALLOCATION\n'
    return txt + '\n' * empty_lines_end


def opt_evaluation(opt_o):
    """Text evaluation part of description."""
    def results_dic_to_df(results_dic):
        treat_values = ['Share of ' + str(s) + ' in %'
                        for s in opt_o.gen_dict['d_values']]
        columns = ['Value function', *treat_values]
        for idx, key in enumerate(results_dic):
            index = next(iter(results_dic[key]))   # first key
            if isinstance(results_dic[key][index], (list, tuple)) and (
                    len(results_dic[key][index]) == 1):
                data = [round(results_dic[key][index][0], 4),
                        *list(np.round(results_dic[key]['treatment share']*100,
                                       2))]
                tmp_df = pd.DataFrame([data], index=(index,), columns=columns,
                                      copy=True)

            elif isinstance(results_dic[key][index], float) and key == index:
                data = [round(results_dic[key][index], 4),
                        *list(np.round(results_dic[key]['treatment share']*100,
                                       2))]
                tmp_df = pd.DataFrame([data], index=(index,), columns=columns,
                                      copy=True)

            else:
                data = [*list(np.round(results_dic[key]['treatment share']*100,
                                       2))]
                tmp_df = pd.DataFrame([data], index=(key,),
                                      columns=columns[1:], copy=True)
            if idx == 0:
                table_df = tmp_df
            else:
                table_df = pd.concat((table_df.copy(), tmp_df), axis=0)
        return table_df

    txt_table_list = []
    general_txt = ('\nMain evaluation results.'
                   + '\n' * 2 + 'Note: The output files contain relevant '
                   'additional information, like a descriptive analysis of '
                   'the treatment groups'
                   )
    if (opt_o.gen_dict['variable_importance']
            and (opt_o.var_dict['vi_x_name']
                 or opt_o.var_dict['vi_to_dummy_name'])):
        general_txt += ' and variable importance statistics'
    general_txt += (
        '. Also Qini-like plots are produced and saved in the same location as '
        'the text output. Those plots compare the optimal allocation to a '
        'reference allocation (3 allocations are used as such reference '
        'allocations, if available: (i) observed, (ii) random, (iii) the '
        'treatment with the highest ATE is allocated to everybody). '
        'They show the mean welfare when an increasing share of observations '
        '(starting with those who gain most from the optimal allocation '
        'compared to the reference allocation) is allocated using the optimal '
        'allocation rule.'
        )
    for evalu in opt_o.report['evalu_list']:
        txt = evalu[0]
        table_df = results_dic_to_df(evalu[1])
        txt_table_list.append((txt, table_df))

    return general_txt, txt_table_list


def opt_training(opt_o, empty_lines_end=2):
    """Text training part of description."""
    txt = '\nCOMPUTATION'
    txt += (f'\n{opt_o.gen_dict["mp_parallel"]} logical cores are used for '
            'processing.')
    if (opt_o.int_dict['xtr_parallel'] and
            opt_o.gen_dict['method'] in ('policy_tree', 'policy tree old',)):
        txt += ('\nContinuous variables are internally split for best use '
                'of cpu ressources.')

    screen = opt_o.dc_dict['screen_covariates']
    percorr = opt_o.dc_dict['check_perfectcorr']
    clean = opt_o.dc_dict['clean_data']
    dummy = opt_o.dc_dict['min_dummy_obs']
    if screen or percorr or clean or dummy:
        txt += '\n' * 2 + 'DATA PREPARATION'
        if screen:
            txt += '\nVariables without variation are removed.'
        if percorr:
            txt += ('\nVariables that are perfectly correlated with other '
                    'variables are removed.')
        if dummy:
            txt += ('\nDummy variables with less than '
                    f'{opt_o.dc_dict["min_dummy_obs"]} observations in the '
                    'smaller group are removed.')
        if clean:
            txt += ('\nRows with any missing values for variables needed for'
                    ' training are removed.')
    if opt_o.gen_dict['method'] in ('best_policy_score', 'bps_classifier'):
        if opt_o.other_dict['restricted']:
            txt += '\n' * 2 + 'RESTRICTIONS on treatment shares'
            txt += '\nRestrictions are ignored if they are not binding.'
            txt += (
                '\nIf they are binding, then several methods are used '
                'to enforce them (almost) exactly:'
                '\n1) Prioritize units that benefit most.'
                '\n2) Deny a random selection of units their best option.'
                )
            if opt_o.var_dict['bb_restrict_name']:
                txt += (
                    '\n3) Prioritize units with higher values of '
                    f'{", ".join(opt_o.var_dict["bb_restrict_name"])}.'
                    )
    elif opt_o.gen_dict['method'] in ('policy_tree', 'policy tree old',):
        if opt_o.other_dict['restricted']:
            txt += '\n' * 2 + 'RESTRICTIONS on treatment shares'
            txt += (
                '\nRestrictions are taken into account by modifying '
                'the policy scores with artificial costs. These '
                'artificial costs are computed such that a Black-Box '
                'allocation will respect the constraints automatically.'
                )
            cm_1 = [cm == 1 for cm in opt_o.other_dict['costs_of_treat_mult']]
            if not all(cm_1):
                mult = [str(mul)
                        for mul in opt_o.other_dict['costs_of_treat_mult']]
                txt += ('These automatic costs are adjusted by the user '
                        f'using the following multipliers: {", ".join(mult)}.')
            txt += (
                '\nIf the allocated treatment shares are not close enough '
                'to the desired shares, then these artificial costs can '
                'be adjusted by specifying/changing the cost '
                'multiplier (keyword "costs_of_treat_mult"). '
                )
            if opt_o.pt_dict['enforce_restriction']:
                txt += (
                    '\nTrees that violate restrictions are not considered. '
                    'Note that this option increases computation substantially '
                    'and may not work well for two sequential optimal trees. '
                    'Use only if you know what you do.'
                    )
        txt += '\n' * 2 + 'COMPUTATIONAL EFFICIENCY'
        txt += (
            '\nOptimal policy trees are computationally very demanding. '
            'Therefore, several approximation parameters are used.'
            '\nWhen splitting, instead of evaluating all values of continuous '
            'variables and combinations of values of categorical variables, '
            f'only {opt_o.pt_dict["no_of_evalupoints"]} values are considered. '
            'These values are equally '
            'spaced for continuous variables and random combinations for '
            'categorical variables. This number is used for EVERY '
            'splitting decision, i.e. the approximation improves the '
            'smaller the data in the leaf becomes. Increasing this value '
            'can significantly improve the computational performance at '
            'the price of a certain approximation loss.'
            )
        txt += (
            '\nThe depth of the tree is also a key parameter. Usually, it '
            'is very hard to estimate trees beyond the depth of 4 '
            '(16 leaves) with reasonably sized training data. There are '
            'two options to improve the computational performance. The '
            'first one is to reduce the depth (leading to loss of '
            'efficiency but a gain in interpretability). The second option '
            'is to split the tree building into several steps. '
            )
        if opt_o.pt_dict["depth_tree_2"] > 2:
            txt += (
                '\nIn this application, this two-step tree buildung option is '
                'implemented in the following way: after buildung the first '
                f'tree of depth {opt_o.pt_dict["depth_tree_1"]-1}, in each '
                'leaf of this tree, a second optimal tree of depth '
                f'{opt_o.pt_dict["depth_tree_2"]-1} is built. Subsequently, '
                'these trees are combined to form the final tree of depth '
                f'{opt_o.pt_dict["depth"]-1}. For given final tree depth, '
                'the more similar the depths of the two trees are, the faster '
                'the algorithm. However, the final tree will of course be '
                'subject to an additional approximation error.'
                )
        else:
            txt += 'However, the second option is not used here.'
        txt += (
            '\nAnother parameter crucial for performance is the '
            'minimum leaf size. Too small leaves may be undesirable for '
            'practical purposes (and they increase computation times). '
            'The minimum leaf size in this application is set to '
            f' {round(opt_o.pt_dict["min_leaf_size"])}.'
            )
        txt += (
            '\nIn addition, the user may reduce the size of the training '
            'data to increase speed, but this will increase sampling noise.'
            )
        txt += '\n' * 2 + 'CATEGORICAL VARIABLES'
        txt += (
            '\nThere are two different approximation methods for larger '
            'categorical variables. Since we build optimal trees, '
            'for categorical variables we need to check all possible '
            'combinations of the different values that lead to binary splits. '
            'This number could indeed be huge. Therefore, we compare only '
            f'{opt_o.pt_dict["no_of_evalupoints"]*2} different combinations. '
            'The available methods differ on how these methods are '
            'implemented. '
            '\nIn this application, at each possible split, '
            )
        if opt_o.pt_dict['select_values_cat']:
            txt += (
                'we randomly draw values from the particular categorical '
                'variable and form groups only using those values only.'
                )
        else:
            txt += (
                'we sort the values of the categorical variables according '
                'to the values of the policy scores as one would do for a '
                'standard random forest. If this set is still too large, a '
                'random sample of the entailed combinations is drawn.'
                )
        if opt_o.pt_dict['eva_cat_mult'] != 1:
            txt += (
                '\nMultiplier used for the number of split points for '
                'categorical variables considered (relative to ordered '
                f'variables: {opt_o.pt_dict["eva_cat_mult"]}')
        if opt_o.report['training_leaf_information'] is not None:
            txt += '\n' * 3 + 'STRUCTURE OF FINAL TREE'
            txt += opt_o.report['training_leaf_information']
    return txt + '\n' * empty_lines_end


def opt_general(opt_o, empty_lines_end):
    """Collect general optimal policy information."""
    txt = '\nMETHOD'
    txt += '\nThe assignment rule is based on allocating units '
    if opt_o.gen_dict['method'] == 'best_policy_score':
        txt += 'to the treatment with the highest score.'
    elif opt_o.gen_dict['method'] in ('policy_tree', 'policy tree old',):
        txt += ('using a shallow decision tree of depth '
                f'{opt_o.pt_dict["depth"]-1}')
        if opt_o.pt_dict["depth_tree_2"] > 2:
            txt += (' (based on 2 optimal trees, '
                    f'depth of 1st tree: {opt_o.pt_dict["depth_tree_1"]-1}, '
                    f'depth of 2nd tree: {opt_o.pt_dict["depth_tree_2"]-1})')
        txt += '.'
        if opt_o.gen_dict['method'] == 'policy tree old':
            txt += '\n   Older, less efficient method used for tree building.'
    elif opt_o.gen_dict['method'] == 'bps_classifier':
        txt += ('using a classifier (for all allocations considered by the '
                'best_policy_score algorithm).')
    txt += '\n\nVARIABLES provided'
    var_dic = opt_o.var_dict
    txt += f'\nPolicy scores: {", ".join(var_dic["polscore_name"])}'
    if var_dic['effect_vs_0']:
        txt += ('\nIATEs relative to first treatment state: '
                f'{", ".join(var_dic["effect_vs_0"])}')
    if var_dic['effect_vs_0_se']:
        txt += ('\nStandard errors of IATEs relative to first treatment state: '
                f'{", ".join(var_dic["effect_vs_0_se"])}')
    if var_dic['polscore_desc_name']:
        txt += ('\nTreatment dependent variables for descriptive analysis: '
                f'{", ".join(var_dic["polscore_desc_name"])}')
    if (opt_o.gen_dict['method'] in ('best_policy_score', 'bps_classifier')
            and var_dic['bb_restrict_name'] and opt_o.other_dict[
                'restricted']):
        txt += ('\nVariables determining prioritisation of units in case of '
                'binding constraints for the best_policy_score method: '
                f'{", ".join(var_dic["bb_restrict_name"])}')
    if var_dic['d_name']:
        txt += f'\nTreatment: {var_dic["d_name"][0]}'
    if var_dic['id_name']:
        txt += f'\nIdentifier: {var_dic["id_name"][0]}'
    if var_dic['x_ord_name']:
        txt += ('\nOderered features: '
                f'{", ".join(var_dic["x_ord_name"])}')
    if var_dic['x_unord_name']:
        txt += ('\nCategorical / unorderered features: '
                f'{", ".join(var_dic["x_unord_name"])}')
    if var_dic['vi_x_name'] and opt_o.gen_dict['variable_importance']:
        txt += ('\nFeatures used for variable importance statistics without '
                f'transformations: {", ".join(var_dic["vi_x_name"])}')
    if var_dic['vi_to_dummy_name'] and opt_o.gen_dict['variable_importance']:
        txt += ('\nFeatures that are '
                'transformed to indicator/dummy variables for variable '
                'importance computations (only): '
                f'{", ".join(var_dic["vi_to_dummy_name"])}')
    txt += '\n' * 2 + 'COSTS'
    if (opt_o.other_dict['costs_of_treat'] is not None
        and len(opt_o.other_dict['costs_of_treat']) > 1
            and sum(opt_o.other_dict["costs_of_treat"]) > 0):
        cost = [str(cos) for cos in opt_o.other_dict["costs_of_treat"]]
        txt += ('\n' * 2 + 'Used provided treatment specific costs: '
                f'{", ".join(cost)}')
    else:
        txt += '\nNo user provided costs of specific treatments.'
    txt += '\n' * 2 + 'RESTRICTIONS of treatment shares'
    if opt_o.other_dict['restricted']:
        share = [f'\nTreatment: {treat:4d}     maximum share: {shr:>7.1%}'
                 for treat, shr in enumerate(opt_o.other_dict['max_shares'])]
        txt += ('\nThe following restrictions on the treatment shares are '
                f'specified: {" ".join(share)}.'
                )
    else:
        txt += '\nTreatment shares are unrestricted.'

    txt += '\n' * 2 + 'FAIRNESS'
    if opt_o.fair_dict['solvefair_used']:
        txt += '\nFairness adjustments prior to policy learning.'
        protected_var = var_dic['protected_ord_name'].copy()
        protected_var.extend(var_dic['protected_unord_name'])
        txt += f'\nProtected variables: {", ".join(protected_var)}'
        material_var = var_dic['material_ord_name'].copy()
        material_var.extend(var_dic['material_unord_name'])
        if material_var:
            txt += f'\nMaterially relevant variables: {", ".join(material_var)}'
        if opt_o.fair_dict["adjust_target"] == 'xvariables':
            adjust_objects = 'Decision variables'
        elif opt_o.fair_dict["adjust_target"] == 'scores':
            adjust_objects = 'Policy scores'
        else:
            adjust_objects = 'Policy scores and decision variables'
        txt += f'\nAdjusted objects: {adjust_objects}'
        txt += f'\nAdjustmend method used: {opt_o.fair_dict["adj_type"]}'
        txt += ('\nDiscretization method used for protected variables: '
                f' {opt_o.fair_dict["protected_disc_method"]}'
                )
        if material_var:
            txt += ('\nDiscretization method used for materially relevant '
                    f'variables: {opt_o.fair_dict["material_disc_method"]}'
                    )
        txt += '\nFor more detail information on methods used see output files.'
    else:
        txt += '\nNo fairness adjustments performed.'

    txt += '\n' * 2 + 'ESTIMATION RISK'
    if opt_o.estrisk_dict['estrisk_used']:
        txt += ('\nScores are adjusted for estimation error of the policy '
                'scores prior to policy learning.')
    else:
        txt += '\nNo adjustments for estimation error performed.'

    return txt + '\n' * empty_lines_end
