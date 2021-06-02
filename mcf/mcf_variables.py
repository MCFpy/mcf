"""Definition of variables for MCF.

Created on Fri Apr  3 11:03:23 2020

@author: MLechner

# -*- coding: utf-8 -*-
"""


def make_user_variable(flag1=None, flag2=None, flag3=None, flag4=None):
    """Variables for the MCF - has to be modified by user.

    IMPORTANT NOTE: PROGRAMME WILL NOT DISTINGHUISH VARIABLES NAMES IN CAPITAL
    OR SMALL LETTERS. THEY WILL ALL BE TREATED THE SAME.
    MAKE SURE THAT IN THE DATA SET SUCH DISTINCTION DOES NOT MATTER, OTHERWISE
    ALL VARIABLES THAT HAVE THE SAME NAME WHEN CAPITALIZED WILL BE DELETED
    Use empty list if there is no variable from this category
    use ['variable1','variable2',...] syntax (with strings and comma)

    FLAG1-FLAG4 may used to pass additional variables to this procedure
    (e.g. for defining conditional variable lists)

    Returns
    -------
    Dictionary with different variable names as lists.
    """
# Identifier
    id_name = ['ID']
# If no identifier -> it will be added the data that is saved for later use
# Cluster and panel identifiers
    cluster_name = ['cluster']  # Variable defining the clusters
    w_name = ['weight']         # Name of weight, if weighting option is used
# Treatment
    d_name = ['d']              # Must be discrete
# Dependent (outcome) variable
    y_tree_name = ['y']         # Variable to build trees
    y_name = ['y']       # Used to evaluate the effect,
# if not included variable to build tree will be added

# Features, predictors, independent, confounders variables: ordered
    x_name_ord = []
    if flag1:
        for i in range(120):
            x_name_ord.append('cont' + str(i))
        for i in range(10):
            x_name_ord.append('dum' + str(i))
        for i in range(10):
            x_name_ord.append('ord' + str(i))
    else:
        for i in range(5):
            x_name_ord.append('cont' + str(i))
        for i in range(3):
            x_name_ord.append('dum' + str(i))
        for i in range(3):
            x_name_ord.append('ord' + str(i))

# Features,predictors, independent, confounders variables: unordered,categorial
# PROGRAMME WILL BE MUCH FASTER IF VARIABLES HAVE fewer CATEGORIES !!!!
# More categories are possible but will slow down programme and increase memory
    x_name_unord = []
    if flag1:
        for i in range(10):
            x_name_unord.append('cat' + str(i))
    else:
        for i in range(2):
            x_name_unord.append('cat' + str(i))
    # Variables always included when deciding next split
    x_name_always_in_ord = []
    x_name_always_in_unord = []

# Variables to define policy relevant heterogeneity in Multiple treatment
# procedure; this is based on weight-based inference
# variables must not be influenced by the treatment; they will be added to the
# conditioning set
    if flag1:
        Z_name_list = [
         'cont0', 'cont2',
        ]
    else:
        Z_name_list = [
          'cont0'
            ]
# Ordered variables with many values
# ( Put discrete variables with few values in category below!!)
# These variables have to be recoded to define the split, they will be added
# to the list of confounders since they are broken up in dummies for all
# their observed values, it does not matter whether they are coded as ordered
# or unordered --> dummies are always treated as ordered because this is
# fastes --> PUT them below
    if flag1:
        z_name_split_ord = [
        'dum0', 'dum2', 'ord0', 'ord2']
    else:    
        z_name_split_ord = [
       'ord0', 'dum0'
        ]
# Variables that are discrete and define a unique sample split for each value
# which the effects will be evaluated, will be added to the list of confounded
    if flag1:
        z_name_split_unord = [
        'cat0', 'cat2'
                ]
    else:    
        z_name_split_unord = [
        'cat0', 
        ]
# Variables that are discrete and define a unique sample split for each value
# which the effects will be evaluated; will be added to the list of confounders

    if flag1:
        z_name_mgate = [
        'cont0', 'cont2', 'cat0', 'cat2', 'ord0', 'ord2'
                ]
    else:    
        z_name_mgate = [
            'cont0', 'cat0', 'ord0'
            ]
    # Names of variables for which marginal GATE (at median) will be computed
    # Variable must be in included x_name_ord or x_name_unord; otherwise
    # variables will be deleted from list
    if flag1: 
        z_name_amgate = [
            'cont0', 'cont2', 'cat0', 'cat2', 'ord0', 'ord2'
            ]
    else:    
        z_name_amgate = [ 
             'cont0',  'cat0', 'ord0'
            ]
    # Names of variables for which average marginal GATE will be computed
    # Variable must be in included x_name_ord or x_name_unord; otherwise
    # variables will be deleted from list

# Variable to be excluded from preliminary feature selection
    x_name_remain_ord = []
    x_name_remain_unord = []

# Variables for balancing tests (will also be excluded from feature selection)
    if flag1:
        x_balance_name_ord = ['cont0', 'cont1', 'cont2', 'cont3', 'cont4',
                              'cont5', 'cont6', 'cont7', 'cont8', 'cont9',
                              'dum0', 'dum1', 'ord0', 'ord1']
        x_balance_name_unord = ['cat0', 'cat1']
    else:
        x_balance_name_ord = ['cont0']
        x_balance_name_unord = ['cat0']

# No change below this line -------------------------------------------------
    variable_dict = {'id_name': id_name, 'cluster_name': cluster_name,
                     'w_name': w_name, 'd_name': d_name,
                     'y_tree_name': y_tree_name, 'y_name': y_name,
                     'x_name_ord': x_name_ord, 'x_name_unord': x_name_unord,
                     'x_name_always_in_ord': x_name_always_in_ord,
                     'z_name_list': Z_name_list,
                     'x_name_always_in_unord': x_name_always_in_unord,
                     'z_name_ord': z_name_split_ord,
                     'z_name_unord': z_name_split_unord,
                     'z_name_mgate': z_name_mgate,
                     'z_name_amgate': z_name_amgate,
                     'x_name_remain_ord': x_name_remain_ord,
                     'x_name_remain_unord': x_name_remain_unord,
                     'x_balance_name_ord': x_balance_name_ord,
                     'x_balance_name_unord': x_balance_name_unord,
                     }
    return variable_dict
