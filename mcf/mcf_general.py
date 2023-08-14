"""
Contains general functions.

Created on Thu May 11 16:30:11 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from math import log, prod

import numpy as np
from sympy.ntheory import primefactors


def dic_get_list_of_key_by_item(dic, value):
    """Get list of keys by item of a dictionary.

    Parameters
    ----------
    dic : Dictionary.
    value : Particular value of interest.

    Returns
    -------
    key_list : List of keys that have the value VALUE.

    """
    key_list = []
    for keys in dic.keys():
        if dic[keys] in value:
            key_list += [keys]
    if not key_list:
        raise ValueError('Retrieving items from list was not succesful')
    return key_list


def get_key_values_in_list(dic):
    """Create two lists with keys and values of dictionaries.

    Parameters
    ----------
    dic : Dictionary.

    Returns
    -------
    key_list : List of keys.
    value_list : List of values.

    """
    key_list, value_list = [], []
    for keys in dic.keys():
        key_list += [keys]
        value_list += [dic[keys]]
    return key_list, value_list


def list_product(factors):
    """Prodcuce a product of a list keeping python data format.

    Parameters
    ----------
    factors : List.

    Returns
    -------
    prod : INT or Float.

    """
    return prod(factors)  # should be faster and keep python format


def substitute_variable_name(var_dic, old_name, new_name):
    """Exchanges values in a dictionary.

    Parameters
    ----------
    v_dict : Dictionary
    old_name : string, Value to change
    new_name : string, new value

    Returns
    -------
    v_dict : Dictionary with changed names

    """
    vn_dict = deepcopy(var_dic)
    for i in var_dic:
        list_of_this_dic = var_dic[i]
        if list_of_this_dic is not None:
            for j, elem in enumerate(list_of_this_dic):
                if elem == old_name:
                    list_of_this_dic[j] = new_name
            vn_dict[i] = list_of_this_dic
    return vn_dict


def adjust_vars_vars(var_in, var_weg):
    """Remove from VAR_IN those strings that are also in VAR_WEG.

    Parameters
    ----------
    var_in : list of strings.
    var_weg : list of strings.

    Returns
    -------
    ohne_var_weg : list of strings

    """
    # v_inter = set(var_in).intersection(set(var_weg))
    # ohne_var_weg = (list(set(var_in)-v_inter) if not v_inter == set()
    #                 else deepcopy(var_in))
    ohne_var_weg = [var for var in var_in if var not in var_weg]
    return ohne_var_weg


def adjust_var_name(var_to_check, var_names):
    """
    Check if upper or lower case version of name is in namelist and adjust.

    Parameters
    ----------
    var_to_check : Str.
    var_names : List of Str.

    Returns
    -------
    var_to_check : Str.

    """
    if var_to_check not in var_names:
        for name in var_names:
            if (var_to_check.upper() == name.upper()) or (
                    var_to_check.lower() == name.lower()):
                var_to_check = name
                break
    return var_to_check


def cleaned_var_names(var_name):
    """Clean variable names.

    Cleaning variable by removing empty list and zero and None and putting
    everything to upper case & removing duplicates

    Parameters
    ----------
    var_name : List with variable names

    Returns
    -------
    var_name2 : List with variable names

    """
    if var_name is None:
        var_name = []
    if any(s is isinstance(s, (tuple, list)) for s in var_name):
        raise ValueError(f'{var_name} must be a list or tuple. It seems '
                         ' that it is list/tuple of lists/tuples.')
    var_name1 = [s.upper() for s in var_name]
    var_name2 = []
    for var in var_name1:
        if (var not in var_name2) and (var != '0') and (var != 0) and (
                var != []) and (var is not None):
            var_name2.append(var)
    return var_name2


def add_var_names(names1, names2=None, names3=None, names4=None, names5=None,
                  names6=None, names7=None, names8=None, names9=None,
                  names10=None):
    """Return a list of strings with unique entries."""
    def none_to_empty_list(name):
        if name is None:
            name = []
        return name

    names2 = none_to_empty_list(names2)
    names3 = none_to_empty_list(names3)
    names4 = none_to_empty_list(names4)
    names5 = none_to_empty_list(names5)
    names6 = none_to_empty_list(names6)
    names7 = none_to_empty_list(names7)
    names8 = none_to_empty_list(names8)
    names9 = none_to_empty_list(names9)
    names10 = none_to_empty_list(names10)
    new_names = deepcopy(names1)
    new_names.extend(names2)
    new_names.extend(names3)
    new_names.extend(names4)
    new_names.extend(names5)
    new_names.extend(names6)
    new_names.extend(names7)
    new_names.extend(names8)
    new_names.extend(names9)
    new_names.extend(names10)
    new_names = cleaned_var_names(new_names)
    return new_names


def to_list_if_needed(string_or_list):
    """Help for initialisation."""
    if isinstance(string_or_list, (tuple, set)):
        return list(string_or_list)
    if isinstance(string_or_list, str):
        return [string_or_list]
    return string_or_list


def check_if_iterable(variable):
    """Return an iterable if not already iterable."""
    iter_type = (tuple, list, np.ndarray)
    variable_list = variable if isinstance(variable, iter_type) else [variable]
    return variable_list


def recode_if_all_prime(values, name):
    """
    Recode array-like to of prime to list of integers.

    Parameters
    ----------
    values : array like.
        Input with numbers.

    Returns
    -------
    new_values : List of INT.
        List of positions of prime in list of consequative primes.

    """
    values_l = [int(val) for val in values]   # list of integers
    is_prime = set(values_l).issubset(primes_list())
    new_name = name
    if is_prime:
        values_l = primeposition(values_l, start_with_1=False)
        if name is not None and name[-2:] == 'PR':
            new_name = name[:-2] + '(mayberec0)'
    return values_l, new_name


def primeposition(x_values, start_with_1=False):
    """
    Give position of elements of x_values in list of primes.

    Parameters
    ----------
    x_values : List of int.

    Returns
    -------
    position : List of int.

    """
    add = 1 if start_with_1 else 0
    primes = primes_list(1000)
    position = [primes.index(val)+add for val in x_values]
    return position


def primes_list(number=1000):
    """List the first 1000 prime numbers.

    Parameters
    ----------
    number : INT. length of vector with the first primes.

    Returns
    -------
    primes: list of INT, prime numbers

    """
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
              61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
              127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
              193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
              269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347,
              349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
              431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
              503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593,
              599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661,
              673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757,
              761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839,
              853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937,
              941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019,
              1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
              1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171,
              1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249,
              1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319,
              1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429,
              1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489,
              1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571,
              1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637,
              1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733,
              1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823,
              1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907,
              1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999,
              2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083,
              2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153,
              2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267,
              2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341,
              2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411,
              2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521,
              2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617,
              2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689,
              2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753,
              2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843,
              2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939,
              2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037,
              3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137,
              3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229,
              3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323,
              3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407,
              3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511,
              3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581,
              3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671,
              3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761,
              3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851,
              3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929,
              3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021,
              4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127,
              4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219,
              4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289,
              4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409,
              4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507,
              4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597,
              4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679,
              4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789,
              4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903,
              4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973,
              4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059,
              5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167,
              5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273,
              5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387,
              5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449,
              5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531,
              5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651,
              5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737,
              5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827,
              5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897,
              5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037,
              6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121,
              6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217,
              6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301,
              6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373,
              6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491,
              6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599,
              6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701,
              6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793,
              6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883,
              6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977,
              6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069,
              7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193,
              7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297,
              7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417,
              7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517,
              7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583,
              7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681,
              7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759,
              7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879,
              7883, 7901, 7907, 7919)
    return primes[0:number]


def primes_reverse(number, int_type=True):
    """Give the prime factors of integers.

    Parameters
    ----------
    number : INT, the variable to split into prime factors
    int_type : Boolean: One of number is of type INT32 or INT64.
                        The default is True.
                        It is easier to use TRUE in other operations, but with
                        False it may be possible to pass (and split) much
                        larger numbers
    Returns
    -------
    list_of_primes : INT (same as input)

    """
    if int_type:
        number = number.tolist()
    list_of_primes = primefactors(number)  # Should be faster
    return list_of_primes


def check_if_not_number(data_df, variable):
    """Check if elements in column of pandas dataframe are not a number.

    Parameters
    ----------
    data_df : Dataframe. Variables to check.
    variable : String or list of strings.

    Raises
    ------
    Exception : Stops programme.

    Returns
    -------
    None.

    """
    is_number_mask = np.array(data_df[variable].applymap(np.isreal))
    var_not_a_number = []
    for idx, var in enumerate(variable):
        if not np.all(is_number_mask[:, idx]):
            var_not_a_number.append(var)
    if var_not_a_number:
        raise ValueError(' '.join(var_not_a_number) + 'are no numbers.'
                         ' Number format is needed for this variable.')


def grid_log_scale(large, small, number):
    """Define a logarithmic grid.

    Parameters
    ----------
    large : INT or FLOAT: Largest value of grid.
    small : INT or FLOAT: Smallest value of grid.
    number : INT: Number of grid points.

    Returns
    -------
    List with grid.

    """
    if small <= 0.0000001:
        small = 0.00000001
    small, large = log(small), log(large)
    sequence = np.unique(np.round(np.exp(np.linspace(small, large, number))))
    sequence_p = sequence.tolist()
    return sequence_p


def share_completed(current, total):
    """Count how much of a task is completed and print to terminal.

    Parameters
    ----------
    current : INT. No of tasks completed.
    total : INT. Total number of tasks.

    Returns
    -------
    None.

    """
    if current == 1:
        print("\nShare completed (%):", end=" ")
    share = current / total * 100
    if total < 20:
        print(f'{share:4.0f}', end=" ", flush=True)
    else:
        points_to_print = range(1, total, round(total/20))
        if current in points_to_print:
            print(f'{share:4.0f}', end=" ", flush=True)
    if current == total:
        print('Task completed')


def bound_norm_weights(weight, max_weight, renormalize=True):
    """Bound and renormalized weights.

    Parameters
    ----------
    weight : 1d Numpy array. Weights.
    max_weight : Scalar Float. Maximum value of any weight
    renormalize : Boolean, optional. If True renormalize the weights that they
               add to 1. The default is True.

    Returns
    -------
    weight_norm : Numpy array of same size as input. Normalized weights.
    no_censored: NP float. Number of censored observations.
    share_censored: NP float. Share of censored observations (0-1).

    """
    weight_norm = weight.flatten()
    too_large = (weight + 1e-15) > max_weight
    if np.any(too_large):
        no_censored = np.count_nonzero(too_large)
        weight_norm[too_large] = max_weight
    else:
        no_censored = 0
    share_censored = no_censored / len(weight)
    if renormalize:
        sum_w = np.sum(weight_norm)
        if not ((-1e-10 < sum_w < 1e-10) or (1-1e-10 < sum_w < 1+1e-10)):
            weight_norm = weight_norm / sum_w
    return weight_norm, no_censored, share_censored


def remove_dupl_keep_order(input_list):
    """Remove duplicates from a list but preserves the order."""
    output_list = []
    seen = set()  # Keep track of seen elements
    for item in input_list:
        if item not in seen:
            output_list.append(item)
            seen.add(item)
    return output_list
