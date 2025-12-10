"""
Contains general functions.

Created on Thu May 11 16:30:11 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from dataclasses import is_dataclass, fields
from functools import lru_cache
from math import log, prod, isnan
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from sympy.ntheory import primefactors

try:
    import torch  # type: ignore[import]
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]


# Alternatively using functools.lru_cache may be faster
def memoize(func: Any) -> Any:
    """Save in for storing computed results."""
    cache = {}  # Cache for storing computed results

    def wrapper(*args):
        if args in cache:
            return cache[args]

        result = func(*args)
        cache[args] = result

        return result

    return wrapper


def memoize_list(func: Any) -> Any:
    """Save in for storing computed results."""
    cache = {}  # Cache for storing computed results

    def wrapper(args):
        arg_tuple = tuple(args)
        if arg_tuple in cache:
            return cache[arg_tuple]

        result = func(args)
        cache[arg_tuple] = result

        return result

    return wrapper


def dic_get_list_of_key_by_item(dic: dict,
                                value: list[Any] | tuple[Any] | set[Any],
                                ) -> list[Any]:
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


def get_key_values_in_list(dic: dict) -> tuple[list[Any], list[Any]]:
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


# @memoize_list  To much overhead, does not save anything relevant
def list_product(factors: list[int]) -> int:
    """Prodcuce a product of a list keeping python data format.

    Parameters
    ----------
    factors : List of Int.

    Returns
    -------
    prod : Int. Product of primes.

    """
    return prod(factors)  # should be fast and keep python format


def substitute_variable_name(var_cfg: Any,
                             old_name: str,
                             new_name: str
                             ) -> dict:
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
    vn_new = deepcopy(var_cfg)

    if is_dataclass(var_cfg):
        for f in fields(vn_new):
            list_of_this_dc = getattr(vn_new, f.name)
            if isinstance(list_of_this_dc, (list, tuple,)):
                for j, elem in enumerate(list_of_this_dc):
                    val = getattr(vn_new, f.name)
                    if isinstance(val, list):
                        # Transform in-place so no subsequent setattr needed
                        val[:] = [new_name if x == old_name else x for x in val]
    else:  # No longer used. Speed increase possible by using list comprehension
        for i in vn_new:
            list_of_this_dic = vn_new[i]
            if list_of_this_dic is not None:
                for j, elem in enumerate(list_of_this_dic):
                    if elem == old_name:
                        list_of_this_dic[j] = new_name
                vn_new[i] = list_of_this_dic

    return vn_new


def adjust_vars_vars(var_in: list[str], var_weg: list[str]) -> list[str]:
    """Remove from VAR_IN those strings that are also in VAR_WEG.

    Parameters
    ----------
    var_in : list of strings.
    var_weg : list of strings.

    Returns
    -------
    ohne_var_weg : list of strings

    """
    ohne_var_weg = [var for var in var_in if var not in var_weg]

    return ohne_var_weg


def adjust_var_name(var_to_check: str, var_names: list[str]) -> str:
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
            if ((var_to_check.upper() == name.upper())
                or (var_to_check.lower() == name.lower())
                    or (var_to_check.casefold() == name.casefold())):
                var_to_check = name
                break

    return var_to_check


def cleaned_var_names(var_name: list[str]) -> list[str]:
    """Clean variable names.

    Cleaning variable by removing empty list and zero and None and putting
    everything to lower case & removing duplicates

    Parameters
    ----------
    var_name : List with variable names

    Returns
    -------
    var_name2 : List with variable names

    """
    if var_name is None or var_name == [None]:
        var_name = []
    if isinstance(var_name, str):
        var_name = [var_name]
    if any(s is isinstance(s, (tuple, list)) for s in var_name):
        raise ValueError(f'{var_name} must be a list or tuple. It seems '
                         ' that it is list/tuple of lists/tuples.')
    var_name1 = [s.casefold() for s in var_name]
    var_name2 = []
    for var in var_name1:
        if (var not in var_name2) and (var != '0') and (var != 0) and (
                var != []) and (var is not None):
            var_name2.append(var)

    return var_name2


def add_var_names(names1: list[str],
                  names2: list[str] = None,
                  names3: list[str] = None,
                  names4: list[str] = None,
                  names5: list[str] = None,
                  names6: list[str] = None,
                  names7: list[str] = None,
                  names8: list[str] = None,
                  names9: list[str] = None,
                  names10: list[str] = None
                  ) -> list[str]:
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


def to_list_if_needed(string_or_list: str | list[str] | tuple[str] | set[str]
                      ) -> list[str]:
    """Help for initialisation."""
    if isinstance(string_or_list, (tuple, set)):
        if len(string_or_list) == 0:
            return []
        if isinstance(string_or_list[0], (tuple, set)):
            string_or_list = [item for sublist in string_or_list
                              for item in sublist]
        return list(string_or_list)

    if isinstance(string_or_list, str):
        return [string_or_list]

    return string_or_list


def check_if_iterable(variable: str | tuple | list | NDArray
                      ) -> tuple[tuple | list | NDArray]:
    """Return an iterable if not already iterable."""
    iter_type = (tuple, list, np.ndarray)
    variable_list = variable if isinstance(variable, iter_type) else [variable]

    return variable_list


def recode_if_all_prime(values: tuple | list | NDArray,
                        name: str | None,
                        values_unord_org: str | None
                        ) -> tuple[list[int], str | None]:
    """
    Recode array-like of prime to list of integers.

    Parameters
    ----------
    values : array like.
        Input with numbers.

    Returns
    -------
    new_values : List of INT.
        List of positions of prime in list of consequative primes.

    """
    values_l = [int(val) if not isnan(val) else val for val in values]
    # list of integers
    is_prime = set(values_l).issubset(primes_list())
    new_name = name
    if is_prime:
        if name is not None and name.endswith('_prime'):
            # new_name = name[:-6] + '(mayberec0)'
            # new_name_dic = name[:-6]
            new_name = name[:-6]
        else:
            new_name = None
        indices_l = primeposition(values_l, start_with_1=False)
        if values_unord_org is None:
            values_l = indices_l
        else:
            if new_name is None:
                values_unord = values_unord_org
            else:
                values_unord = values_unord_org[new_name]
            values_l = [values_unord[j] for j in indices_l]

    return values_l, new_name


def primeposition(x_values: list[int],
                  start_with_1: bool = False
                  ) -> list[int]:
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


def primes_list(number: int = 1000) -> list[int]:
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


@lru_cache(maxsize=200000)
def primes_reverse(number: int, int_type: bool = True) -> list[int]:
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


def check_if_not_number(data_df: DataFrame, variable: str | list[str]) -> None:
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
    is_number_mask = np.array(data_df[variable].map(np.isreal))
    var_not_a_number = []
    for idx, var in enumerate(variable):
        if not np.all(is_number_mask[:, idx]):
            var_not_a_number.append(var)
    if var_not_a_number:
        raise ValueError(' '.join(var_not_a_number) + 'are no numbers.'
                         ' Number format is needed for this variable.')


def grid_log_scale(large: int | float,
                   small: int | float,
                   number: int
                   ) -> list[int | float]:
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


def share_completed(current: int, total: int) -> None:
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


def bound_norm_weights_not_one(weight: NDArray,
                               max_weight: float = 0.05,
                               renormalize: bool = True,
                               zero_tol: float = 1e-15,
                               sum_tol: float = 1e-12,
                               ) -> tuple[NDArray, np.intp, np.floating]:
    """Bound and renormalized weights that do not add up to 1.

    Parameters
    ----------
    weight : 1d Numpy array. Weights.
    max_weight : Scalar Float. Maximum value of any weight (if weights sum to 1)
    renormalize : Boolean, optional. If True renormalize the weights that they
               add to their previous sum. The default is True.

    Returns
    -------
    weight_norm : Numpy array of same size as input. Normalized weights.
    no_censored: NP float. Number of censored observations.
    share_censored: NP float. Share of censored observations (0-1).

    """
    weight_norm = weight.flatten()
    sum_w_org = np.sum(weight_norm)
    sum_w_org_abs = np.sum(np.abs(weight_norm))

    # Define threshold (no change if sum of weights equals 1)
    max_weight_adj = max_weight * sum_w_org_abs

    # Restrict weights that that are too positive or too negative
    too_large = (weight + zero_tol) > max_weight_adj
    too_small = (weight - zero_tol) < -max_weight_adj
    no_censored = 0

    # Set too large weights to max_weight_adj
    if np.any(too_large):
        no_censored += np.count_nonzero(too_large)
        weight_norm[too_large] = max_weight_adj

    # Set too small negative weights to minus max_weight_adj
    if np.any(too_small):
        no_censored += np.count_nonzero(too_small)
        weight_norm[too_small] = -max_weight_adj

    share_censored = no_censored / len(weight)

    # Renormalize sum of weights to orginal sum of weights
    if renormalize:
        factor = sum_w_org / np.sum(weight_norm)

        # Require no sign change and sufficiently different from 1
        if factor > sum_tol and not 1-sum_tol < factor < 1+sum_tol:
            weight_norm = weight_norm * factor

    return weight_norm, no_censored, share_censored


def bound_norm_weights(weight: NDArray,
                       max_weight_share: float = 0.05,
                       renormalize: bool = True,
                       zero_tol: float = 1e-15,
                       sum_tol: float = 1e-15,
                       negative_weights_possible: bool = False,
                       ) -> tuple[NDArray, np.intp, np.floating]:
    """Bound and renormalized weights.

    Parameters
    ----------
    weight : 1d Numpy array. Weights.
    max_weight : Scalar Float. Maximum value of any weight.
    renormalize : Boolean, optional. If True renormalize the weights that they
               add to their previous sum. The default is True.
    ...

    Returns
    -------
    weight_norm : Numpy array of same size as input. Normalized weights.
    no_censored: NP float. Number of censored observations.
    share_censored: NP float. Share of censored observations (0-1).

    """
    no_censored = 0
    weight_norm = weight.flatten()
    if negative_weights_possible:
        # There could be negative weights; pos and neg weights sum up tp 1
        max_weight = np.sum(np.abs(weight_norm)) * max_weight_share
    else:
        # All weights are positive and sum up to 1
        max_weight = max_weight_share

    too_positive = (weight + zero_tol) > max_weight
    if np.any(too_positive):
        no_censored += np.count_nonzero(too_positive)
        weight_norm[too_positive] = max_weight

    if negative_weights_possible:
        too_negative = (weight - zero_tol) < -max_weight
        if np.any(too_negative):
            no_censored += np.count_nonzero(too_negative)
            weight_norm[too_negative] = -max_weight

    share_censored = no_censored / len(weight)

    if renormalize:
        sum_w = np.sum(weight_norm)
        if not ((-sum_tol < sum_w < sum_tol)
                or (1-sum_tol < sum_w < 1+sum_tol)):
            weight_norm = weight_norm / sum_w

    return weight_norm, no_censored, share_censored


def bound_norm_weights_cuda(weight: torch.Tensor,
                            max_weight_share: float = 0.05,
                            renormalize: bool = True,
                            zero_tol: float = 1e-15,
                            sum_tol: float = 1e-12,
                            negative_weights_possible: bool = False,
                            ) -> tuple[torch.Tensor,
                                       torch.Tensor,
                                       torch.Tensor]:
    """Bound and renormalize weights (tensor version).

    Parameters
    ----------
    weight : 1d Tensor. Weights, typically on CUDA.
    max_weight_share : float. Maximum share of the (absolute) total weight
        that any single weight is allowed to have.
    renormalize : bool, optional. If True, renormalize the weights so they
        sum to 1 (unless the sum is ~0 or already ~1). Default is True.
    negative_weights_possible : bool, optional. If True, allow negative
        weights and bound both positive and negative sides.

    Returns
    -------
    weight_norm : Tensor of same size as input. Normalized weights.
    no_censored : 0-dim integer tensor. Number of censored observations.
    share_censored : 0-dim float tensor. Share of censored observations (0–1).
    """

    # Work on a flattened view
    weight_norm = weight.reshape(-1)

    # Determine max allowed absolute weight
    if negative_weights_possible:
        # Positive and negative weights may exist; they (roughly) sum to 1
        max_weight = weight_norm.abs().sum() * max_weight_share
    else:
        # All weights are positive and (roughly) sum to 1
        max_weight = torch.as_tensor(max_weight_share,
                                     dtype=weight_norm.dtype,
                                     device=weight_norm.device,
                                     )
    # Masks on the flattened tensor
    too_positive = (weight_norm + zero_tol) > max_weight
    if negative_weights_possible:
        too_negative = (weight_norm - zero_tol) < -max_weight
        censored_mask = too_positive | too_negative
    else:
        too_negative = None  # for clarity
        censored_mask = too_positive

    # Number of censored entries as tensor
    no_censored = torch.count_nonzero(censored_mask)

    # Apply clipping
    if torch.any(too_positive):
        weight_norm[too_positive] = max_weight
    if (negative_weights_possible
        and too_negative is not None
            and torch.any(too_negative)):
        weight_norm[too_negative] = -max_weight

    # Share censored: keep as tensor on same device
    share_censored = (no_censored.to(dtype=weight_norm.dtype)
                      / weight_norm.numel()
                      )

    # Renormalize to sum ≈ 1 if requested
    if renormalize:
        sum_w = weight_norm.sum()
        sum_w_val = sum_w.item()

        # Do NOT renormalize if sum is ~0 or already ~1
        close_to_zero = abs(sum_w_val) < sum_tol
        close_to_one = abs(sum_w_val - 1.0) < sum_tol

        if not (close_to_zero or close_to_one):
            # Safe: we've excluded near-zero sums
            weight_norm = weight_norm / sum_w_val

    return weight_norm, no_censored, share_censored


def bound_norm_weights_not_one_cuda(weight: torch.Tensor,
                                    max_weight: float = 0.05,
                                    renormalize: bool = True,
                                    zero_tol: float = 1e-15,
                                    sum_tol: float = 1e-12,
                                    ) -> tuple[torch.Tensor,
                                               torch.Tensor,
                                               torch.Tensor]:
    """Bound and renormalize weights that do not add up to 1 (tensor version).

    Parameters
    ----------
    weight : 1d Tensor. Weights, typically on CUDA.
    max_weight : float. Maximum absolute value of any weight (if weights sum to 1).
    renormalize : bool, optional. If True, renormalize the weights so they
        add up to their original sum. Default is True.

    Returns
    -------
    weight_norm : Tensor of same size as input. Normalized weights.
    no_censored : 0-dim integer tensor. Number of censored observations.
    share_censored : 0-dim float tensor. Share of censored observations (0–1).
    """
    # Work on a flattened view (no copy if weight is already 1d/contiguous)
    weight_norm = weight.reshape(-1)

    # Original sums
    sum_w_org = weight_norm.sum()
    sum_w_org_abs = weight_norm.abs().sum()

    # Threshold (no change if sum of weights equals 1)
    max_weight_adj = max_weight * sum_w_org_abs

    # Masks for too-large / too-small weights (with zero tolerance buffer)
    too_large = (weight_norm + zero_tol) > max_weight_adj
    too_small = (weight_norm - zero_tol) < -max_weight_adj
    censored_mask = too_large | too_small

    # Number of censored entries as tensor
    no_censored = torch.count_nonzero(censored_mask)

    # Apply bounding only where needed
    if torch.any(too_large):
        weight_norm[too_large] = max_weight_adj
    if torch.any(too_small):
        weight_norm[too_small] = -max_weight_adj

    # Fraction censored, keep as float tensor on same device
    share_censored = (no_censored.to(dtype=weight_norm.dtype)
                      / weight_norm.numel()
                      )
    # Renormalize sum of weights to original sum of weights
    if renormalize:
        denom = weight_norm.sum()
        # Avoid division by ~0
        if torch.abs(denom) > sum_tol:
            factor = sum_w_org / denom
            factor_val = factor.item()  # scalar float

            # Require positive factor and sufficiently different from 1
            if (factor_val > sum_tol
                    and not 1.0 - sum_tol < factor_val < 1.0 + sum_tol):
                weight_norm = weight_norm * factor_val

    return weight_norm, no_censored, share_censored


def remove_dupl_keep_order(input_list: list[Any]) -> list[Any]:
    """Remove duplicates from a list but preserves the order."""
    output_list = []
    seen = set()  # Keep track of seen elements
    for item in input_list:
        if item not in seen:
            output_list.append(item)
            seen.add(item)

    return output_list


def include_org_variables(names: list[str],
                          names_in_data: list[str]
                          ) -> list[str]:
    """Add levels if not already in included."""
    new_names = names[:]
    for name in names:
        if len(name) > 4:
            if name.endswith('catv') and name[:-4] in names_in_data:
                if name[:-4] not in names:
                    new_names.append(name[:-4])
            if len(name) > 6:
                if name.endswith('_prime') and name[:-6] in names_in_data:
                    if name[:-6] not in names:
                        new_names.append(name[:-6])
    return new_names


def check_reduce_dataframe(data_df: DataFrame,
                           title: str = '',
                           max_obs: int = 100000,
                           seed: int = 124535,
                           ignore_index: bool = True
                           ) -> tuple[DataFrame, bool, str]:
    """Randomly reduce dataframe to a certain number of observations."""
    total_obs = len(data_df)
    if rnd_reduce := total_obs > max_obs:
        data_df = data_df.sample(n=max_obs,
                                 random_state=seed,
                                 replace=False,
                                 ignore_index=ignore_index)
        txt = (f'{title}: Sample randomly reduced from {total_obs} to '
               f'{max_obs} observations.')
    else:
        txt = ''
    return data_df, rnd_reduce, txt


def to_numpy_big_data(data_df: DataFrame, obs_bigdata: int) -> NDArray:
    """Determine datatype when transforming to numpy."""
    data_np = data_df.to_numpy()

    if len(data_np) > obs_bigdata and data_np.dtype == np.float64:
        data_np = data_np.astype(np.float32)

    return data_np


def remove_duplicates(lst: list[Any]) -> list[Any]:
    """Remove duplicates from list without changing order."""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)

    return result


def unique_list(list_tuple: list[Any] | tuple[Any] | set[Any]) -> list[Any]:
    """Remove duplicate elements from list without changing order."""
    unique = []
    for item in list_tuple:
        if item not in unique:
            unique.append(item)

    return unique
