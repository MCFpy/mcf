"""General purpose procedures.

# -*- coding: utf-8 -*-.
Created on Thu Apr  2 17:55:24 2020

@author: MLechner
"""
import copy
import math
import itertools
from datetime import datetime, timedelta
import sys
import os.path
import scipy.stats as sct
from sympy.ntheory import primefactors
import pandas as pd
import numpy as np
from numba import njit


def check_if_not_number(data_df, variable):
    """
    Check if the pandas dataframe is not a number.

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
        print(var_not_a_number, 'do not contain numbers.')
        raise Exception('Number format is needed for this variable.')


def delete_file_if_exists(file_name):
    """Delete existing file.

    This function also exists in general_purpose_system_files.
    """
    if os.path.exists(file_name):
        os.remove(file_name)


# define an Output class for simultaneous console - file output
class OutputTerminalFile():
    """Output class for simultaneous console/file output."""

    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.output = open(file_name, "w")

    def write(self, message):
        """Write both into terminal and file."""
        self.terminal.write(message)
        self.output.write(message)

    def flush(self):
        """Python 3 compatibility."""


def print_dic(dic):
    """Print dictionary in a simple way."""
    for keys, values in dic.items():
        if isinstance(values, list):
            print(keys, ':  ', *values)
        else:
            print(keys, ':  ', values)


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
            if (var_to_check.upper() == name) or (
                    var_to_check.lower() == name):
                var_to_check = name
                break
    return var_to_check


@njit
def quadratic_form(vec, mat):
    """Quadratic form for Numpy: vec'mat*vec.

    Parameters
    ----------
    vec : 1D Numpy Array.
    mat : 2D quadratic Numpy Array.

    Returns
    -------
    Numpy Float. The Quadratic form.

    """
    return np.dot(vec, mat @ vec)


def waldtest(diff, variance):
    """Compute Wald Chi2 statistics.

    Parameters
    ----------
    diff : 1D Numy array.
    variance : 2D Numpy array.

    Returns
    -------
    stat : Numpy float. Test statistic.
    df : Int. Degrees of freedom.
    p : Numpy float. p-value.

    """
    degfr = int(round(np.linalg.matrix_rank(variance)))
    stat = -1
    pval = -1
    if degfr == len(variance):
        if np.all(np.linalg.eigvals(variance) > 1e-15):
            weight = np.linalg.inv(variance)
            stat = quadratic_form(diff, weight)
            pval = sct.chi2.sf(stat, degfr)
    return stat, degfr, pval


def add_dummies(cat_vars, indatafile):
    """Add dummy variables to csv data file."""
    dat_df = pd.read_csv(indatafile)
    dat_df.columns = [s.upper() for s in dat_df.columns]
    x_dummies = pd.get_dummies(dat_df[cat_vars], columns=cat_vars)
    new_x_name = x_dummies.columns.tolist()
    dat_df = pd.concat([dat_df, x_dummies], axis=1)
    delete_file_if_exists(indatafile)
    dat_df.to_csv(indatafile, index=False)
    return new_x_name


def print_effect(est, stderr, t_val, p_val, effect_list, add_title=None,
                 add_info=None):
    """Print treatment effects.

    Parameters
    ----------
    est : Numpy array. Point estimate.
    stderr : Numpy array. Standard error.
    t_val : Numpy array. t/z-value.
    p_val : Numpy array. p-value.
    effect_list : List of Int. Treatment values involved in comparison.
    add_title: None or string. Additional title.
    add_info: None or Int. Additional information about parameter.

    Returns
    -------
    None.
    """
    if add_title is None:
        print('Comparison    Estimate   Standard error t-value   p-value')
    else:
        print('Comparison ', add_title,
              '   Estimate   Standard error t-value   p-value')
    print('- ' * 40)
    for j in range(np.size(est)):
        print('{:<3} vs {:>3}'.format(effect_list[j][0], effect_list[j][1]),
              end=' ')
        if add_title is not None:
            print('{:6.2f}'.format(add_info), end=' ')
        print('{:12.6f}  {:12.6f}'.format(est[j], stderr[j]), end=' ')
        print('{:8.2f}  {:8.3f}%'.format(t_val[j], p_val[j]*100), end=' ')
        if p_val[j] < 0.001:
            print('****')
        elif p_val[j] < 0.01:
            print(' ***')
        elif p_val[j] < 0.05:
            print('  **')
        elif p_val[j] < 0.1:
            print('   *')
        else:
            print()
    print('- ' * 40)


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
    small = math.log(small)
    large = math.log(large)
    sequence = np.unique(np.round(np.exp(np.linspace(small, large, number))))
    sequence_p = sequence.tolist()
    return sequence_p


def share_completed(current, total):
    """Counter for how much of a task is completed.

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
        print('{:4.0f}'.format(share), end=" ", flush=True)
    else:
        points_to_print = range(1, total, round(total/20))
        if current in points_to_print:
            print('{:4.0f}'.format(share), end=" ", flush=True)
    if current == total:
        print('Task completed')


def statistics_covariates(indatei, dict_var_type):
    """Analyse covariates wrt to variable type.

    Parameters
    ----------
    indatei : string. File with data.
    dict_var_type : dictionary with covariates and their type.

    Returns
    -------
    None.

    """
    data = pd.read_csv(indatei)
    print('\nCovariates that enter forest estimation', '(categorical',
          'variables are recoded with primes)')
    print('-' * 80)
    print('Name of variable             ', 'Type of variable         ',
          '# unique values', '  Min     ', 'Mean      ', 'Max       ')
    for keys in dict_var_type.keys():
        x_name = keys
        print('{:30}'.format(keys), end='')
        if dict_var_type[keys] == 0:
            print('ordered                     ', end='')
        elif dict_var_type[keys] == 1:
            print('unordered categorical short ', end='')
        elif dict_var_type[keys] == 2:
            print('unordered categorical long  ', end='')
        print('{:<12}'.format(len(data[x_name].unique())), end='')
        print('{:10.5f}'.format(data[x_name].min()), end='')
        print('{:10.5f}'.format(data[x_name].mean()), end='')
        print('{:10.5f}'.format(data[x_name].max()))


def substitute_variable_name(v_dict, old_name, new_name):
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
    vn_dict = copy.deepcopy(v_dict)
    for i in v_dict:
        list_of_this_dic = v_dict[i]
        if list_of_this_dic is not None:
            for j, _ in enumerate(list_of_this_dic):
                if list_of_this_dic[j] == old_name:
                    list_of_this_dic[j] = new_name
            vn_dict[i] = list_of_this_dic
    return vn_dict


def sample_split_2_3(indatei, outdatei1, share1, outdatei2, share2,
                     outdatei3=None, share3=0, random_seed=None,
                     with_output=True):
    """Split sample in 2 or 3 random subsamples (depending on share3 > 0).

    Parameters
    ----------
    indatei : String. File with data to split
    outdatei*: String. Files with splitted data
    random_seed: Fixes seed of random number generator
    with_output: Some infos about split.

    Returns
    -------
    outdatei1,outdatei2,outdatei3: names of files with splitted data

    """
    if ((share1+share2+share3) > 1.01) or ((share1 + share2 + share3) < 0.99):
        print('Sample splitting: Shares do not add up to 1')
        sys.exit()
    data = pd.read_csv(filepath_or_buffer=indatei, header=0)
    # split into 2 or 3 dataframes
    data1 = data.sample(frac=share1, random_state=random_seed)
    if share3 == 0:
        data2 = data.drop(data1.index)
    else:
        tmp = data.drop(data1.index)
        data2 = tmp.sample(frac=share2/(1-share1), random_state=random_seed)
        data3 = tmp.drop(data2.index)
    delete_file_if_exists(outdatei1)
    delete_file_if_exists(outdatei2)
    data1.to_csv(outdatei1, index=False)
    data2.to_csv(outdatei2, index=False)
    if share3 != 0:
        delete_file_if_exists(outdatei3)
        data3.to_csv(outdatei3, index=False)
    if with_output:
        print('\nRandom sample splitting')
        print('Number of obs. in org. data', indatei,
              '{0:>5}'.format(data.shape[0]))
        print('Number of obs. in', outdatei1,
              '{0:>5}'.format(data1.shape[0]))
        print('Number of obs. in', outdatei2,
              '{0:>5}'.format(data2.shape[0]))
        if share3 != 0:
            print('Number of observations in', outdatei3,
                  '{0:>5}'.format(data3.shape[0]))
    return outdatei1, outdatei2, outdatei3


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
    v_inter = set(var_in).intersection(set(var_weg))
    if not v_inter == set():
        ohne_var_weg = list(set(var_in)-v_inter)
    else:
        ohne_var_weg = copy.deepcopy(var_in)
    return ohne_var_weg


def screen_variables(indatei, var_names, perfectcorr, min_dummy, with_output):
    """Screen variables to decide whether they could be deleted.

    Parameters
    ----------
    indatei : string
    var_names : list of strings
    perfectcorr : Boolean
        Check for too high correlation among variables
    min_dummy : integer
        Check for minimum number of observation on both side of dummies
    with_output : string

    Returns
    -------
    variables_remain : List of strings
    variables_delete : list of strings

    """
    data = pd.read_csv(filepath_or_buffer=indatei, header=0)
    data = data[var_names]
    check_if_not_number(data, var_names)
    k = data.shape[1]
    all_variable = set(data.columns)
    # Remove variables without any variation
    to_keep = data.std(axis=0) > 1e-10
    datanew = data.loc[:, to_keep].copy()  # Keeps the respective columns
    if with_output:
        print('\n')
        print('-' * 80)
        print('Control variables checked')
        kk1 = datanew.shape[1]
        all_variable1 = set(datanew.columns)
        if kk1 != k:
            deleted_vars = list(all_variable - all_variable1)
            print('Variables without variation: ', deleted_vars)
    if perfectcorr:
        corr_matrix = datanew.corr().abs()
        # Upper triangle of corr matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                          k=1).astype(np.bool))
        to_delete = [c for c in upper.columns if any(upper[c] > 0.999)]
        if not to_delete == []:
            datanew = datanew.drop(columns=to_delete)
    if with_output:
        kk2 = datanew.shape[1]
        all_variable2 = set(datanew.columns)
        if kk2 != kk1:
            deleted_vars = list(all_variable1-all_variable2)
            print('Correlation with other variable > 99.9%: ', deleted_vars)
    if min_dummy > 1:
        shape = datanew.shape
        to_keep = []
        for cols in range(shape[1]):
            value_count = datanew.iloc[:, cols].value_counts()
            if len(value_count) == 2:  # Dummy variable
                if value_count.min() < min_dummy:
                    to_keep = to_keep + ([False])
                else:
                    to_keep = to_keep + ([True])
            else:
                to_keep = to_keep + ([True])
        datanew = datanew.loc[:, to_keep]
    if with_output:
        kk3 = datanew.shape[1]
        all_variable3 = set(datanew.columns)
        if kk3 != kk2:
            deleted_vars = list(all_variable2-all_variable3)
            print('Dummy variables with too few 0 or 1: ', deleted_vars)
    kk2 = datanew.shape[1]
    variables_remain = datanew.columns
    variables_delete = list(all_variable-set(datanew.columns))
    if with_output:
        if kk2 == k:
            print('All control variables have been retained')
        else:
            print('\nThe following variables have been deleted:',
                  variables_delete)
            desc_stat = data[variables_delete].describe()
            to_print = desc_stat.transpose()
            pd.set_option('display.max_rows', len(desc_stat.columns),
                          'display.max_columns', len(to_print.columns))
            print(to_print)
            pd.reset_option('display.max_rows', 'display.max_columns')
            print('\nThe following variables have been retained:',
                  variables_remain)
    return variables_remain, variables_delete


def clean_reduce_data(infile, outfile, names_to_inc, with_output, desc_stat,
                      print_to_file, add_missing_vars=False):
    """Remove obs. with missings and variables not included in 'names_to_inc'.

    Parameters     (does not check whether all variables exist)
    ----------
    infile : STRING.        CSV-file with input data.
    outfile : STRING.        CSV-file with output data.
    names_to_inc : list of strings.        Variables to be kept in data.
    with_output : Boolean.        Give some information
    desc_stat : Boolean.        Show descriptive stats of new file.
    add_missing_vars: Boolean. Add missing variables with values 0. Default is
                               False.

    Returns
    -------
    outfile : string
        Output file.
    """
    data = pd.read_csv(filepath_or_buffer=infile)
    shape = data.shape
    data.columns = [s.upper() for s in data.columns]
    if add_missing_vars:
        add_df = add_missing_vars_fct(data.columns, names_to_inc,
                                      len(data), with_output)
        if add_df is not None:
            data = pd.concat([data, add_df], axis=1)
    datanew = data[names_to_inc].copy()
    datanew.dropna(inplace=True)
    shapenew = datanew.shape
    delete_file_if_exists(outfile)
    datanew.to_csv(outfile, index=False)
    if with_output:
        print("\nCheck for missing and unnecessary variables.")
        if shape == shapenew:
            print('File has not been changed')
        else:
            if shapenew[0] == shape[0]:
                print('  No observations deleted')
            else:
                print('  {0:<5}'.format(shape[0]-shapenew[0]),
                      'observations deleted')
            if shapenew[1] == shape[1]:
                print('No variables deleted')
            else:
                print('  {0:<4}'.format(shape[1]-shapenew[1]),
                      'variables deleted:')
                liste = list(data.columns.difference(datanew.columns))
                print(*liste)
        if desc_stat:
            print_descriptive_stats_file(outfile, 'all', print_to_file)
    return outfile


def add_missing_vars_fct(names_in_data, names_to_inc, obs, with_output):
    """Include missing variables as zeros."""
    missing_vars = list(set(names_to_inc).difference(names_in_data))
    if missing_vars:
        if with_output:
            print('=' * 80)
            print('The following variables will be addded with zero values: ',
                  missing_vars)
        add_df = pd.DataFrame(0, index=np.arange(obs), columns=missing_vars)
        return add_df
    return None


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
    """Return a list of strings with unique entries.

    Parameters
    ----------
    names1 to 10: lists of strings
        Names to merge.

    Returns
    -------
    new_names : list of strings
        All unique names in one list.

    """
    if names2 is None:
        names2 = []
    if names3 is None:
        names3 = []
    if names4 is None:
        names4 = []
    if names5 is None:
        names5 = []
    if names6 is None:
        names6 = []
    if names7 is None:
        names7 = []
    if names8 is None:
        names8 = []
    if names9 is None:
        names9 = []
    if names10 is None:
        names10 = []
    new_names = copy.deepcopy(names1)
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


def print_descriptive_stats_file(indata, varnames='all', to_file=False,
                                 df_instead_of_file=False):
    """Print descriptive statistics of a dataset on file.

    Parameters
    ----------
    indata : string, Name of input file
    varnames : List of strings, Variable names of selected variables
              The default is 'all'.

    Returns
    -------
    None.
    """
    if df_instead_of_file:
        data = indata.copy()
    else:
        data = pd.read_csv(filepath_or_buffer=indata, header=0)
    if varnames != 'all':
        data_sel = data[varnames]
    else:
        data_sel = data
    desc_stat = data_sel.describe()
    if (varnames == 'all') or len(varnames) > 10:
        to_print = desc_stat.transpose()
        rows = len(desc_stat.columns)
        cols = len(desc_stat.index)
    else:
        to_print = desc_stat
        rows = len(desc_stat.index)
        if isinstance(desc_stat, pd.DataFrame):
            cols = len(desc_stat.columns)
        else:
            cols = 1
    if not df_instead_of_file:
        print('\nData set:', indata)
    if to_file:
        expand = False
    else:
        expand = True
    with pd.option_context('display.max_rows', rows,
                           'display.max_columns', cols+1,
                           'display.expand_frame_repr', expand,
                           'chop_threshold', 1e-13):
        print(to_print)


def check_all_vars_in_data(indata, variables):
    """Check whether all variables are contained in data set.

    Parameters
    ----------
    indata : string. Input data.
    variables : list of strings. Variables to check indata for.

    Returns
    -------
    None. Programme terminates if some variables are not found

    """
    headers = pd.read_csv(filepath_or_buffer=indata, nrows=0)
    header_list = [s.upper() for s in list(headers.columns.values)]
    all_available = all(i in header_list for i in variables)
    if not all_available:
        missing_variables = [i for i in variables if i not in header_list]
        print('\nVariables not in ', indata, ':', missing_variables)
        sys.exit()
    else:
        print("\nAll variables found in ", indata)


def print_timing(text, time_diff):
    """Show date and duration of a programme and its different parts.

    Parameters
    ----------
    text : list of strings
        Explanation for the particular time difference
    time_diff :time differnce in time2-time1 [time.time() format]
        Time difference

    """
    print('\n\n')
    print('-' * 80)
    print("Programme executed at: ", datetime.now())
    print('-' * 80)
    for i in range(0, len(text), 1):
        print(text[i], timedelta(seconds=time_diff[i]))
    print('-' * 80, '\n')


def randomsample(datapath, indatafile, outdatafile, fraction,
                 replacement=False):
    """Generate a random sample of the data in a file.

    Parameters
    ----------
    datapath : string.        location of files.
    indatafile : string.        input file as csv file.
    outdatafile : string.        output file as csv file.
    fraction : float.  share of sample to be randomly put into output file.
    replacement : boolean, optinal.
        True: Sampling with replacement ; False: without rplm.

    """
    indatafile = datapath + '/' + indatafile
    outdatafile = datapath + '/' + outdatafile
    data = pd.read_csv(filepath_or_buffer=indatafile, header=0)
    data = data.sample(frac=fraction, replace=replacement)
    delete_file_if_exists(outdatafile)
    data.to_csv(outdatafile, index=False)


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
    if start_with_1:
        add = 1
    else:
        add = 0
    primes = primes_list(1000)
    position = []
    for val in x_values:
        position.append(primes.index(val)+add)
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
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
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
              7883, 7901, 7907, 7919]
    return primes[0:number]


def pause():
    """Pause."""
    inp = input("Press the <ENTER> key to continue...")
    return inp


def total_sample_splits_categorical(no_of_values):
    """
    Compute total number of sample splits that can generated by categoricals.

    Parameters
    ----------
    no_of_values : Int.

    Returns
    -------
    no_of_splits: Int.

    """
    no_of_splits = 0
    for i in range(1, no_of_values):
        no_of_splits += math.factorial(no_of_values) / (
            math.factorial(no_of_values-i) * math.factorial(i))
    return no_of_splits/2  # no complements


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
    key_list = []
    value_list = []
    for keys in dic.keys():
        key_list += [keys]
        value_list += [dic[keys]]
    return key_list, value_list


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
    if key_list == []:
        print('Retrieving items from list was not succesful')
        sys.exit()
    return key_list


def list_product(factors):
    """Prodcuce a product of a list keeping python data format.

    Parameters
    ----------
    factors : List.

    Returns
    -------
    prod : INT or Float.

    """
    return math.prod(factors)  # should be faster and keep python format


def all_combinations_no_complements(values):
    """Create all possible combinations of list elements, removing complements.

    Parameters
    ----------
    values : List. Elements to be combined.

    Returns
    -------
    list_without_complements : List of tuples.

    """
    list_all = []
    # This returns a list with tuples of all possible combinations of tuples
    for length in range(1, len(values)):
        list_all.extend(list(itertools.combinations(values, length)))
    # Next, the complements to each list will be removed
    list_wo_compl, _ = drop_complements(list_all, values)
    return list_wo_compl


def drop_complements(list_all, values):
    """
    Identify and remove complements.

    Parameters
    ----------
    list_all : List of tuples. Tuples with combinations.
    values : List. All relevant values.

    Returns
    -------
    list_wo_compl : List of Tuples. List_all with complements removed.

    """
    list_w_compl = []
    list_wo_compl = []
    for i in list_all:
        if i not in list_w_compl:
            list_wo_compl.append(i)
            compl_of_i = values[:]
            for drop_i in i:
                compl_of_i.remove(drop_i)
            list_w_compl.append(tuple(compl_of_i))
    return list_wo_compl, list_w_compl


def resort_list(liste, idx_list, n_x):
    """
    Make sure list is in the same order as some index_variable.

    Parameters
    ----------
    liste : List of lists.
    idx_list : List with indices as finished by multiprocessing.
    n_x : Int. Number of observations.

    Returns
    -------
    weights : As above, but sorted.

    """
    check_idx = list(range(n_x))
    if check_idx != idx_list:
        liste = [liste[i] for i in idx_list]
    return liste
