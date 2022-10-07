"""General purpose procedures.

# -*- coding: utf-8 -*-.
Created on Thu Apr  2 17:55:24 2020

@author: MLechner
"""
from concurrent import futures
import pickle
import importlib.util
import os.path
from shutil import rmtree
import gc
import sys
from itertools import chain

import pandas as pd
import psutil


def delete_file_if_exists(file_name):
    """Delete existing file.

    This function also exists in general_purpose.
    """
    if os.path.exists(file_name):
        os.remove(file_name)


def delete_dir(dir_path):
    """Delete directory."""
    if os.path.isdir(dir_path):
        for file in os.listdir(dir_path):
            remove_files_and_subdir(dir_path, file)
        try:
            os.rmdir(dir_path)
        except OSError:
            print("Removal of the temorary directory %s failed" % dir_path)
    else:
        raise Exception('Directory %s to remove not found.')


def remove_files_and_subdir(path, file_or_dir):
    """Remove files or subdirectories."""
    if os.path.isdir(os.path.join(path, file_or_dir)):
        rmtree(os.path.join(path, file_or_dir))  # del dir+all subdirs
    else:
        os.remove(os.path.join(path, file_or_dir))


def create_dir(dir_path):
    """Create directory."""
    if os.path.isdir(dir_path):
        file_list = os.listdir(dir_path)
        if file_list:
            for file in file_list:
                remove_files_and_subdir(dir_path, file)
    else:
        try:
            os.mkdir(dir_path)
        except OSError as oserr:
            raise Exception("Creation of the directory %s failed" % dir_path
                            ) from oserr
    return dir_path


def auto_garbage_collect(pct=80.0):
    """
    Call garbage collector if memory used > pct% of total available memory.

    This is called to deal with an issue in Ray not freeing up used memory.
    pct - Default value of 80%.  Amount of memory in use that triggers
          the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()


def save_load(file_name, object_to_save=None, save=True, output=True):
    """
    Save and load objects via pickle.

    Parameters
    ----------
    file_name : String. File to save to or to load from.
    object_to_save : any python object that can be pickled, optional.
                     The default is None.
    save : Boolean., optional The default is True. False for loading.

    Returns
    -------
    object_to_load : Unpickeled Python object (if save=False).
    """
    if save:
        delete_file_if_exists(file_name)
        with open(file_name, "wb+") as file:
            pickle.dump(object_to_save, file)
        object_to_load = None
        text = '\nObject saved to '
    else:
        with open(file_name, "rb") as file:
            object_to_load = pickle.load(file)
        text = '\nObject loaded from '
    if output:
        print(text + file_name)
    return object_to_load


def csv_var_names_upper(path, csvfile):
    """
    Convert all variable names in csv-file to upper case.

    Parameters
    ----------
    path : Str. Directory of location of csv_file.
    csvfile: Str.

    Returns
    -------
    csvfile_new : Str.

    """
    indatafile = path + '/' + csvfile + '.csv'
    data_df = pd.read_csv(indatafile)
    names = data_df.columns.tolist()
    any_change = False
    rename_dic = {}
    for name in names:
        if name != name.upper():
            any_change = True
            name_new = name.upper()
        else:
            name_new = name
        rename_dic.update({name: name_new})
    if any_change:
        csvfile_new = csvfile + 'UP'
        outdatafile = path + '/' + csvfile_new + '.csv'
        data_df.rename(rename_dic)   # pylint: disable=E1101
        delete_file_if_exists(outdatafile)
        data_df.to_csv(outdatafile, index=False)   # pylint: disable=E1101
    else:
        csvfile_new = csvfile
    return csvfile_new


def remove_temporary_directory(pfad, verbose=True):
    """Remove temporary dir."""
    if os.path.isdir(pfad):
        for temp_file in os.listdir(pfad):
            os.remove(os.path.join(pfad, temp_file))
        try:
            os.rmdir(pfad)
        except OSError:
            if verbose:
                print(f'Removal of the temorary directory {pfad:s} failed')
        else:
            if verbose:
                print(f'Successfully removed the directory {pfad:s}')
    else:
        if verbose:
            print('Temporary directory does not exist.')


def create_temp_directory(pfad, verbose=True):
    """Create temporary directory if it does not exist."""
    if os.path.isdir(pfad):
        file_list = os.listdir(pfad)
        if file_list:
            for temp_file in file_list:
                os.remove(os.path.join(pfad, temp_file))
        if verbose:
            print('Temporary directory  %s exists' % pfad)
            if file_list:
                print('All files in %s deleted.' % pfad)
    else:
        try:
            os.mkdir(pfad)
        except OSError as oserr:
            raise Exception("Creation of the directory %s failed" % pfad
                            ) from oserr
        else:
            if verbose:
                print("Successfully created the directory %s" % pfad)


def load_module_path(name_modul, path_modul):
    """Load modules with given path and name.

    Parameters
    ----------
    name_modul : string. Name of module to be used
    path_modul : string. Full name of file that contains modul

    Returns
    -------
    modul : name space of modul

    """
    spec = importlib.util.spec_from_file_location(name_modul, path_modul)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def clean_futures():
    """Clean up workers in memory."""
    workers = psutil.cpu_count()
    with futures.ProcessPoolExecutor() as fpp:
        ret_fut = {fpp.submit(do_almost_nothing, i): i for i in range(workers)}
    del ret_fut


def do_almost_nothing(value):
    """Do almost nothing."""
    value += 1
    return value


def create_output_directory(pfad, verbose=True):
    """Create directory for output if it does not exist."""
    if os.path.isdir(pfad):
        if verbose:
            print("Directory for output %s already exists" % pfad)
    else:
        try:
            os.mkdir(pfad)
        except OSError as oserr:
            raise Exception("Creation of the directory %s failed" % pfad
                            ) from oserr
        else:
            if verbose:
                print("Successfully created the directory %s" % pfad)


def copy_csv_file_pandas(new_file, old_file):
    """Copy csv file with pandas."""
    dat_df = pd.read_csv(old_file)
    delete_file_if_exists(new_file)
    dat_df.to_csv(new_file, index=False)


def memory_statistics(with_output=True):
    """
    Give memory statistics.

    Parameters
    ----------
    with_output : Boolean. Print output. The default is True.

    Returns
    -------
    total : Float. Total memory in GB.
    available : Float. Available memory in GB.
    used : Float. Used memory in GB.
    free : Float. Free memory in GB.

    """
    memory = psutil.virtual_memory()
    total = round(memory.total / (1024 * 1024), 2)
    available = round(memory.available / (1024 * 1024), 2)
    used = round(memory.used / (1024 * 1024), 2)
    free = round(memory.free / (1024 * 1024), 2)
    if with_output:
        print(f'RAM total: {total:6} MB,  used: {used:6} MB, ',
              f'available: {available:6} MB, free: {free:6} MB')
    return total, available, used, free


def total_size(ooo, handlers=None, verbose=False):
    """Return the approximate memory footprint an object & all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, (deque), dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
        https://code.activestate.com/recipes/577504/

    """
    #  dict_handler = lambda d: chain.from_iterable(d.items())
    if handlers is None:
        handlers = {}

    def dict_handler(ddd):
        return chain.from_iterable(ddd.items())

    all_handlers = {tuple: iter,
                    list: iter,
                    # deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter}
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()               # track which object id's have already been seen
    default_size = sys.getsizeof(0)
    # estimate sizeof object without __sizeof__

    def sizeof(ooo):
        if id(ooo) in seen:       # do not double count the same object
            return 0
        seen.add(id(ooo))
        sss = sys.getsizeof(ooo, default_size)

        if verbose:
            print(sss, type(ooo), repr(ooo), file=sys.stderr)

        for typ, handler in all_handlers.items():
            if isinstance(ooo, typ):
                sss += sum(map(sizeof, handler(ooo)))
                break
        return sss

    return sizeof(ooo)
