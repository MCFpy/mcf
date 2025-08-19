"""
Contains system and file related commands.

Created on Thu May 11 16:30:11 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from gc import collect
from itertools import chain
from math import ceil, floor
from pathlib import Path
from pickle import dump, load
from sys import getsizeof, stderr

from psutil import virtual_memory
import ray

from mcf import mcf_print_stats_functions as mcf_ps


def delete_file_if_exists(file_name):
    """Delete existing file."""
    if file_name.exists():
        Path.unlink(file_name)


def define_outpath(outpath, new_outpath=True):
    """Verify outpath and create new one if needed."""
    path_programme_run = Path.cwd()
    if outpath in (None, [], '') or not isinstance(outpath, (Path, str)):
        outpath = path_programme_run / 'output'
    elif isinstance(outpath, str):
        outpath = Path(outpath)
    if new_outpath:
        out_temp = outpath
        for i in range(1000):
            if out_temp.is_dir():
                print(f'Directory for output {out_temp} already exists',
                      'A new directory is created for the output.')
                out_temp = outpath.with_name(f'{outpath.name}{i}')
            else:
                try:
                    out_temp.mkdir(parents=True)
                except OSError as oserr:
                    raise OSError(f'Creation of the directory {out_temp}'
                                  ' failed') from oserr
                print(f'Successfully created the directory {out_temp}')
                if out_temp != outpath:
                    outpath = out_temp
                break
    else:
        if not outpath.is_dir():
            try:
                outpath.mkdir(parents=True)
            except OSError as oserr:
                raise OSError(
                    f'Creation of the directory {outpath} failed') from oserr
    return outpath


def get_fig_path(dic_to_update: dict,
                 outpath: Path,
                 add_name: str,
                 create_dir: bool,
                 no_csv: bool = False
                 ):
    """Define and create directories to store figures."""
    fig_pfad = outpath / ('plots_' + add_name)
    fig_pfad_jpeg = fig_pfad / 'jpeg'
    fig_pfad_csv = fig_pfad / 'csv'
    fig_pfad_pdf = fig_pfad / 'pdf'
    if create_dir:
        if not fig_pfad.is_dir():
            fig_pfad.mkdir(parents=True)
        if not fig_pfad_jpeg.is_dir():
            fig_pfad_jpeg.mkdir(parents=True)
        if not fig_pfad_csv.is_dir() and not no_csv:
            fig_pfad_csv.mkdir(parents=True)
        if not fig_pfad_pdf.is_dir():
            fig_pfad_pdf.mkdir(parents=True)
    dic_to_update[add_name + '_fig_pfad_jpeg'] = fig_pfad_jpeg
    dic_to_update[add_name + '_fig_pfad_csv'] = fig_pfad_csv
    dic_to_update[add_name + '_fig_pfad_pdf'] = fig_pfad_pdf

    return dic_to_update


def check_ray_shutdown(gen_dic, reference_duration, duration, no_of_workers,
                       max_multiplier=3, with_output=True, err_txt=''
                       ):
    """Shutdown ray with there is substantial increase in computation time."""
    if (no_of_workers == 1 or not ray.is_initialized()
            or duration < reference_duration * max_multiplier):
        return (reference_duration + duration) / 2, ''

    ray.shutdown()
    txt = (err_txt +
           '\nRay shutdown because the time needed is '
           f'{duration/reference_duration:.1%} of last time this part '
           'was running in ray. Maybe some workers do not work anymore. '
           'Ray will be restarted if needed.'
           )
    if with_output:
        mcf_ps.print_mcf(gen_dic, txt, summary=False)

    return reference_duration, txt


def find_no_of_workers(maxworkers, sys_share=0):
    """
    Find the optimal number of workers for MP such that system does not crash.

    Parameters
    ----------
    maxworkers : Int. Maximum number of workers allowed.

    Returns
    -------
    workers : Int. Workers used.
    sys_share: Float. System share.
    max_cores: Bool. Limit to number of physical(not logical cores)

    """
    # Currently this procedure does not make much sense as it only leaves the
    # numbers unchanges or sets them to 1.
    share_used = getattr(virtual_memory(), 'percent') / 100
    if sys_share >= share_used:
        sys_share = 0.9 * share_used
    sys_share = sys_share / 2
    workers = (1 - sys_share) / (share_used - sys_share)
    if workers > maxworkers:
        workers = maxworkers
    elif workers < 1.9:
        workers = 1
    else:  # TODO This seems to be a bug. 25.3.2025
        workers = maxworkers
    workers = floor(workers + 1e-15)
    return workers


def init_ray_with_fallback(maxworkers, int_dic, gen_dic, mem_object_store=None,
                           ray_err_txt=''):
    """Start ray in cases when this can be problematic."""
    while maxworkers >= 2:
        try:
            if mem_object_store is None:
                ray.init(num_cpus=maxworkers,
                         include_dashboard=False,
                         ignore_reinit_error=False,
                         )
            else:
                ray.init(
                    num_cpus=maxworkers,
                    include_dashboard=False,
                    ignore_reinit_error=False,
                    object_store_memory=mem_object_store,
                    )
            mcf_ps.print_mcf(gen_dic,
                             '\n'
                             + f'Ray started with {maxworkers} workers',
                             summary=False)

            return True, maxworkers

        except OSError:
            if int_dic['mem_object_store_2'] is not None:  # Check memory needed
                memory = virtual_memory()
                memory_needed = mem_object_store * 1.1
                if memory.free < memory_needed:
                    if int_dic['with_output'] and int_dic['verbose']:
                        _, _, _, _, txt_memory = memory_statistics()
                        txt = ('\n' + ray_err_txt
                               + ' Potential lack of memory for object store.'
                               + '\nMemory needed: '
                               + f'{round(memory_needed / (1024 * 1024), 2)} MB'
                               + '\n' + txt_memory
                               )
                        mcf_ps.print_mcf(gen_dic, txt, summary=False)

            ray.shutdown()
            if maxworkers > 50:
                maxworkers = maxworkers // 2
            elif maxworkers > 10:
                maxworkers = round(maxworkers * 0.75)
            elif maxworkers > 5:
                maxworkers -= 2
            else:
                maxworkers -= 1
            if int_dic['with_output'] and int_dic['verbose']:
                txt = ('\n' + ray_err_txt +
                       f' Number of workers reduced to {maxworkers}')
                mcf_ps.print_mcf(gen_dic, txt, summary=False)

    if int_dic['with_output'] and int_dic['verbose']:
        txt = ('\n' + ray_err_txt +
               'RAY NOT USED. No multiprocessing. This will slow down execution'
               )
        mcf_ps.print_mcf(gen_dic, txt, summary=False)

    return False, maxworkers


def no_of_boot_splits_fct(size_of_object_mb, workers):
    """
    Compute size of chunks for MP.

    Parameters
    ----------
    size_of_forest_MB : Float. Size of the object in MB.
    workers : Int. Number of workers in MP.

    Returns
    -------
    no_of_splits : Int. Number of splits.

    """
    basic_size_mb = 53
    total, available, used, free, _ = memory_statistics()

    if size_of_object_mb > basic_size_mb:
        multiplier = 1/8 * (14 / workers)
        chunck_size_mb = basic_size_mb * (1 + (available - 33000) / 33000
                                          * multiplier)
        chunck_size_mb = min(chunck_size_mb, 2000)
        chunck_size_mb = max(chunck_size_mb, 10)
        no_of_splits = ceil(size_of_object_mb / chunck_size_mb)
    else:
        no_of_splits = 1
        chunck_size_mb = size_of_object_mb

    txt = ('\nAutomatic determination of tree batches'
           f'\nSize of object:   {round(size_of_object_mb, 2):6} MB '
           f'\nNumber of workers {workers:2} No of splits: {no_of_splits:2}'
           '\nSize of chunk:  {round(chunck_size_mb, 2):6} MB '
           f'\nRAM total: {total:6} MB,  used: {used:6} MB, '
           f'available: {available:6} MB, free: {free:6} MB'
           )

    return no_of_splits, txt


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
    default_size = getsizeof(0)
    # estimate sizeof object without __sizeof__

    def sizeof(ooo):
        if id(ooo) in seen:       # do not double count the same object
            return 0
        seen.add(id(ooo))
        sss = getsizeof(ooo, default_size)

        if verbose:
            print(sss, type(ooo), repr(ooo), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(ooo, typ):
                sss += sum(map(sizeof, handler(ooo)))
                break
        return sss

    return sizeof(ooo)


def memory_statistics():
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
    memory = virtual_memory()
    total = round(memory.total / (1024 * 1024), 2)
    available = round(memory.available / (1024 * 1024), 2)
    used = round(memory.used / (1024 * 1024), 2)
    free = round(memory.free / (1024 * 1024), 2)
    txt = (f'\nRAM total: {total:6} MB,  used: {used:6} MB, '
           f'available: {available:6} MB, free: {free:6} MB')
    return total, available, used, free, txt


def print_mememory_statistics(gen_dic, location_txt):
    """Print memory statistics."""
    if gen_dic['with_output'] and gen_dic['verbose']:
        mcf_ps.print_mcf(gen_dic, '\n' + '-' * 100 + '\n'
                         + location_txt
                         + memory_statistics()[4]
                         + '\n',
                         summary=False)


def auto_garbage_collect(pct=80.0):
    """
    Call garbage collector if memory used > pct% of total available memory.

    This is called to deal with an issue in Ray not freeing up used memory.
    pct - Default value of 80%.  Amount of memory in use that triggers
          the garbage collection call.
    """
    if virtual_memory().percent >= pct:
        collect()


def print_size_weight_matrix(weights, weight_as_sparse, no_of_treat,
                             no_text=False):
    """
    Print size of weight matrix in MB.

    Parameters
    ----------
    weights : Sparse (CSR) or dense 2D Numpy array. Weight matrix.
    weight_as_sparse : Boolean.
    no_of_treat : Int. Number of treatments.

    Returns
    -------
    None.

    """
    total_bytes = total_size(weights)
    if weight_as_sparse:
        for d_idx in range(no_of_treat):
            total_bytes += (weights[d_idx].data.nbytes
                            + weights[d_idx].indices.nbytes
                            + weights[d_idx].indptr.nbytes)
    if no_text:
        return total_bytes
    return f'Size of weight matrix: {total_bytes / (1024 * 1024): .2f} MB'


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
            dump(object_to_save, file)
        object_to_load = None
        text = '\nObject saved to '
    else:
        with open(file_name, "rb") as file:
            object_to_load = load(file)
        text = '\nObject loaded from '
    if output:
        print(text + file_name)
    return object_to_load
