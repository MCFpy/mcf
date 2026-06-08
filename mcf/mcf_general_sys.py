"""
Contains system and file related commands.

Created on Thu May 11 16:30:11 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from itertools import chain
from math import ceil, floor
import os
from pathlib import Path
from pickle import dump, load
from shutil import rmtree
from stat import S_IWRITE
from sys import getsizeof, stderr
from typing import Any, TYPE_CHECKING

from numpy.typing import NDArray
from psutil import virtual_memory
from scipy.sparse import csr_matrix

from mcf import mcf_print_stats as mcf_ps

if TYPE_CHECKING:
    from mcf.mcf_init import GenCfg


def _make_writable_and_retry(func, path, _):
    # rmtree calls: onexc(func, path, exc_info)
    os.chmod(path, S_IWRITE)
    func(path)


def delete_path_or_file_if_exists(path: Path) -> None:
    """Delete file/symlink/dir if it exists (no-op if missing)."""
    p = Path(path)

    try:
        if p.is_dir() and not p.is_symlink():
            rmtree(p, onexc=_make_writable_and_retry)
        else:
            # file or symlink (incl. broken) -> unlink
            p.unlink()
    except FileNotFoundError:
        pass


def delete_file_if_exists(file_name: Path) -> None:
    """Delete existing file."""
    if file_name.exists():
        Path.unlink(file_name)


def define_outpath(outpath: Path | str | None, new_outpath: bool = True,) -> Path:
    """Verify outpath and create new one if needed."""
    path_programme_run = Path.cwd()
    match outpath:
        case str() as s if s:        # non-empty string → Path
            outpath = Path(s)
        case Path():                 # already a Path → keep
            pass
        case _:                      # None, '', [], or any other type → default
            outpath = path_programme_run / 'output'

    if new_outpath:
        out_temp = outpath
        for i in range(1000):
            if out_temp.is_dir():
                if any(out_temp.iterdir()):
                    print(f'Non-empty directory for output {out_temp} already exists. '
                          'A new directory is created for the output.'
                          )
                    out_temp = outpath.with_name(f'{outpath.name}{i}')
                else:
                    print(f'An empty directory for output {out_temp} already exists. ')
                    if out_temp != outpath:
                        outpath = out_temp
                    break
            else:
                try:
                    out_temp.mkdir(parents=True)
                except OSError as oserr:
                    raise OSError(f'Creation of the directory {out_temp} failed') from oserr
                print(f'Successfully created the directory {out_temp}')
                if out_temp != outpath:
                    outpath = out_temp
                break
    else:
        if not outpath.is_dir():
            try:
                outpath.mkdir(parents=True)
            except OSError as oserr:
                raise OSError(f'Creation of the directory {outpath} failed') from oserr

    return outpath


def get_fig_path(dic_to_update: dict,
                 outpath: Path,
                 add_name: str,
                 create_dir: bool,
                 no_csv: bool = False
                 ) -> dict:
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


def find_no_of_workers(maxworkers: int,
                       sys_share: float = 0,
                       zero_tol: float = 1e-10,
                       ) -> int:
    """
    Find the optimal number of workers for MP such that system does not crash.

    Parameters
    ----------
    maxworkers : Int. Maximum number of workers allowed.
    sys_share: Float. System share. Default is 0.

    Returns
    -------
    workers : Int. Workers used.
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
    else:
        workers = maxworkers
    workers = floor(workers + zero_tol)

    return workers


def no_of_boot_splits_fct(size_of_object_mb: int | float, workers: int) -> int:
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


def total_size(ooo: Any,
               handlers: Any = None,
               verbose: bool = False
               ) -> float | int:
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


def memory_statistics() -> tuple[int, int, int, int, str]:
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


def print_mememory_statistics(gen_cfg: 'GenCfg', location_txt: str) -> None:
    """Print memory statistics."""
    mcf_ps.print_mcf(gen_cfg, '\n'
                              + location_txt
                              + memory_statistics()[4]
                              + '\n',
                     summary=False
                     )


def print_size_weight_matrix(weights: csr_matrix | NDArray[Any],
                             weight_as_sparse: bool,
                             no_of_treat: int,
                             no_text: bool = False
                             ) -> None:
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


def save_load(file_name: Path | str,
              object_to_save: Any = None,
              save: bool = True,
              output: bool = True) -> Any:
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
        print(text + str(file_name))

    return object_to_load
