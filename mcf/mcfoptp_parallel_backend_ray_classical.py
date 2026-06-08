"""
Created on Wed Mar 18 13:37:41 2026.

@author: MLechner

# -*- coding: utf-8 -*-

Contains functions of the classical ray implementation.

"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING
from psutil import virtual_memory

try:
    import ray
except ImportError:
    ray = None

from mcf.mcf_print_stats import print_mcf
from mcf.mcf_general_sys import memory_statistics

if TYPE_CHECKING:
    from mcf.mcf_init import GenCfg

type FloatLike = float | None


def check_ray_shutdown(mp_ray_shutdown: bool, mp_parallel: int) -> None:
    """Shut down ray if it is still running."""
    if ray is not None and mp_ray_shutdown and mp_parallel > 1 and ray.is_initialized():
        ray.shutdown()


def check_ray_shutdown_legacy(gen_cfg: 'GenCfg',
                              reference_duration: float,
                              duration: float,
                              no_of_workers: int, *,
                              max_multiplier: float | int = 3,
                              with_output: bool = True,
                              err_txt: str = ''
                              ) -> tuple[float, str]:
    """Shutdown ray with there is substantial increase in computation time."""
    if ray is None:
        raise RuntimeError('Ray has not been loaded.')
    if (no_of_workers == 1 or not ray.is_initialized()
            or duration < reference_duration * max_multiplier):

        return (reference_duration + duration) / 2, ''

    ray.shutdown()
    txt = (err_txt +
           f'\nRay shutdown because the time needed is {duration/reference_duration:.1%} of last '
           'time this part was running in ray. Maybe some workers do not work anymore. Ray will be '
           'restarted if needed.'
           )
    if with_output:
        print_mcf(gen_cfg, txt, summary=False)

    return reference_duration, txt


def init_ray_with_fallback(maxworkers: int,
                           gen_cfg: 'GenCfg', *,
                           mem_object_store: FloatLike = None,
                           mem_object_store_2: FloatLike = None,
                           ray_err_txt: str = ''
                           ) -> tuple[bool, int]:
    """Start ray in cases when this can be problematic."""
    if ray is None:
        raise RuntimeError('Ray has not been loaded.')
    while maxworkers >= 2:
        try:
            if mem_object_store is None:
                ray.init(num_cpus=maxworkers, include_dashboard=False, ignore_reinit_error=False,)
            else:
                ray.init(num_cpus=maxworkers, include_dashboard=False, ignore_reinit_error=False,
                         object_store_memory=mem_object_store,
                         )
            print_mcf(gen_cfg, f'\nRay started with {maxworkers} workers', summary=False)

            return True, maxworkers

        except (OSError, RuntimeError, ValueError):
            if mem_object_store_2 is not None:  # Check memory needed
                memory = virtual_memory()
                memory_needed = mem_object_store * 1.1
                if memory.free < memory_needed and gen_cfg.with_output and gen_cfg.verbose:
                    _, _, _, _, txt_memory = memory_statistics()
                    txt = ('\n' + ray_err_txt
                           + ' Potential lack of memory for object store. '
                           f'\nMemory needed: {round(memory_needed / (1024 * 1024), 2)} MB '
                           '\n' + txt_memory
                           )
                    print_mcf(gen_cfg, txt, summary=False)

            ray.shutdown()
            if maxworkers > 50:
                maxworkers = maxworkers // 2
            elif maxworkers > 10:
                maxworkers = round(maxworkers * 0.75)
            elif maxworkers > 5:
                maxworkers -= 2
            else:
                maxworkers -= 1
            if gen_cfg.with_output and gen_cfg.verbose:
                txt = '\n' + ray_err_txt + f' Number of workers reduced to {maxworkers}'
                print_mcf(gen_cfg, txt, summary=False)

    if gen_cfg.with_output and gen_cfg.verbose:
        txt = '\n' + ray_err_txt + 'RAY NOT USED. No multiprocessing. This will slow down execution'
        print_mcf(gen_cfg, txt, summary=False)

    return False, maxworkers


def ray_running_or_init(maxworkers: int,
                        gen_cfg: 'GenCfg', *,
                        mem_object_store: FloatLike = None,
                        mem_object_store_2: FloatLike = None,
                        ray_err_txt: str = ''
                        ) -> bool:
    """Check if Ray is running, and if not, try to start it."""
    if ray is None:
        raise RuntimeError('Ray has not been loaded.')
    if ray.is_initialized():
        ray_is_running = True
    else:
        ray_err_txt = 'Problems starting ray when filling forests.'
        ray_is_running, _ = init_ray_with_fallback(maxworkers,
                                                   gen_cfg,
                                                   mem_object_store=mem_object_store,
                                                   mem_object_store_2=mem_object_store_2,
                                                   ray_err_txt=ray_err_txt,
                                                   )
    return ray_is_running


def ray_put_all(*args: Any) -> ray.ObjectRef | tuple[ray.ObjectRef, ...]:
    """Apply ray.put() to each input.

    Returns
    -------
    ray.ObjectRef | tuple[ray.ObjectRef, ...]
        - 1 arg: single ObjectRef
        - 2+ args: tuple of ObjectRefs
    """
    refs = tuple(ray.put(arg) for arg in args)

    return refs[0] if len(refs) == 1 else refs


def ray_del_refs(*args: ray.ObjectRef | None,
                 f1: list[Any] | None = None,
                 f2: list[ray.ObjectRef] | None = None,
                 mp_ray_del: str | None = None,
                 ) -> tuple[object | None, ...]:
    """Replace selected Ray references / results by None and return them flattened.

    Parameters
    ----------
    *args
        Ray object references corresponding to the old 'refs' block.
        May be empty.
    f1
        Result of ray.get(f2), i.e. a list of materialized Python objects.
    f2
        First return of ray.wait(...), i.e. a list of ready ObjectRefs.
    mp_ray_del
        Control string:
        - if it contains 'refs', all *args are replaced by None
        - if it contains 'rest', f1 and f2 are replaced by None

    Returns
    -------
    tuple[object | None, ...]
        Flattened tuple:
        (*processed_args, processed_f1, processed_f2)
    """
    mp_ray_del_s = '' if mp_ray_del is None else mp_ray_del

    if 'refs' in mp_ray_del_s:
        args_out = (None,) * len(args)
    else:
        args_out = args

    if 'rest' in mp_ray_del_s:
        f1_out = None
        f2_out = None
    else:
        f1_out = f1
        f2_out = f2

    return args_out + (f1_out, f2_out,)


def print_object_store(gen_cfg: 'GenCfg', mem_object_store: FloatLike,) -> None:
    """Print the object store."""
    if gen_cfg.with_output and gen_cfg.verbose and mem_object_store is not None:
        num = round(mem_object_store / (1024 * 1024))
        print_mcf(gen_cfg, f'\nSize of Ray Object Store: {num} MB', summary=False)
