"""
Created on Mon Mar 23 15:15:44 2026.

@author: MLechner

# -*- coding: utf-8 -*-

Executors for forest building.


"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

try:
    import ray
except ImportError:  # pragma: no cover
    ray = None

from mcf.mcfoptp_parallel_backends_base import (ParallelExecutor,
                                                RayExecutor,
                                                SequentialExecutor,
                                                TaskSpec,
                                                batched_tasks,
                                                batch_size_fct,
                                                )
from mcf.mcfoptp_parallel_backends_memmap import make_memmap_joblib_executor, MemmapJoblibCfg

R = TypeVar('R')


def _unique_dicts(dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep order, remove duplicates."""
    unique: list[dict[str, Any]] = []

    for candidate in dicts:
        if not any(candidate == existing for existing in unique):
            unique.append(candidate)

    return unique


def init_ray_for_forest_with_fallback(*, maxworkers: int,
                                      mem_object_store_1: int | None = None,
                                      mem_object_store_2: int | None = None,
                                      ray_err_txt: str = 'Ray did not start in forest building.',
                                      ray_init_kwargs: dict[str, Any] | None = None,
                                      ) -> int | None:
    """
    Initialize Ray with object-store fallback.

    Attempt order:
    - if mem_object_store_1 is not None:
        1) object_store_memory=mem_object_store_1
        2) object_store_memory=mem_object_store_2   (if given and different)
        3) default Ray init without explicit object store
    - if mem_object_store_1 is None:
        1) default Ray init without explicit object store
        2) object_store_memory=mem_object_store_2   (if given)

    Returns
    -------
    int | None
        The object-store size that succeeded, or None if default init succeeded.
    """
    if ray is None:  # pragma: no cover
        raise ImportError('ray is required for mp_backend="ray"')

    if ray.is_initialized():
        return None

    base_kwargs = {} if ray_init_kwargs is None else dict(ray_init_kwargs)
    base_kwargs.setdefault('num_cpus', maxworkers)

    attempts: list[dict[str, Any]] = []

    if mem_object_store_1 is not None:
        attempt = dict(base_kwargs)
        attempt['object_store_memory'] = mem_object_store_1
        attempts.append(attempt)

        if mem_object_store_2 is not None and mem_object_store_2 != mem_object_store_1:
            attempt = dict(base_kwargs)
            attempt['object_store_memory'] = mem_object_store_2
            attempts.append(attempt)

        attempts.append(dict(base_kwargs))

    else:
        attempts.append(dict(base_kwargs))

        if mem_object_store_2 is not None:
            attempt = dict(base_kwargs)
            attempt['object_store_memory'] = mem_object_store_2
            attempts.append(attempt)

    attempts = _unique_dicts(attempts)

    last_exc: Exception | None = None

    for init_kwargs in attempts:
        try:
            ray.init(**init_kwargs)
            return init_kwargs.get('object_store_memory')
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
            last_exc = exc

    raise RuntimeError(ray_err_txt) from last_exc


def make_forest_executor(*, int_cfg: Any,
                         maxworkers: int,
                         memmap_min_bytes: int = 4 * 1024 * 1024,
                         pickle_min_bytes: int = 128 * 1024 * 1024,
                         joblib_backend_large: str = 'loky',
                         temp_dir: str | Path | None = None,
                         cleanup_on_shutdown: bool = True,
                         ray_err_txt: str = 'Ray did not start in forest building.',
                         ray_init_kwargs: dict[str, Any] | None = None,
                         ray_remote_kwargs: dict[str, Any] | None = None,
                         ) -> ParallelExecutor:
    """
    Create executor for forest building.

    Parameters
    ----------
    int_cfg:
        Internal configuration object. Expected to contain at least:
        - mp_backend
        - mem_object_store_1
        - mem_object_store_2

    maxworkers:
        Number of workers / CPUs.

    memmap_min_bytes:
        Threshold for switching from normal joblib to memmap-aware joblib.


    joblib_backend_large:
        Joblib backend for larger shared data. Usually 'loky'.
    """
    match int_cfg.mp_backend:
        case 'sequential':
            return SequentialExecutor()

        case 'joblib':
            # if shared_nbytes >= memmap_min_bytes:
            return make_memmap_joblib_executor_from_settings(n_jobs=maxworkers,
                                                             backend=joblib_backend_large,
                                                             verbose=0,
                                                             memmap_min_bytes=memmap_min_bytes,
                                                             pickle_min_bytes=pickle_min_bytes,
                                                             temp_dir=temp_dir,
                                                             cleanup_on_shutdown=cleanup_on_shutdown
                                                             )
        case 'ray':
            store_1 = int_cfg.mem_object_store_1 if hasattr(int_cfg, 'mem_object_store_1') else None
            store_2 = int_cfg.mem_object_store_2 if hasattr(int_cfg, 'mem_object_store_2') else None
            ray_was_running = ray.is_initialized()

            init_ray_for_forest_with_fallback(maxworkers=maxworkers,
                                              mem_object_store_1=store_1,
                                              mem_object_store_2=store_2,
                                              ray_err_txt=ray_err_txt,
                                              ray_init_kwargs=ray_init_kwargs,
                                              )
            executor = RayExecutor(num_cpus=maxworkers,
                                  ray_init_kwargs={} if ray_init_kwargs is None
                                                     else ray_init_kwargs,
                                  ray_remote_kwargs={} if ray_remote_kwargs is None
                                                       else ray_remote_kwargs,
                                  )
            executor._started_here = not ray_was_running   # pylint: disable=W0212

            return executor
        case _:
            raise ValueError(f'Unknown forest backend: {int_cfg.mp_backend}')


def make_memmap_joblib_executor_from_settings(*, n_jobs: int,
                                              backend: str = 'loky',
                                              verbose: int = 0,
                                              memmap_min_bytes: int = 4 * 1024 * 1024,
                                              pickle_min_bytes: int = 128 * 1024 * 1024,
                                              temp_dir: str | Path | None = None,
                                              cleanup_on_shutdown: bool = True,
                                              ):
    """Small wrapper to keep the forest factory compact."""
    return make_memmap_joblib_executor(MemmapJoblibCfg(n_jobs=n_jobs,
                                                       backend=backend,
                                                       verbose=verbose,
                                                       memmap_min_bytes=memmap_min_bytes,
                                                       pickle_min_bytes=pickle_min_bytes,
                                                       temp_dir=temp_dir,
                                                       cleanup_on_shutdown=cleanup_on_shutdown,
                                                       )
                                       )


@contextmanager    # converts this function to a generator
def forest_executor_with_shared(*, int_cfg: Any,
                                maxworkers: int,
                                shared_obj: Any,
                                shared_name: str,
                                ray_err_txt: str,
                                fail_txt: str,
                                ) -> Iterator[tuple[ParallelExecutor, Any, int]]:
    """Create forest executor, apply ray->sequential fallback, register shared data."""
    try:
        executor = make_forest_executor(int_cfg=int_cfg,
                                        maxworkers=maxworkers,
                                        ray_err_txt=ray_err_txt,
                                        )
    except RuntimeError as exc:
        if int_cfg.mp_backend == 'ray':
            maxworkers = 1
            executor = SequentialExecutor()
        else:
            raise RuntimeError(fail_txt) from exc

    try:
        data_handle = executor.put_shared(shared_obj, name=shared_name)
        yield executor, data_handle, maxworkers
    finally:
        executor.shutdown()


def map_task_batches(*, executor: ParallelExecutor,
                     tasks: list[TaskSpec[R]],
                     int_cfg: Any,
                     maxworkers: int,
                     min_worker_waves: int = 1,
                     stream_results: bool = False,
                     ) -> Iterator[R]:
    """Yield task results batch by batch."""
    map_iter = getattr(executor, 'map_iter', None)
    if stream_results and callable(map_iter):
        yield from executor.map_iter(tasks)
        return
    
    batch_size = batch_size_fct(no_batches=int_cfg.mp_batches,
                                no_workers=maxworkers,
                                no_tasks=len(tasks),
                                min_worker_waves=min_worker_waves,
                                )
    for task_batch in batched_tasks(tasks, batch_size=batch_size):
        yield from executor.map(task_batch)


@contextmanager
def forest_executor_context(*,
                            int_cfg: Any,
                            maxworkers: int,
                            ray_err_txt: str,
                            fail_txt: str,
                            ) -> Iterator[tuple[ParallelExecutor, int]]:
    """Create forest executor and apply ray->sequential fallback."""
    try:
        executor = make_forest_executor(int_cfg=int_cfg, maxworkers=maxworkers,
                                        ray_err_txt=ray_err_txt,
                                        )
    except RuntimeError as exc:
        if int_cfg.mp_backend == 'ray':
            maxworkers = 1
            executor = SequentialExecutor()
        else:
            raise RuntimeError(fail_txt) from exc

    try:
        yield executor, maxworkers
    finally:
        executor.shutdown()
