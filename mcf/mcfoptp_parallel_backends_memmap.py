"""
Created on Mon Mar 23 10:56:32 2026.

@author: MLechner

-*- coding: utf-8 -*-

Memmap for joblib parallelization.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
import pickle
from shutil import rmtree
from tempfile import mkdtemp
from typing import Any, Literal
from uuid import uuid4
import warnings

import numpy as np

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = delayed = None

from mcf.mcfoptp_parallel_backends_base import (JoblibBackend,
                                                ParallelCfg,
                                                ParallelExecutor,
                                                SharedHandle,
                                                TaskSpec,
                                                joblib_parallel_kwargs,
                                                )

_PICKLE_FILE_CACHE: dict[str, Any] = {}
MmapMode = Literal['r', 'r+']


@dataclass(slots=True, kw_only=True)
class MemmapArrayRef:
    """Lightweight descriptor for a NumPy array stored on disk as a .npy file."""

    path: str
    mmap_mode: MmapMode = 'r'


@dataclass(slots=True, kw_only=True)
class PickleFileRef:
    """Lightweight descriptor for a Python object stored on disk."""

    path: str


@dataclass(slots=True, kw_only=True)
class MemmapJoblibCfg:
    """Configuration for the memmap-aware Joblib executor."""

    n_jobs: int = -1
    backend: JoblibBackend = 'loky'
    verbose: int = 0
    memmap_min_bytes: int = 4 * 1024 * 1024
    pickle_min_bytes: int = 128 * 1024 * 1024
    mmap_mode: MmapMode = 'r'
    temp_dir: str | Path | None = None
    cleanup_on_shutdown: bool = True
    idle_worker_timeout: int = 3600
    inner_max_num_threads: int | None = 1


@dataclass(slots=True)
class MemmapSharedRegistry:
    """Shared registry that writes large arrays and large Python objects once to disk."""

    memmap_min_bytes: int = 4 * 1024 * 1024
    pickle_min_bytes: int = 128 * 1024 * 1024
    mmap_mode: MmapMode = 'r'
    temp_dir: str | Path | None = None
    cleanup_on_shutdown: bool = True
    _store: dict[str, Any] = field(default_factory=dict)
    _key_counter: count = field(default_factory=count)
    _file_counter: count = field(default_factory=count)
    _session_dir: Path | None = None
    _joblib_temp_dir: Path | None = None
    _base_dir: Path | None = None
    _created_base_dir: bool = False
    _closed: bool = False

    def __post_init__(self) -> None:
        """Check directories and paths."""
        self._session_dir = self._make_session_dir(self.temp_dir)
        self._joblib_temp_dir = self._session_dir / 'joblib_tmp'
        self._joblib_temp_dir.mkdir(parents=True, exist_ok=True)

    def _make_session_dir(self, temp_dir: str | Path | None) -> Path:
        """Make the directory for the session."""
        if temp_dir is None:
            self._base_dir = None
            self._created_base_dir = False

            return Path(mkdtemp(prefix='mcf_joblib_memmap_'))

        base = Path(temp_dir)
        self._base_dir = base
        self._created_base_dir = not base.exists()

        base.mkdir(parents=True, exist_ok=True)
        session_dir = base / f'mcf_joblib_memmap_{uuid4().hex}'
        session_dir.mkdir(parents=True, exist_ok=False)

        return session_dir

    def clear(self) -> None:
        """Clear shared data and remove temporary memmap directories."""
        self._store.clear()

        if self._closed or not self.cleanup_on_shutdown or self._session_dir is None:
            return

        try:
            rmtree(self._session_dir)
        except FileNotFoundError:
            # Directory is already gone; cleanup target state is reached.
            pass
        except OSError as exc:
            warnings.warn(f'Could not remove memmap temp directory {self._session_dir!s}: {exc}')
            return

        self._closed = True
        self._remove_created_empty_base_dir()

    def _remove_created_empty_base_dir(self) -> None:
        """Remove temp_dir itself only if this registry created it and it is empty."""
        if not self._created_base_dir or self._base_dir is None:
            return

        try:
            self._base_dir.rmdir()
        except FileNotFoundError:
            # Directory is already gone; cleanup target state is reached.
            return
        except OSError as exc:
            warnings.warn(f'Could not remove empty memmap base directory {self._base_dir!s}: {exc}')

    def put(self, obj: Any, *, name: str | None = None) -> SharedHandle:
        """Put data to shared."""
        key = name if name is not None else f'shared_{next(self._key_counter)}'
        if key in self._store:
            raise ValueError(f'Shared key already exists: {key}')

        externalized = self._externalize_obj(obj, prefix=key)
        self._store[key] = externalized
        return SharedHandle(key=key)

    def get_payload(self, handle: SharedHandle) -> Any:
        """Get data."""
        try:
            return self._store[handle.key]
        except KeyError as exc:
            raise KeyError(f'Unknown shared handle: {handle.key}') from exc

    def _externalize_obj(self, obj: Any, *, prefix: str) -> Any:
        """
        Replace large shared payloads by lightweight file-backed descriptors.
    
        Large numeric NumPy arrays become MemmapArrayRef objects. Large non-NumPy
        Python objects become PickleFileRef objects and are loaded once per worker
        process by _materialize_memmap_obj(...).
        """
        if isinstance(obj, np.ndarray):
            return self._externalize_array(obj, prefix=prefix)

        if isinstance(obj, dict):
            return {key: self._externalize_obj(value, prefix=f'{prefix}_{key}')
                    for key, value in obj.items()
                    }
        if isinstance(obj, list):
            if self._is_array_container(obj):
                return [self._externalize_obj(value, prefix=f'{prefix}_{idx}')
                        for idx, value in enumerate(obj)
                        ]
            externalized = self._externalize_pickle_obj(obj, prefix=prefix)
            if isinstance(externalized, PickleFileRef):
                return externalized

            return [self._externalize_obj(value, prefix=f'{prefix}_{idx}')
                    for idx, value in enumerate(obj)
                    ]
        if isinstance(obj, tuple):
            if self._is_array_container(obj):
                return tuple(self._externalize_obj(value, prefix=f'{prefix}_{idx}')
                             for idx, value in enumerate(obj)
                             )
            externalized = self._externalize_pickle_obj(obj, prefix=prefix)
            if isinstance(externalized, PickleFileRef):
                return externalized

            return tuple(self._externalize_obj(value, prefix=f'{prefix}_{idx}')
                         for idx, value in enumerate(obj)
                         )
        return self._externalize_pickle_obj(obj, prefix=prefix)

    def _externalize_array(self, arr: np.ndarray, *, prefix: str) -> Any:
        """Externalize large numeric arrays to .npy files."""
        if arr.dtype == object:
            return arr

        if arr.nbytes < self.memmap_min_bytes:
            return arr

        path = self._next_array_path(prefix=prefix)
        np.save(path, arr, allow_pickle=False)

        return MemmapArrayRef(path=str(path), mmap_mode=self.mmap_mode)

    def _next_array_path(self, *, prefix: str) -> Path:
        safe_prefix = ''.join(char if char.isalnum() or char in {'_', '-'} else '_'
                              for char in prefix
                              )
        filename = f'{safe_prefix}_{next(self._file_counter):08d}.npy'
        return self._session_dir / filename

    @property
    def session_dir(self) -> Path:
        """Return session directory."""
        return self._session_dir

    @property
    def joblib_temp_dir(self) -> Path:
        """Return temporary directory."""
        return self._joblib_temp_dir

    def _externalize_pickle_obj(self, obj: Any, *, prefix: str) -> Any:
        """Externalize large non-NumPy Python objects to pickle files."""
        try:
            payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, TypeError, AttributeError, ValueError):
            return obj

        if len(payload) < self.pickle_min_bytes:
            return obj

        path = self._next_pickle_path(prefix=prefix)
        with open(path, 'wb') as file:
            file.write(payload)

        return PickleFileRef(path=str(path))

    def _next_pickle_path(self, *, prefix: str) -> Path:
        """Create the next pickle-backed object path."""
        safe_prefix = ''.join(
            char if char.isalnum() or char in {'_', '-'} else '_'
            for char in prefix
        )
        filename = f'{safe_prefix}_{next(self._file_counter):08d}.pkl'
        return self._session_dir / filename

    def _is_array_container(self, obj: Any) -> bool:
        """Return True for containers that mainly hold NumPy arrays or nested array containers."""
        if isinstance(obj, np.ndarray):
            return True

        if isinstance(obj, dict):
            if not obj:
                return False
            return all(self._is_array_container(value) for value in obj.values())

        if isinstance(obj, list | tuple):
            if not obj:
                return False
            return all(self._is_array_container(value) for value in obj)

        return False


def _resolve_shared_handles_for_submission(obj: Any,
                                           shared: MemmapSharedRegistry,
                                           ) -> Any:
    """
    Replace SharedHandle by registry payload recursively.

    Only simple containers are traversed. Arbitrary dataclass instances are
    left unchanged to avoid rebuilding normalized config/state objects.
    """
    if isinstance(obj, SharedHandle):
        return shared.get_payload(obj)

    if isinstance(obj, dict):
        return {key: _resolve_shared_handles_for_submission(value, shared)
                for key, value in obj.items()
                }
    if isinstance(obj, list):
        return [_resolve_shared_handles_for_submission(value, shared)
                for value in obj
                ]
    if isinstance(obj, tuple):
        return tuple(_resolve_shared_handles_for_submission(value, shared)
                     for value in obj
                     )
    return obj


def _materialize_memmap_obj(obj: Any) -> Any:
    """
    Replace file-backed descriptors by worker-local materialized objects.

    MemmapArrayRef is loaded as np.memmap. PickleFileRef is loaded once per
    worker process and then reused from _PICKLE_FILE_CACHE.
    """
    if isinstance(obj, MemmapArrayRef):
        return np.load(obj.path,
                       mmap_mode=obj.mmap_mode,
                       allow_pickle=False,
                       )

    if isinstance(obj, PickleFileRef):
        cached = _PICKLE_FILE_CACHE.get(obj.path)
        if cached is not None:
            return cached

        with open(obj.path, 'rb') as file:
            loaded = pickle.load(file)

        _PICKLE_FILE_CACHE[obj.path] = loaded
        return loaded

    if isinstance(obj, dict):
        return {key: _materialize_memmap_obj(value)
                for key, value in obj.items()
                }

    if isinstance(obj, list):
        return [_materialize_memmap_obj(value) for value in obj]

    if isinstance(obj, tuple):
        return tuple(_materialize_memmap_obj(value) for value in obj)

    return obj


def _run_memmap_task(task: TaskSpec[Any]) -> Any:
    """Materialize memmap descriptors in the worker and then run the task."""
    materialized_task = TaskSpec(func=task.func,
                                 args=_materialize_memmap_obj(task.args),
                                 kwargs=_materialize_memmap_obj(task.kwargs),
                                 name=task.name,
                                 )
    return materialized_task.func(*materialized_task.args, **materialized_task.kwargs)


@dataclass(slots=True)
class MemmapJoblibExecutor(ParallelExecutor):
    """Joblib executor with explicit disk-backed shared-array handling."""

    n_jobs: int = -1
    backend: JoblibBackend = 'loky'
    verbose: int = 0
    memmap_min_bytes: int = 4 * 1024 * 1024
    pickle_min_bytes: int = 128 * 1024 * 1024
    mmap_mode: MmapMode = 'r'
    temp_dir: str | Path | None = None
    cleanup_on_shutdown: bool = True
    idle_worker_timeout: int = 3600
    inner_max_num_threads: int | None = 1
    _shared: MemmapSharedRegistry | None = None

    def __post_init__(self) -> None:
        """Check consistency of inputs."""
        if Parallel is None or delayed is None:  # pragma: no cover
            raise ImportError('joblib is required for MemmapJoblibExecutor')

        if self._shared is None:
            self._shared = MemmapSharedRegistry(memmap_min_bytes=self.memmap_min_bytes,
                                                pickle_min_bytes=self.pickle_min_bytes,
                                                mmap_mode=self.mmap_mode,
                                                temp_dir=self.temp_dir,
                                                cleanup_on_shutdown=self.cleanup_on_shutdown,
                                                )

    def _resolved_tasks(self, tasks: list[TaskSpec[Any]]) -> list[TaskSpec[Any]]:
        """Resolve shared handles before submitting tasks to joblib."""
        return [TaskSpec(func=task.func,
                         args=_resolve_shared_handles_for_submission(task.args, self._shared),
                         kwargs=_resolve_shared_handles_for_submission(task.kwargs, self._shared),
                         name=task.name,
                         )
                for task in tasks
                ]

    def map_iter(self, tasks: list[TaskSpec[Any]]) -> Iterator[Any]:
        """Yield joblib results as soon as they become available."""
        resolved_tasks = self._resolved_tasks(tasks)
    
        parallel_kwargs = joblib_parallel_kwargs(backend=self.backend,
                                                 verbose=self.verbose,
                                                 temp_folder=str(self._shared.joblib_temp_dir),
                                                 max_nbytes=None,
                                                 set_max_nbytes=True,
                                                 idle_worker_timeout=self.idle_worker_timeout,
                                                 inner_max_num_threads=self.inner_max_num_threads,
                                                 return_as='generator_unordered',
                                                 )
        yield from Parallel(n_jobs=self.n_jobs, **parallel_kwargs
                            )(delayed(_run_memmap_task)(task) for task in resolved_tasks)

    def put_shared(self, obj: Any, *, name: str | None = None) -> SharedHandle:
        """Put shared to storage."""
        return self._shared.put(obj, name=name)

    def map(self, tasks: list[TaskSpec[Any]]) -> list[Any]:
        """Create task list."""
        resolved_tasks = self._resolved_tasks(tasks)

        parallel_kwargs = joblib_parallel_kwargs(backend=self.backend,
                                                  verbose=self.verbose,
                                                  temp_folder=str(self._shared.joblib_temp_dir),
                                                  max_nbytes=None,
                                                  set_max_nbytes=True,
                                                  idle_worker_timeout=self.idle_worker_timeout,
                                                  inner_max_num_threads=self.inner_max_num_threads,
                                                  )
        return Parallel(n_jobs=self.n_jobs, **parallel_kwargs
                        )(delayed(_run_memmap_task)(task) for task in resolved_tasks)

    def shutdown(self) -> None:
        """Shut down storage."""
        if self._shared is not None:
            self._shared.clear()

    @property
    def session_dir(self) -> Path:
        """Return session directory."""
        return self._shared.session_dir


def make_memmap_joblib_executor(cfg: MemmapJoblibCfg | None = None) -> MemmapJoblibExecutor:
    """Create simple factory for the memmap-aware Joblib executor."""
    cfg = MemmapJoblibCfg() if cfg is None else cfg
    return MemmapJoblibExecutor(n_jobs=cfg.n_jobs,
                                backend=cfg.backend,
                                verbose=cfg.verbose,
                                memmap_min_bytes=cfg.memmap_min_bytes,
                                pickle_min_bytes=cfg.pickle_min_bytes,
                                mmap_mode=cfg.mmap_mode,
                                temp_dir=cfg.temp_dir,
                                cleanup_on_shutdown=cfg.cleanup_on_shutdown,
                                idle_worker_timeout=cfg.idle_worker_timeout,
                                inner_max_num_threads=cfg.inner_max_num_threads,
                                )


def make_memmap_joblib_executor_from_parallel_cfg(cfg: ParallelCfg, *,
                                                  memmap_min_bytes: int = 4 * 1024 * 1024,
                                                  pickle_min_bytes: int = 128 * 1024 * 1024,
                                                  mmap_mode: MmapMode = 'r',
                                                  temp_dir: str | Path | None = None,
                                                  cleanup_on_shutdown: bool = True,
                                                  ) -> MemmapJoblibExecutor:
    """Build bridge from the base module's ParallelCfg."""
    if cfg.backend != 'joblib':
        raise ValueError('make_memmap_joblib_executor_from_parallel_cfg requires '
                         "cfg.backend == 'joblib'"
                         )
    inner_max_num_threads = (1 if cfg.joblib_inner_max_num_threads is None
                             else cfg.joblib_inner_max_num_threads
                             )
    return MemmapJoblibExecutor(n_jobs=cfg.n_jobs,
                                backend=cfg.joblib_backend,
                                verbose=cfg.joblib_verbose,
                                memmap_min_bytes=memmap_min_bytes,
                                pickle_min_bytes=pickle_min_bytes,
                                mmap_mode=mmap_mode,
                                temp_dir=temp_dir,
                                cleanup_on_shutdown=cleanup_on_shutdown,
                                idle_worker_timeout=cfg.joblib_idle_worker_timeout,
                                inner_max_num_threads=inner_max_num_threads,
                                )


# ----------------------------- This part shows how to use this module -----------------------------
def _demo_chunk_sum(*, start: int, stop: int, data: dict[str, Any]) -> float:
    """Show example of top-level worker used by demo_large_array_sum()."""
    x = data['x']
    return float(np.sum(x[start:stop]))


def demo_large_array_sum(*, n: int = 2_000_000,
                         chunk_size: int = 200_000,
                         memmap_min_bytes: int = 1_000_000,
                         ) -> float:
    """Show minimal demonstration of explicit memmap externalization."""
    executor = MemmapJoblibExecutor(n_jobs=-1,
                                    backend='loky',
                                    memmap_min_bytes=memmap_min_bytes,
                                    )
    try:
        x = np.arange(n, dtype=np.float64)
        data_handle = executor.put_shared({'x': x}, name='demo_data')

        tasks: list[TaskSpec[float]] = []
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            tasks.append(TaskSpec(func=_demo_chunk_sum,
                                  kwargs={'start': start,
                                          'stop': stop,
                                          'data': data_handle,
                                          },
                                  name=f'chunk_{start}_{stop}',
                                  )
                         )
        partial = executor.map(tasks)
        return float(sum(partial))
    finally:
        executor.shutdown()

# --------- Only the following classes, methods and functions are meant to be public. --------------
__all__ = ['MemmapArrayRef',
           'PickleFileRef',
           'MemmapJoblibCfg',
           'MemmapJoblibExecutor',
           'MemmapSharedRegistry',
           'MmapMode',
           'demo_large_array_sum',
           'make_memmap_joblib_executor',
           'make_memmap_joblib_executor_from_parallel_cfg',
           ]
