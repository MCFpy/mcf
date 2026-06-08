"""
Created on Mon Mar 23 10:56:32 2026.

@author: MLechner

-*- coding: utf-8 -*-

Joblib and ray classes, methods and functions for parallel computating.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import count
from math import ceil
from platform import system
from typing import Any, Callable, ClassVar, Generic, Literal, TypeVar, TYPE_CHECKING
from mcf.mcf_print_stats import print_mcf
try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = delayed = None
try:
    import ray
except ImportError:
    ray = None

if TYPE_CHECKING:
    from mcf.mcf_init import GenCfg, IntCfg

R = TypeVar('R')

BackendName = Literal['sequential', 'joblib', 'ray']
JoblibBackend = Literal['loky', 'threading']


@dataclass(slots=True, kw_only=True)
class SharedHandle:
    """Opaque handle for backend-managed shared state."""

    key: str


@dataclass(slots=True, kw_only=True)
class TaskSpec(Generic[R]):
    """Description of one unit of work for a parallel backend."""

    func: Callable[..., R]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    name: str | None = None


@dataclass(slots=True, kw_only=True)
class ParallelCfg:
    """Generic backend configuration."""

    backend: BackendName = 'joblib'
    n_jobs: int = -1
    joblib_backend: JoblibBackend = 'loky'
    joblib_verbose: int = 0
    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)
    ray_remote_kwargs: dict[str, Any] = field(default_factory=dict)
    joblib_idle_worker_timeout: int | None = 3600
    joblib_inner_max_num_threads: int | None = None


class SharedRegistry(ABC):
    """Abstract storage for backend-managed shared objects."""

    @abstractmethod
    def put(self, obj: Any, *, name: str | None = None) -> SharedHandle:
        """Store an object and return a handle."""

    @abstractmethod
    def get(self, handle: SharedHandle) -> Any:
        """Resolve a handle to the stored object."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored objects."""


@dataclass(slots=True)
class LocalSharedRegistry(SharedRegistry):
    """Local in-process shared registry."""

    _store: dict[str, Any] = field(default_factory=dict)
    _counter: count = field(default_factory=count)

    def put(self, obj: Any, *, name: str | None = None) -> SharedHandle:
        """Put data to store."""
        key = name if name is not None else f'shared_{next(self._counter)}'
        if key in self._store:
            raise ValueError(f'Shared key already exists: {key}')
        self._store[key] = obj
        return SharedHandle(key=key)

    def get(self, handle: SharedHandle) -> Any:
        """Get data from store."""
        try:
            return self._store[handle.key]
        except KeyError as exc:
            raise KeyError(f'Unknown shared handle: {handle.key}') from exc

    def clear(self) -> None:
        """"Clear store."""
        self._store.clear()


@dataclass(slots=True)
class RaySharedRegistry(SharedRegistry):
    """Ray-backed shared registry storing Ray object references internally."""

    _store: dict[str, Any] = field(default_factory=dict)
    _counter: count = field(default_factory=count)

    def __post_init__(self) -> None:
        """Check if ray is loaded."""
        if ray is None:  # pragma: no cover
            raise ImportError('ray is required for RaySharedRegistry')

    def put(self, obj: Any, *, name: str | None = None) -> SharedHandle:
        """Put data to shared."""
        key = name if name is not None else f'shared_{next(self._counter)}'
        if key in self._store:
            raise ValueError(f'Shared key already exists: {key}')
        self._store[key] = ray.put(obj)
        return SharedHandle(key=key)

    def get(self, handle: SharedHandle) -> Any:
        """Get data from shared."""
        try:
            ref = self._store[handle.key]
        except KeyError as exc:
            raise KeyError(f'Unknown shared handle: {handle.key}') from exc
        return ray.get(ref)

    def get_ref(self, handle: SharedHandle) -> Any:
        """Get references."""
        try:
            return self._store[handle.key]
        except KeyError as exc:
            raise KeyError(f'Unknown shared handle: {handle.key}') from exc

    def clear(self) -> None:
        """Ckear shared."""
        self._store.clear()


class ParallelExecutor(ABC):
    """Backend-agnostic execution interface."""

    @abstractmethod
    def put_shared(self, obj: Any, *, name: str | None = None) -> SharedHandle:
        """Register a read-only shared object and return its handle."""

    @abstractmethod
    def map(self, tasks: list[TaskSpec[R]]) -> list[R]:
        """Execute tasks and return results in submission order."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release backend-managed resources."""


def _run_task(task: TaskSpec[R]) -> R:
    """Top-level helper so it is serializable under Joblib and Ray."""
    return task.func(*task.args, **task.kwargs)


def _resolve_local_obj(obj: Any, shared: LocalSharedRegistry) -> Any:
    """Replace SharedHandle by the underlying local object recursively."""
    if isinstance(obj, SharedHandle):
        return shared.get(obj)
    if isinstance(obj, tuple):
        return tuple(_resolve_local_obj(x, shared) for x in obj)
    if isinstance(obj, list):
        return [_resolve_local_obj(x, shared) for x in obj]
    if isinstance(obj, dict):
        return {key: _resolve_local_obj(value, shared) for key, value in obj.items()}

    return obj


def _contains_shared_handle(obj: Any) -> bool:
    """Check recursively whether an object contains a SharedHandle."""
    if isinstance(obj, SharedHandle):
        return True

    if isinstance(obj, tuple):
        return any(_contains_shared_handle(x) for x in obj)

    if isinstance(obj, list):
        return any(_contains_shared_handle(x) for x in obj)

    if isinstance(obj, dict):
        return any(_contains_shared_handle(value) for value in obj.values())

    return False


def _resolve_ray_obj(obj: Any, shared: RaySharedRegistry) -> Any:
    """
    Resolve SharedHandle objects for Ray task submission.

    Allowed:
    - a SharedHandle itself
    - direct SharedHandle entries in the top-level task args tuple
    - direct SharedHandle values in the top-level task kwargs dict

    Rejected:
    - SharedHandle nested deeper inside another container, because Ray does not
      automatically dereference nested ObjectRef values.
    """
    if isinstance(obj, SharedHandle):
        return shared.get_ref(obj)

    if isinstance(obj, tuple):
        resolved: list[Any] = []
        for value in obj:
            if isinstance(value, SharedHandle):
                resolved.append(shared.get_ref(value))
            elif _contains_shared_handle(value):
                raise ValueError('Nested SharedHandle detected in Ray task arguments. Pass '
                                 'shared data as a direct task argument or keyword argument, '
                                 'not inside a container.'
                                 )
            else:
                resolved.append(value)
        return tuple(resolved)

    if isinstance(obj, list):
        resolved_list: list[Any] = []
        for value in obj:
            if isinstance(value, SharedHandle):
                resolved_list.append(shared.get_ref(value))
            elif _contains_shared_handle(value):
                raise ValueError('Nested SharedHandle detected in Ray task arguments. Pass '
                                 'shared data as a direct task argument or keyword argument, '
                                 'not inside a container.'
                                 )
            else:
                resolved_list.append(value)
        return resolved_list

    if isinstance(obj, dict):
        resolved_dict: dict[str, Any] = {}
        for key, value in obj.items():
            if isinstance(value, SharedHandle):
                resolved_dict[key] = shared.get_ref(value)
            elif _contains_shared_handle(value):
                raise ValueError('Nested SharedHandle detected in Ray task arguments. Pass '
                                 'shared data as a direct task argument or keyword argument, '
                                 'not inside a container.'
                                 )
            else:
                resolved_dict[key] = value
        return resolved_dict

    return obj


@dataclass(slots=True)
class SequentialExecutor(ParallelExecutor):
    """Reference backend for debugging, testing, and exact tracing."""

    _shared: LocalSharedRegistry = field(default_factory=LocalSharedRegistry)

    def put_shared(self, obj: Any, *, name: str | None = None) -> SharedHandle:
        """Put to shared."""
        return self._shared.put(obj, name=name)

    def map(self, tasks: list[TaskSpec[R]]) -> list[R]:
        """Create mapping."""
        resolved_tasks = [TaskSpec(func=task.func,
                                   args=_resolve_local_obj(task.args, self._shared),
                                   kwargs=_resolve_local_obj(task.kwargs, self._shared),
                                   name=task.name,
                                   )
                          for task in tasks
                          ]
        return [_run_task(task) for task in resolved_tasks]

    def shutdown(self) -> None:
        """Shut it down."""
        self._shared.clear()


@dataclass(slots=True)
class JoblibExecutor(ParallelExecutor):
    """Joblib-backed executor for local single-machine execution."""

    n_jobs: int = -1
    backend: JoblibBackend = 'loky'
    verbose: int = 0
    idle_worker_timeout: int | None = 3600
    inner_max_num_threads: int | None = None
    _shared: LocalSharedRegistry = field(default_factory=LocalSharedRegistry)

    def __post_init__(self) -> None:
        """Check joblib import."""
        if Parallel is None or delayed is None:  # pragma: no cover
            raise ImportError('joblib is required for JoblibExecutor')

    def put_shared(self, obj: Any, *, name: str | None = None) -> SharedHandle:
        """Put to shared."""
        return self._shared.put(obj, name=name)

    def map(self, tasks: list[TaskSpec[R]]) -> list[R]:
        """Map the tasks to a list."""
        resolved_tasks = [TaskSpec(func=task.func,
                                   args=_resolve_local_obj(task.args, self._shared),
                                   kwargs=_resolve_local_obj(task.kwargs, self._shared),
                                   name=task.name,
                                   )
                          for task in tasks
                          ]
        parallel_kwargs = joblib_parallel_kwargs(backend=self.backend,
                                                 verbose=self.verbose,
                                                 idle_worker_timeout=self.idle_worker_timeout,
                                                 inner_max_num_threads=self.inner_max_num_threads,
                                                 )
        return Parallel(n_jobs=self.n_jobs, **parallel_kwargs
                        )(delayed(_run_task)(task) for task in resolved_tasks)

    def shutdown(self) -> None:
        """Shut down joblib."""
        self._shared.clear()


def _run_ray_task_impl(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Plain top-level function, always defined."""
    return func(*args, **kwargs)



@dataclass(slots=True)
class RayExecutor(ParallelExecutor):
    """Ray-backed executor for object-store sharing or cluster-style execution."""

    num_cpus: int | None = None
    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)
    ray_remote_kwargs: dict[str, Any] = field(default_factory=dict)
    _shared: RaySharedRegistry | None = None
    _started_here: bool = False

    _remote_runner_cls: ClassVar[Any | None] = None

    @classmethod
    def _get_remote_runner_cls(cls) -> Any:
        """Create and cache Ray remote wrapper lazily."""
        if cls._remote_runner_cls is None:
            cls._remote_runner_cls = ray.remote(_run_ray_task_impl)
        return cls._remote_runner_cls

    def __post_init__(self) -> None:
        """Check ray import."""
        if ray is None:  # pragma: no cover
            raise ImportError('ray is required for RayExecutor')

        if not ray.is_initialized():
            init_kwargs = dict(self.ray_init_kwargs)
            if self.num_cpus is not None and 'num_cpus' not in init_kwargs:
                init_kwargs['num_cpus'] = self.num_cpus
            ray.init(**init_kwargs)
            self._started_here = True

        if self._shared is None:
            self._shared = RaySharedRegistry()

        self._get_remote_runner_cls()

    def put_shared(self, obj: Any, *, name: str | None = None) -> SharedHandle:
        """Put to share storage."""
        return self._shared.put(obj, name=name)


    def map(self, tasks: list[TaskSpec[R]]) -> list[R]:
        """Map task list."""
        resolved_tasks = [TaskSpec(func=task.func,
                                   args=_resolve_ray_obj(task.args, self._shared),
                                   kwargs=_resolve_ray_obj(task.kwargs, self._shared),
                                   name=task.name,
                                   )
                          for task in tasks
                          ]
        remote_runner = self._get_remote_runner_cls()
        if self.ray_remote_kwargs:
            remote_runner = remote_runner.options(**self.ray_remote_kwargs)

        futures = [remote_runner.remote(task.func, *task.args, **task.kwargs)
                   for task in resolved_tasks
                   ]
        return ray.get(futures)

    def shutdown(self) -> None:
        """Shut down ray."""
        if self._shared is not None:
            self._shared.clear()

        if self._started_here and ray.is_initialized():
            ray.shutdown()

        if not ray.is_initialized():
            self._clear_remote_runner_cls()

    @classmethod
    def _clear_remote_runner_cls(cls) -> None:
        """Clear the cached Ray remote wrapper."""
        cls._remote_runner_cls = None

def make_executor(cfg: ParallelCfg) -> ParallelExecutor:
    """Make factory for backend-agnostic executor construction."""
    if cfg.backend == 'sequential':
        return SequentialExecutor()

    if cfg.backend == 'joblib':
        n_jobs = 60 if system() == 'Windows' and cfg.n_jobs > 60 else cfg.n_jobs
        return JoblibExecutor(n_jobs=n_jobs,
                              backend=cfg.joblib_backend,
                              verbose=cfg.joblib_verbose,
                              idle_worker_timeout=cfg.joblib_idle_worker_timeout,
                              inner_max_num_threads=cfg.joblib_inner_max_num_threads,
                              )
    if cfg.backend == 'ray':
        num_cpus = None if cfg.n_jobs == -1 else cfg.n_jobs
        return RayExecutor(num_cpus=num_cpus,
                           ray_init_kwargs=cfg.ray_init_kwargs,
                           ray_remote_kwargs=cfg.ray_remote_kwargs,
                           )
    raise ValueError(f'Unknown backend: {cfg.backend}')


def run_tasks(tasks: list[TaskSpec[R]], *, cfg: ParallelCfg | None = None) -> list[R]:
    """Use as convenience helper for fire-and-forget use."""
    executor = make_executor(ParallelCfg() if cfg is None else cfg)
    try:
        return executor.map(tasks)
    finally:
        executor.shutdown()


def _demo_worker(*, start: int, stop: int, data: dict[str, list[int]]) -> int:
    """Use as example of top-level worker used by demo_sum()."""
    values = data['values']
    return sum(values[start:stop])


def batched_tasks(tasks: list[TaskSpec[Any]], *,
                  batch_size: int,
                  ) -> Iterable[list[TaskSpec[Any]]]:
    """Yield consecutive task batches."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')

    for start in range(0, len(tasks), batch_size):
        yield tasks[start:start + batch_size]


def batch_size_fct(*, no_batches: int,
                   no_workers: int,
                   no_tasks: int,
                   min_worker_waves: int = 1,
                   ) -> int:
    """Determine an appropriate task batch size.

    ``no_batches`` is the desired maximum number of batches. The returned batch size
    is at least one worker wave, but never larger than the number of tasks.
    """
    if no_tasks <= 0:
        return 1

    if no_workers <= 0:
        no_workers = 1

    if min_worker_waves <= 0:
        min_worker_waves = 1

    if no_batches == -1:
        no_batches = ceil(no_tasks / (no_workers * min_worker_waves))

    if no_batches <= 0:
        no_batches = 1

    batch_size = ceil(no_tasks / no_batches)
    min_worker_wave = min(no_workers * min_worker_waves, no_tasks)

    return min(no_tasks, max(min_worker_wave, batch_size))


def joblib_parallel_kwargs(*, backend: JoblibBackend,
                              verbose: int,
                              temp_folder: str | None = None,
                              max_nbytes: str | int | None = None,
                              set_max_nbytes: bool = False,
                              idle_worker_timeout: int | None = None,
                              inner_max_num_threads: int | None = 1,
                              return_as: str = 'list',
                              ) -> dict[str, Any]:
    """Collect keyword arguments for joblib.Parallel."""
    kwargs: dict[str, Any] = {'backend': backend,
                              'verbose': verbose,
                              }
    if temp_folder is not None:
        kwargs['temp_folder'] = temp_folder

    if set_max_nbytes:
        kwargs['max_nbytes'] = max_nbytes

    if return_as != 'list':
        kwargs['return_as'] = return_as

    if backend == 'loky':
        if idle_worker_timeout is not None:
            kwargs['idle_worker_timeout'] = idle_worker_timeout
        if inner_max_num_threads is not None:
            kwargs['inner_max_num_threads'] = inner_max_num_threads

    return kwargs


def print_runtime_info(gen_cfg: 'GenCfg',
                       int_cfg: 'IntCfg',
                       maxworkers: int,
                       txt_method: str = ''
                       ) -> None:
    """Print runtime info."""
    if gen_cfg.with_output and gen_cfg.verbose:
        txt = f'\nNumber of parallel processes ({txt_method}): {maxworkers} '
        if maxworkers > 1:
            txt +=  '(using '
            if int_cfg.mp_use_old_ray or int_cfg.mp_backend == 'ray':
                txt += 'ray for multiprocessing) '
            elif int_cfg.mp_backend == 'sequential':
                txt += 'sequential as backend for multiprocessing) '
            else:
                txt += 'joblib for multiprocessing) '
        print_mcf(gen_cfg, txt, summary=False)


# --------- Only the following classes, methods and functions are meant to be public. --------------
__all__ = ['BackendName',
           'JoblibBackend',
           'JoblibExecutor',
           'LocalSharedRegistry',
           'ParallelCfg',
           'ParallelExecutor',
           'RayExecutor',
           'RaySharedRegistry',
           'SequentialExecutor',
           'SharedHandle',
           'TaskSpec',
           'make_executor',
           'run_tasks',
           'batched_tasks',
           'print_runtime_info',
           'joblib_parallel_kwargs',
           ]
