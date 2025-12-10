"""
Contains general functions for cuda estimation on gpu.

Created on Mon Dec  4 14:34:33 2023

@author: MLechner

# -*- coding: utf-8 -*-
"""
import numpy as np

try:
    import torch  # type: ignore[import]
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]


def tdtype(type_: str = 'float', precision: int = 32) -> torch.dtype:
    """Set precision and type of torch tensor."""
    match type_:
        case 'bool':
            return torch.bool

        case 'int':
            match precision:
                case 8: return torch.int8
                case 16: return torch.int16
                case 32: return torch.int32
                case 64: return torch.int64
                case _: raise ValueError('Precision for int must be 8, '
                                         '16, 32, or 64'
                                         )
        case 'float':
            match precision:
                case 16: return torch.float16
                case 32: return torch.float32
                case 64: return torch.float64
                case _: raise ValueError('Precision for float must be '
                                         '16, 32, or 64'
                                         )
        case _:
            raise ValueError('type_ must be bool, int, or float.')


def split_into_batches(data_size: int,
                       batch_max_size: int
                       ) -> list[np.integer]:
    """
    Split a dataset into batches of indices.

    Parameters
    ----------
    data_size : int
        Size of the dataset.

    batch_max : int
        Maximum size of each batch.

    Returns
    -------
    list of tuples
        Each tuple contains indices for a batch.
    """
    indices = np.arange(data_size)
    batches = [tuple(indices[i:i + batch_max_size])
               for i in range(0, data_size, batch_max_size)]

    return batches
