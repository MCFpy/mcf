"""
Contains general functions for cuda estimation on gpu.

Created on Mon Dec  4 14:34:33 2023

@author: MLechner

# -*- coding: utf-8 -*-
"""
import numpy as np
import torch


def tdtype(type_='float', precision=32):
    """Set precision and type of torch tensor."""
    if type_ == 'bool':
        datatype = torch.bool
    elif type_ == 'int':
        if precision == 8:
            datatype = torch.int8
        elif precision == 16:
            datatype = torch.int16
        elif precision == 32:
            datatype = torch.int32
        elif precision == 64:
            datatype = torch.int64
        else:
            raise ValueError('Precision for int must be 8, 16, 32, or 64')
    elif type_ == 'float':
        if precision == 16:
            datatype = torch.float16
        elif precision == 32:
            datatype = torch.float32
        elif precision == 64:
            datatype = torch.float64
        else:
            raise ValueError('Precision for float must be 16, 32, or 64')
    else:
        raise ValueError('type_ must be bool, int, or float.')
    return datatype


def split_into_batches(data_size, batch_max_size):
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
