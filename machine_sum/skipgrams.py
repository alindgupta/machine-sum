""" Skipgrams and n-grams """

import itertools
from typing import Iterable, Iterator, Generator, Any
import numpy as np


def add_padding(data: Iterable, num_pads: int, pad: Any,
                left_pad=True, right_pad=True) -> Iterator:
    """ Pad data """
    data = iter(data)

    if left_pad:
        data = itertools.chain((pad, ) * num_pads, data)
    if right_pad:
        data = itertools.chain(data, (pad, ) * num_pads)
    return data


def exec_ngrams(data: Iterable, ngrams: int) -> Generator:
    """ """
    data = iter(data)
    gramlist = []
    while ngrams > 1:
        gramlist.append(next(data))
        ngrams -= 1
    for item in data:
        gramlist.append(item)
        yield (np.array(tuple(gramlist))).T
        del gramlist[0]


def exec_ngrams2(data: Iterable, ngrams: int, batch_size: int) -> Generator:
    """ """
    data = iter(data)
    result = itertools.islice(data, ngrams)
    if len(result) == ngrams:
        yield (np.transpose(np.array(result, np.int32)))
    for elem in result:
        result = result + (elem, )
        yield (np.transpose(np.array(result, np.int32)))
