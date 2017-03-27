#
#   Updated: March 25, 2017
#   TODO: Add support for processing multiple files
#
#
# ============================================================================

""" Module for lazy text file IO, tokenization """

import os
import sys
import pickle
from typing import List, Callable, Generator

try:
    import nltk
except ImportError:
    print('Could not import nltk package')


# aliases for typing hints
Token = str
Id = int

class DataUtils:
    """

    Parameters
    ----------
    filepath: absolute file path for a text file
    buffering: buffer for reads, not really important for lazy IO
    encoding: text-encoding, eg. utf-8, ascii

    """
    def __init__(self, filepath, buffering=-1, encoding='utf-8'):

        # require python 3.4 or higher
        assert sys.version_info >= (3, 4)

        if not os.path.isfile(filepath):
            raise IOError('Could not find ', filepath)
        self.filename = filepath
        self.buffering = buffering  # -1 is default
        self.encoding = encoding

        # fields set when context manager invoked
        self._fhandle = None            # file handle
        self._lines = None              # line iterator
        self._context_managed = False

    def __enter__(self):
        """ Open a file, set a line iterator """
        self._fhandle = open(self.filename, 'r', self.buffering, self.encoding)
        self._lines = (line for line in self._fhandle)
        self._context_managed = True
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        """ Check for errors and close file handle """
        if exec_type is not None:
            print(exec_type, exec_value, traceback)
        if self._fhandle:
            self._fhandle.close()
        else:
            raise IOError('__DEBUG__: File handle closed prematurely (code 1)')

    def process(self, func, *args, **kwargs) -> Generator:
        """ Apply a text processing function lazily """
        if not self._context_managed:
            raise Exception('Object must be initialized using a' \
            'context manager')
        assert isinstance(func, callable), 'Require a function'
        yield from (func(line, *args, **kwargs) for line in self._lines)

    def dump(self):
        """
        Call the process method, consume the entire file
        Pickle Dict[Token, Id] and List[List[Id]]
        """
        pass
