from libcpp.vector cimport vector

import os
import nltk

cdef inline bool is_numeric(str x):
    try:
        float(x)
        return True
    except:
        return False

@cython.boundscheck(False)
cdef process_file(str filename):
    # file is unchecked so make sure it is readable
    cdef vector[str] vect
    with open(filename, 'r') as _file:
        for line in _file:
            temp = nltk.word_tokenize(line)
            for item in temp:
                if not is_numeric(item) and item not in invalid:
                    vect += item




