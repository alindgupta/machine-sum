import os
import sys
sys.path.insert(
        0, os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), '..')))

from machine_sum.data_utils import DataUtils  # NOQA

""" proper initialization tests """


def test_ioerror():
    test_token = 1
    try:
        DataUtils('file_doesnt_exist.txt')
    except IOError:
        test_token = 0
    assert test_token == 0


def test_fields_initialization():
    test_token = 0
    obj = DataUtils('text_file.txt')
    if obj._fhandle:
        test_token += 1
    if obj._lines:
        test_token += 1
    if obj._context_managed:
        test_token += 1
    assert test_token == 0, '{}/3 test(s) failed'.format(test_token)


""" context handler tests """


def test_fields_updated():
    with DataUtils('text_file.txt') as obj:
        test_token = 0
        with DataUtils('text_file.txt') as obj:
            if not obj._fhandle:
                test_token += 1
            if not obj._lines:
                test_token += 1
            if not obj._context_managed:
                test_token += 1
        assert test_token == 0, '{}/3 test(s) failed'.format(test_token)


def test_ctxt_handler_requirement():
    pass
