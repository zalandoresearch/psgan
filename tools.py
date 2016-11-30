# -*- coding: utf-8 -*-
import sys, os
from time import time


def create_dir(folder):
    '''
    creates a folder, if necessary
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)


class TimePrint(object):
    '''
    Simple convenience class to print who long it takes between successive calls to its __init__ function.
    Usage example:
        TimePrint("some text")          -- simply prints "some text"
        <do some stuff here>
        TimePrint("some other text ")   -- prints "some other text (took ?s)", where ? is the time passed since TimePrint("some text") was called
    '''
    t_last = None

    def __init__(self, text):
        TimePrint.p(text)

    @classmethod
    def p(cls, text):
        t = time()
        print text,
        if cls.t_last!=None:
            print " (took ", t-cls.t_last, "s)"
        cls.t_last = t
        sys.stdout.flush()


if __name__=="__main__":
    print "this is just a library."
