# A5Cuda Python wrapper

from ctypes import cdll

class A5Cuda(object):
    def __init__(self, lib_path):
        self.a5cuda = cdll.LoadLibrary(lib_path)

    def submit():

