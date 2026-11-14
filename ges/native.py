import ctypes as C
import json
import os
from pathlib import Path


class Native:
    def __init__(self, graph, library=None):
        self.lib=C.CDLL(str(library or os.environ.get('E4_LIBRARY') or Path(__file__).with_name('libregional.so')))
        self.lib.e4_create.argtypes=[C.c_char_p];self.lib.e4_create.restype=C.c_void_p
        self.lib.e4_destroy.argtypes=[C.c_void_p]
        self.lib.e4_error.restype=C.c_char_p
        self.lib.e4_query.argtypes=[C.c_void_p,*([C.c_longlong]*4),C.c_int,C.POINTER(C.c_ubyte)]
        self.lib.e4_query.restype=C.c_char_p
        self.h=self.lib.e4_create(str(graph).encode())
        if not self.h:raise RuntimeError(self.lib.e4_error().decode())

    def query(self,s,d,lo,hi,method=0,mask=None):
        ptr=None if mask is None else (C.c_ubyte*4096)(*mask)
        raw=self.lib.e4_query(self.h,s,d,lo,hi,method,ptr)
        if raw is None:raise RuntimeError(self.lib.e4_error().decode())
        return json.loads(raw)

    def close(self):
        if self.h:self.lib.e4_destroy(self.h);self.h=None
