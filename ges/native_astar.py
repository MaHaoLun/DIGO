"""Separate API: full graph and continuous interval only; no region/method flags."""
import ctypes as C
import json
from pathlib import Path


class ContinuousIntervalAStar:
    def __init__(self, graph, library=None):
        self.lib=C.CDLL(str(library or Path(__file__).with_name('libcontinuous_interval_astar_v1.so')))
        self.lib.ca_create.argtypes=[C.c_char_p];self.lib.ca_create.restype=C.c_void_p
        self.lib.ca_destroy.argtypes=[C.c_void_p];self.lib.ca_error.restype=C.c_char_p
        self.lib.ca_query.argtypes=[C.c_void_p,*([C.c_longlong]*4)];self.lib.ca_query.restype=C.c_char_p
        self.h=self.lib.ca_create(str(graph).encode())
        if not self.h:raise RuntimeError(self.lib.ca_error().decode())

    def query(self,s,d,lo,hi):
        raw=self.lib.ca_query(self.h,s,d,lo,hi)
        if raw is None:raise RuntimeError(self.lib.ca_error().decode())
        return json.loads(raw)

    def close(self):
        if self.h:self.lib.ca_destroy(self.h);self.h=None
