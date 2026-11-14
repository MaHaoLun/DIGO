"""Explicit raw4096+guard4096 ABI for experimental guarded method 9.

This wrapper does not construct a guard or infer an endpoint. Both masks must
already include the source and destination cells, and guard must contain raw.
Legacy methods retain the original 4096-byte mask ABI. Graph/model loading and
Python-side argument validation are outside the returned native search timer.
"""
import ctypes as C
import json
import operator
import os
from pathlib import Path


GRID_CELLS = 4096
LEGACY_METHODS = frozenset((0, 1, 4, 5, 6, 8))
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _integer(value, name):
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise ValueError(f"{name} is outside signed 64-bit range")
    return value


def _mask(mask, name):
    if mask is None:
        raise ValueError(f"{name} is required")
    shape = getattr(mask, "shape", None)
    if shape is not None and tuple(shape) != (GRID_CELLS,):
        raise ValueError(f"{name} must have shape (4096,), got {tuple(shape)}")
    try:
        if len(mask) != GRID_CELLS:
            raise ValueError(f"{name} must contain exactly 4096 entries")
    except TypeError as exc:
        raise ValueError(f"{name} must be a flat mask") from exc
    values = []
    for value in mask:
        if getattr(value, "shape", ()) != ():
            raise ValueError(f"{name} must be flat, with scalar entries")
        try:
            integer = int(value)
            valid = value == integer and integer in (0, 1)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"{name} entries must be binary 0/1 values")
        values.append(integer)
    return tuple(values)


class GuardedNative:
    """One immutable graph handle, used by one query thread at a time."""

    def __init__(self, graph, library=None):
        self.graph = Path(graph).resolve(strict=True)
        stat = self.graph.stat()
        self._graph_stamp = stat.st_size, stat.st_mtime_ns
        self._node_cells = None
        path = library or os.environ.get("E4_GUARDED_LIBRARY")
        if path is None:
            path = Path(__file__).with_name("libguarded_regional_v1.so")
        self.lib = C.CDLL(str(path))
        self.lib.e4_create.argtypes = [C.c_char_p]
        self.lib.e4_create.restype = C.c_void_p
        self.lib.e4_destroy.argtypes = [C.c_void_p]
        self.lib.e4_error.restype = C.c_char_p
        self.lib.e4_query.argtypes = [
            C.c_void_p, *([C.c_longlong] * 4), C.c_int,
            C.POINTER(C.c_ubyte),
        ]
        self.lib.e4_query.restype = C.c_char_p
        self.h = self.lib.e4_create(os.fsencode(self.graph))
        if not self.h:
            self._raise_native("could not load graph")

    def _raise_native(self, fallback):
        error = self.lib.e4_error()
        raise RuntimeError(error.decode(errors="replace") if error else fallback)

    def _endpoints(self, source, destination):
        """Read only the immutable graph header/node table, once per handle."""
        if self._node_cells is None:
            stat = self.graph.stat()
            if (stat.st_size, stat.st_mtime_ns) != self._graph_stamp:
                raise RuntimeError("graph file changed after native graph load")
            with self.graph.open() as stream:
                tokens = (token for line in stream for token in line.split())
                try:
                    n, m, knots, step = [int(next(tokens)) for _ in range(4)]
                    if n <= 0 or m < 0 or knots <= 0 or step <= 0:
                        raise ValueError("invalid graph header")
                    cells = {}
                    for _ in range(n):
                        node, cell = int(next(tokens)), int(next(tokens))
                        if node in cells or not 0 <= cell < GRID_CELLS:
                            raise ValueError("invalid graph node/cell table")
                        cells[node] = cell
                except StopIteration as exc:
                    raise ValueError("truncated graph header/node table") from exc
            stat = self.graph.stat()
            if (stat.st_size, stat.st_mtime_ns) != self._graph_stamp:
                raise RuntimeError("graph file changed while reading endpoint cells")
            self._node_cells = cells
        try:
            return self._node_cells[source], self._node_cells[destination]
        except KeyError as exc:
            raise ValueError(f"unknown graph endpoint ID: {exc.args[0]}") from exc

    def query(self, s, d, lo, hi, method=0, mask=None, guard_mask=None):
        if not self.h:
            raise RuntimeError("native handle is closed")
        s, d = _integer(s, "source"), _integer(d, "destination")
        lo, hi = _integer(lo, "lo"), _integer(hi, "hi")
        method = _integer(method, "method")
        if lo < 0 or lo >= hi:
            raise ValueError("expected a nonnegative, nonempty closed interval")
        if method not in LEGACY_METHODS and method != 9:
            raise ValueError("unsupported method")
        if method == 9:
            raw = _mask(mask, "raw_mask")
            guard = _mask(guard_mask, "guard_mask")
            if any(r and not g for r, g in zip(raw, guard)):
                raise ValueError("guard_mask must contain raw_mask")
            source_cell, destination_cell = self._endpoints(s, d)
            if not raw[source_cell] or not raw[destination_cell]:
                raise ValueError("raw_mask must include both endpoint cells")
            # A distinct 8192-byte allocation is essential. Passing the old
            # 4096-byte buffer to native method 9 would read beyond its bounds.
            pointer = (C.c_ubyte * (2 * GRID_CELLS))(*(raw + guard))
        else:
            if guard_mask is not None:
                raise ValueError("guard_mask is only accepted for method 9")
            if mask is None:
                if method not in (0, 6):
                    raise ValueError("mask is required for this legacy method")
                pointer = None
            else:
                values = _mask(mask, "mask")
                pointer = (C.c_ubyte * GRID_CELLS)(*values)
        result = self.lib.e4_query(self.h, s, d, lo, hi, method, pointer)
        if result is None:
            self._raise_native("native query failed")
        return json.loads(result)

    def query_guarded(self, s, d, lo, hi, raw_mask, guard_mask):
        return self.query(s, d, lo, hi, 9, raw_mask, guard_mask)

    def close(self):
        if self.h:
            self.lib.e4_destroy(self.h)
            self.h = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


# Compatible import spelling for existing one-handle worker code.
Native = GuardedNative
