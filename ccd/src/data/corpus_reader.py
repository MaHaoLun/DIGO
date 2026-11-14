"""Strict, shard-lazy training reader for the direct C++ image corpus.

Physical indices address the files' SSSP traversal order. ``row_index`` is the
frozen split identity and is deliberately not used as a physical array index.
The cache contains at most one complete image/route shard, charged by actual
array bytes. Temporary validation masks and caller-owned batches are separate
from the cache payload budget. Returned arrays are immutable views; callers retaining old batches
own that additional lifetime, which is not part of the reader cache.
"""
from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path
import time
from typing import Any, Iterator
import zipfile

import numpy as np


class CorpusError(ValueError):
    """A corpus violates its manifest, storage or row contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusError(message)


def integer(value: Any, name: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def plain_file(path: Path) -> Path:
    # Check before resolving: resolve() alone would conceal an input symlink.
    for item in (path, *path.parents):
        require(not item.is_symlink(), f"linked path: {path}")
    require(path.is_file() and path.stat().st_nlink == 1, f"missing or hardlinked file: {path}")
    return path


def relative_file(root: Path, value: Any) -> Path:
    require(isinstance(value, str) and bool(value), "file path must be a nonempty string")
    rel = Path(value)
    require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe relative path: {value}")
    path = root / rel
    require(path.resolve().is_relative_to(root.resolve()), f"path escapes corpus: {value}")
    return plain_file(path)


_IMAGE_SCHEMA = {"image": ("<f4", (2, 32, 32)), "odt": ("<f4", (5,)), "arr": ("<f4", ())}
_ROUTE_ROWS = {"query_id": "<u8", "interval_id": "<u8", "row_index": "<u8",
               "source": "<i8", "destination": "<i8", "departure": "<i8",
               "travel_time": "<i8", "status": "|u1"}
_ROUTE_SCHEMA = {**_ROUTE_ROWS, "offsets": "<u8", "node_ids": "<i8", "arrival_times": "<i8"}


def _npz_headers(path: Path, expected: set[str], budget: int) -> int:
    """Validate headers and allocation sizes before allowing numpy to allocate."""
    total = 0
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            require(len(infos) == len(expected) and {i.filename for i in infos} ==
                    {name + ".npy" for name in expected}, f"unexpected/duplicate NPZ fields: {path.name}")
            for info in infos:
                require(info.compress_type == zipfile.ZIP_STORED, "corpus NPZ must use ZIP store")
                require(not info.is_dir() and (info.external_attr >> 16) & 0o170000 != 0o120000,
                        "NPZ contains a linked member")
                require(info.file_size <= budget + 65536, "NPZ member exceeds cache allocation budget")
                with archive.open(info) as stream:
                    version = np.lib.format.read_magic(stream)
                    require(version in ((1, 0), (2, 0)), "unsupported NPY header version")
                    header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, fortran, dtype = header(stream, max_header_size=10000)
                    require(not dtype.hasobject and not fortran, "object/Fortran NPZ arrays are forbidden")
                    count = 1
                    for dim in shape:
                        require(type(dim) is int and dim >= 0, "invalid NPY shape")
                        count *= dim
                    size = count * dtype.itemsize
                    total += size
                    require(total <= budget, "NPZ arrays exceed cache allocation budget")
                    require(stream.tell() + size == info.file_size, "NPY header/data length mismatch")
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as error:
        if isinstance(error, CorpusError):
            raise
        raise CorpusError(f"invalid NPZ {path.name}: {error}") from error
    return total


class CorpusReader:
    """Read one split without loading any NPZ until its rows are requested."""

    def __init__(self, manifest_path: str | Path, split: str, cache_bytes: int | None = None):
        path = Path(manifest_path).absolute()
        plain_file(path)
        try:
            self.manifest = json.loads(path.read_text(encoding="utf-8"))
            self.root = path.parent
            manifest = self.manifest
            state = manifest.get("graph_state", "G0")
            require(state in ("G0", "G1"), "unsupported graph_state")
            roles = ({"adaptation", "post_update_test"} if state == "G1" else
                     {"train", "validation", "base_test"})
            require(split in roles, "unsupported split for graph_state")
            require(isinstance(manifest["splits"], dict) and bool(manifest["splits"]),
                    "splits must be a nonempty object")
            require(set(manifest["splits"]).issubset(roles), "split roles do not match graph_state")
            if state == "G1":
                require(set(manifest["splits"]) == roles, "G1 requires both online split roles")
                require(manifest.get("date") == "20160102", "G1 observation date mismatch")
            require(type(manifest["production"]) is bool, "missing production classification")
            expected_status = "CORPUS_GENERATED" if manifest["production"] else "FIXTURE_GENERATED"
            require(manifest["status"] == expected_status, "manifest is not a completed matching corpus")
            for name in ("city", "snapshot_id", "plan_id", "run_id", "method", "encoding_version"):
                require(isinstance(manifest[name], str) and bool(manifest[name]), f"invalid {name}")
            require(manifest["method"] in ("cluster", "uniform_grid", "hierarchical_grid"), "unknown sampler")
            require(manifest["plan_id"] == manifest["run_id"], "plan/run identity mismatch")
            context = manifest.get("route_context", {})
            for name in ("city", "snapshot_id"):
                require(context.get(name, manifest[name]) == manifest[name], f"route context {name} mismatch")
            require(context.get("status", "ok") == "ok", "route context status mismatch")
            self.split = split
            self.shards = manifest["splits"][split]["shards"]
            self.rows = integer(manifest["splits"][split]["rows"], "split rows")
            require(isinstance(self.shards, list), "shards must be a list")
            self._ends: list[int] = []
            paths: set[Path] = set()
            total = 0
            largest_budget = 0
            for shard in self.shards:
                total += integer(shard["rows"], "shard rows", 1)
                self._ends.append(total)
                budget = integer(shard["byte_budget"], "shard byte_budget", 1)
                largest_budget = max(largest_budget, budget)
                image_path = relative_file(self.root, shard["image_path"])
                route_path = relative_file(self.root, shard["route_path"])
                require(image_path not in paths and route_path not in paths and image_path != route_path,
                        "reused shard path")
                paths.update((image_path, route_path))
                route_size = integer(shard["route_bytes"], "route_bytes", 1)
                file_size = integer(shard["bytes"], "shard bytes", 1)
                require(route_path.stat().st_size == route_size and
                        image_path.stat().st_size + route_size == file_size, "shard file bytes mismatch")
            require(total == self.rows, "manifest shard row sum differs from split rows")
            self.cache_bytes = integer(cache_bytes if cache_bytes is not None else largest_budget,
                                       "cache_bytes", 1)
            self.metrics: dict[str, int | float] = {"read_seconds": 0.0, "bytes": 0, "load_count": 0,
                                                  "peak_cache_bytes": 0, "cache_bytes": 0}
            self._cache_index: int | None = None
            self._cache: dict[str, np.ndarray] | None = None
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise CorpusError(f"invalid corpus manifest: {error}") from error

    def __len__(self) -> int:
        return self.rows

    def clear_cache(self) -> None:
        self._cache = None
        self._cache_index = None
        self.metrics["cache_bytes"] = 0

    def _load(self, index: int) -> dict[str, np.ndarray]:
        if self._cache_index == index:
            assert self._cache is not None
            return self._cache
        self.clear_cache()  # Evict before allocating a replacement shard.
        shard = self.shards[index]
        image_path = relative_file(self.root, shard["image_path"])
        route_path = relative_file(self.root, shard["route_path"])
        require(route_path.stat().st_size == shard["route_bytes"] and
                image_path.stat().st_size + route_path.stat().st_size == shard["bytes"],
                "shard changed after manifest validation")
        budget = min(shard["byte_budget"], self.cache_bytes)
        started = time.perf_counter()
        expected_bytes = (_npz_headers(image_path, set(_IMAGE_SCHEMA), budget) +
                          _npz_headers(route_path, set(_ROUTE_SCHEMA), budget))
        require(expected_bytes <= budget, "combined image and route arrays exceed cache/shard budget")
        if "array_bytes" in shard:
            require(shard["array_bytes"] == expected_bytes, "manifest array_bytes mismatch")
        arrays: dict[str, np.ndarray] = {}
        try:
            for file_path in (image_path, route_path):
                with np.load(file_path, allow_pickle=False, max_header_size=10000) as data:
                    for key in data.files:
                        arrays[key] = data[key]
            self._check_arrays(arrays, shard["rows"])
            for key, name, position in (("first_query_id", "query_id", 0), ("last_query_id", "query_id", -1),
                                        ("first_row_index", "row_index", 0), ("last_row_index", "row_index", -1)):
                if key in shard:
                    require(shard[key] == int(arrays[name][position]), f"manifest {key} mismatch")
        except (ValueError, OSError, zipfile.BadZipFile, EOFError) as error:
            if isinstance(error, CorpusError):
                raise
            raise CorpusError(f"cannot read corpus shard: {error}") from error
        actual = sum(array.nbytes for array in arrays.values())
        require(actual == expected_bytes and actual <= budget, "actual cache arrays exceed budget")
        for array in arrays.values():
            array.flags.writeable = False
        self.metrics["read_seconds"] += time.perf_counter() - started
        self.metrics["bytes"] += shard["bytes"]
        self.metrics["load_count"] += 1
        self.metrics["cache_bytes"] = actual
        self.metrics["peak_cache_bytes"] = max(self.metrics["peak_cache_bytes"], actual)
        self._cache, self._cache_index = arrays, index
        return arrays

    def _check_arrays(self, arrays: dict[str, np.ndarray], rows: int) -> None:
        for name, (dtype, tail) in _IMAGE_SCHEMA.items():
            value = arrays[name]
            require(value.dtype.str == dtype and value.shape == (rows, *tail), f"{name} dtype/shape mismatch")
            require(bool(np.isfinite(value).all()), f"nonfinite {name}")
        for name, dtype in _ROUTE_SCHEMA.items():
            value = arrays[name]
            require(value.dtype.str == dtype and value.ndim == 1, f"{name} dtype/shape mismatch")
            if name in _ROUTE_ROWS:
                require(len(value) == rows, f"{name} row count mismatch")
        offsets = arrays["offsets"]
        require(len(offsets) == rows + 1 and offsets[0] == 0 and
                bool((offsets[1:] > offsets[:-1]).all()), "route offsets must delimit nonempty routes")
        require(int(offsets[-1]) == len(arrays["node_ids"]) == len(arrays["arrival_times"]), "ragged lengths mismatch")
        require(bool((arrays["status"] == 0).all()), "non-ok route in completed corpus")
        require(bool((arrays["row_index"] < self.rows).all()), "frozen row_index outside split")
        require(bool(((arrays["departure"] >= 0) & (arrays["departure"] < 86400)).all()), "departure outside day")
        require(bool((arrays["travel_time"] >= 0).all()), "negative route cost")
        first = offsets[:-1].astype(np.intp)
        last = offsets[1:].astype(np.intp) - 1
        require(bool((arrays["node_ids"][first] == arrays["source"]).all()) and
                bool((arrays["node_ids"][last] == arrays["destination"]).all()), "route endpoint mismatch")
        arrival = arrays["arrival_times"]
        require(bool((arrival[first] == arrays["departure"]).all()) and
                bool((arrival[last] >= arrival[first]).all()), "route endpoint time mismatch")
        require(bool((arrival[last] - arrival[first] == arrays["travel_time"]).all()), "travel_time mismatch")
        edge = np.ones(len(arrival), dtype=bool)
        edge[first] = False
        require(bool((arrival[1:][edge[1:]] > arrival[:-1][edge[1:]]).all()), "route arrival times do not increase")

    def get_row(self, physical_index: int, *, include_route: bool = True) -> dict[str, Any]:
        if not isinstance(physical_index, (int, np.integer)) or not 0 <= physical_index < self.rows:
            raise IndexError("physical corpus row index out of range")
        shard_index = bisect_right(self._ends, physical_index)
        start = self._ends[shard_index - 1] if shard_index else 0
        local = physical_index - start
        data = self._load(shard_index)
        result = {name: data[name][local] for name in _IMAGE_SCHEMA}
        result["identity"] = {"physical_index": int(physical_index), "row_index": int(data["row_index"][local]),
                              "query_id": int(data["query_id"][local]), "interval_id": int(data["interval_id"][local]),
                              "city": self.manifest["city"], "snapshot_id": self.manifest["snapshot_id"], "split": self.split, "plan_id": self.manifest.get("plan_id", self.manifest.get("run_id"))}
        if include_route:
            first, end = map(int, data["offsets"][local:local + 2])
            result["route"] = {name: int(data[name][local]) for name in _ROUTE_ROWS}
            result["route"].update(node_ids=data["node_ids"][first:end], arrival_times=data["arrival_times"][first:end])
        return result

    __getitem__ = get_row

    def iter_shards(self) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Yield (physical start, immutable arrays), loading each shard once."""
        for index in range(len(self.shards)):
            yield (self._ends[index - 1] if index else 0), self._load(index)

    def iter_batches(self, batch_size: int) -> Iterator[dict[str, Any]]:
        """Yield up to batch_size rows; batches never cross a shard boundary."""
        integer(batch_size, "batch_size", 1)
        for start, data in self.iter_shards():
            for first in range(0, len(data["arr"]), batch_size):
                end = min(first + batch_size, len(data["arr"]))
                yield {**{name: data[name][first:end] for name in _IMAGE_SCHEMA},
                       "identity": {"physical_index": np.arange(start + first, start + end, dtype=np.uint64),
                                    **{name: data[name][first:end] for name in ("row_index", "query_id", "interval_id")},
                                    "split": self.split, "city": self.manifest["city"],
                                    "snapshot_id": self.manifest["snapshot_id"],
                                    "plan_id": self.manifest.get("plan_id", self.manifest.get("run_id"))}}
            del data  # Release our previous shard views before loading the next one.
