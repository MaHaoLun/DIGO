"""Shared immutable interfaces for the real quarter-hour revision."""
import json
import os
from pathlib import Path

ROOT = Path(os.environ['DIGO_ROLLING_RUN']).resolve()
PARENT = ROOT.parent
OLD = PARENT / 'rolling_interval_distance_20261028'
BASE = PARENT.parent.parent


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def stamp(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        assert json.loads(path.read_text()) == value, str(path)
    else:
        atomic(path, value)


def frozen_npz(path, **values):
    import numpy as np
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as prior:
            assert set(prior.files) == set(values), str(path)
            for key, value in values.items():
                assert np.array_equal(prior[key], value), (str(path), key)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    tmp.replace(path)
