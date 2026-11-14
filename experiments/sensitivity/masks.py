"""Score-qualified one-cell expansion on the 64 x 64 road grid."""
import math
import time

def expand_mask(scores, support, endpoints, base, guard):
    """Once-only complete additional CPU fee, excluding file I/O and storage."""
    import numpy as np
    from scipy.ndimage import binary_dilation
    start = time.perf_counter()
    assert scores.dtype == np.float32 and scores.shape in ((1, 4096), (16, 4096))
    assert np.isfinite(scores).all() and support.dtype == np.bool_ and support.shape == (4096,)
    assert len(endpoints) == 2 and all(type(value) is int and 0 <= value < 4096 for value in endpoints)
    assert support[endpoints].all() and math.isfinite(base) and math.isfinite(guard)
    score64 = scores.astype(np.float64)
    raw = (score64 >= base) & support
    raw[:, endpoints] = True
    square = binary_dilation(raw.reshape(-1, 64, 64), structure=np.ones((1, 3, 3), bool),
                             iterations=1, border_value=0).reshape(-1, 4096) & support
    thin = raw | (square & (score64 >= guard))
    raw_union, thin_union = raw.any(axis=0), thin.any(axis=0)
    packed = np.packbits(np.stack((raw_union, thin_union)), axis=1, bitorder='big')
    payload = packed.tobytes(order='C')
    wire = np.frombuffer(payload, np.uint8).reshape(2, 512)
    received = np.unpackbits(wire, axis=1, bitorder='big').copy()
    packed_copy = wire.copy()
    assert len(payload) == 1024 and np.array_equal(received[0], raw_union)
    assert np.array_equal(received[1], thin_union) and not (received[0] & (1-received[1])).any()
    elapsed = time.perf_counter()-start
    return packed_copy, elapsed


