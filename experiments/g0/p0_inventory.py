"""Inspect actual inputs and capacity without content hashes."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

BBOX = {"BJ": (116200000,116500000,39800000,40100000),
        "XA": (108920000,109010000,34200000,34280000)}

def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)

def cell_xy(xy, box, dim):
    low = np.array([box[0], box[2]], dtype=np.int64)
    span = np.array([box[1]-box[0], box[3]-box[2]], dtype=np.int64)
    return np.minimum(dim-1, (xy-low)*dim//span)

def count_pairs(cells32, components, indices):
    # Count exact directed node pairs within SCCs. Cross-SCC reachability is
    # deliberately excluded here and identified as a lower bound.
    total = np.zeros(8, dtype=np.int64)
    for component in np.unique(components[indices]):
        z = indices[components[indices] == component]
        counts = np.bincount(cells32[z], minlength=1024).astype(np.int64)
        nonempty = np.flatnonzero(counts)
        for sc in nonempty:
            distances = np.abs(nonempty%32-sc%32)+np.abs(nonempty//32-sc//32)
            good = (distances >= 49) & (distances <= 56)
            np.add.at(total, distances[good]-49, counts[sc]*counts[nonempty[good]])
    return total.tolist()

def inspect(root, city, stage):
    started = time.monotonic()
    name = f"{city}_g0_offline_20260908" if stage == "G0" else f"{city}_g1_online_20260911"
    folder = root/"project/digo_dataset_redesign_v1/data/staged"/name
    cp, sp = folder/"coordinates.txt", folder/"snapshot.txt"
    coords = np.loadtxt(cp, dtype=np.int64, skiprows=1)
    with sp.open() as f:
        declared = list(map(int, next(f).split()))
        rows = np.loadtxt(f, dtype=np.int64)
    n = len(coords)
    assert len(np.unique(coords[:,0])) == n
    assert list(coords[:,0]) == sorted(coords[:,0].tolist())
    assert rows.shape == (declared[1],98) and declared[0] == n
    edge_indices = np.searchsorted(coords[:,0], rows[:,:2])
    assert np.array_equal(coords[edge_indices,0],rows[:,:2])
    arcs = csr_matrix((np.ones(len(rows),dtype=np.uint8),
                       (edge_indices[:,0],edge_indices[:,1])), shape=(n,n))
    nc, labels = connected_components(arcs, directed=True, connection="strong")
    sizes = np.bincount(labels)
    box = BBOX[city]
    xy = coords[:,1:]
    assert ((xy >= [box[0],box[2]]) & (xy <= [box[1],box[3]])).all()
    grid32 = cell_xy(xy,box,32)
    grid64 = cell_xy(xy,box,64)
    grid128 = cell_xy(xy,box,128)
    assert np.array_equal(grid64//2,grid32)
    assert np.array_equal(grid128//2,grid64)
    cell32 = grid32[:,1]*32+grid32[:,0]
    cell64 = grid64[:,1]*64+grid64[:,0]
    cell128 = grid128[:,1]*128+grid128[:,0]
    # One node per occupied 64 cell is an upper bound for the historical
    # Cluster's representative count. Actual representatives are checked next.
    occupied64 = np.unique(cell64)
    all64_counts = np.bincount(cell64,minlength=4096)
    pair_hist = np.zeros(8,dtype=np.int64)
    for c in occupied64:
        d = np.abs(occupied64%64//2-c%64//2)+np.abs(occupied64//64//2-c//64//2)
        good=(d>=49)&(d<=56)
        pair_hist += np.bincount(d[good]-49,minlength=8)
    hist = count_pairs(cell32,labels,np.arange(n))
    weights = rows[:,2:]
    differences = weights-np.roll(weights,-1,axis=1)
    report = dict(city=city,stage=stage,paths={"coordinates":str(cp),"snapshot":str(sp)},
        file_metadata={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                       for p in (cp,sp)},
        nodes=n,input_directed_rows=len(rows),unique_directed_arcs=int(arcs.nnz),
        strongly_connected_components=int(nc),largest_scc_nodes=int(sizes.max()),
        occupied_cells_32=int(len(np.unique(cell32))),occupied_cells_64=len(occupied64),
        occupied_cells_128=int(len(np.unique(cell128))),grid_parent_consistency=True,
        fifo_pwl_violations=int((differences>900).sum()),
        minimum_knot_weight=int(weights.min()),maximum_knot_drop=int(differences.max()),
        legal_directed_node_od_same_scc_by_d32=hist,
        legal_directed_node_od_same_scc_total=sum(hist),
        geometric_ordered_64_cell_pairs_by_d32=pair_hist.tolist(),
        geometric_ordered_64_cell_pairs_total=int(pair_hist.sum()),
        all_scc_node_pairs_are_reachable=True,
        cross_scc_od_excluded_from_node_capacity_lower_bound=True,
        elapsed_seconds=time.monotonic()-started)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    results=[]
    for city in ("XA","BJ"):
        for stage in ("G0","G1"):
            r=inspect(a.root,city,stage)
            results.append(r)
            atomic(a.out/f"{city}_{stage}_inventory.json",r)
            print(json.dumps(r),flush=True)
    atomic(a.out/"input_inventory.json",dict(status="INPUT_INSPECTED_CAPACITY_NOT_YET_PASSED",
        no_content_hashes=True,results=results))

if __name__ == "__main__":
    main()
