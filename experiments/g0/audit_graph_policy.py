"""Reconstruct the canonical FIFO graph policy from original ordered rows."""
import argparse
import json
from pathlib import Path
import numpy as np
from p0_inventory import atomic

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True);a=p.parse_args();results=[]
    for city in ("XA","BJ"):
        folder=a.run/"data"/city;graph=np.load(folder/"routing_graph.npz")
        coords=np.load(folder/"coordinates.npy");n=len(coords)
        source=a.root/"project/digo_dataset_redesign_v1/data/staged"/f"{city}_g0_offline_20260908/snapshot.txt"
        original=np.loadtxt(source,dtype=np.int64,skiprows=1)
        u=np.searchsorted(coords[:,0],original[:,0]);v=np.searchsorted(coords[:,0],original[:,1])
        assert np.array_equal(coords[u,0],original[:,0]) and np.array_equal(coords[v,0],original[:,1])
        canonical=original[u<v,2:];keys=(u*n+v)[u<v]
        order=np.argsort(keys);keys=keys[order];canonical=canonical[order]
        uu,vv,weights=graph["u"],graph["v"],graph["knots"]
        livekeys=uu.astype(np.int64)*n+vv
        assert np.array_equal(livekeys[uu<vv],keys)
        repaired=weights[uu<vv]
        assert np.all(repaired>=canonical)
        changed=np.flatnonzero(np.any(repaired!=canonical,axis=1))
        for i in changed:
            expected=np.array([max(int(canonical[i,j])-900*((k-j)%96) for j in range(96)) for k in range(96)])
            assert np.array_equal(repaired[i],expected)
        reverse=np.searchsorted(livekeys,vv.astype(np.int64)*n+uu)
        assert np.array_equal(livekeys[reverse],vv.astype(np.int64)*n+uu)
        assert np.array_equal(weights[reverse],weights)
        assert np.all(weights-np.roll(weights,-1,axis=1)<=900)
        r=dict(status="PASSED",city=city,input_directed_arcs=len(original),derived_directed_arcs=len(weights),
            canonical_unordered_pairs=len(canonical),least_upward_fifo_majorant_confirmed=True,
            original_canonical_fifo_violations=int((canonical-np.roll(canonical,-1,axis=1)>900).sum()),
            repaired_fifo_violations=0,changed_pairs=len(changed),changed_knots=int((repaired!=canonical).sum()),
            maximum_knot_raise=int((repaired-canonical).max()),sum_knot_raise=int((repaired-canonical).sum()),
            bidirectional_topology_and_shared_weights=True,source_snapshot_preserved=True,no_content_hashes=True)
        atomic(folder/"graph_policy_audit.json",r);results.append(r)
    print(json.dumps(results),flush=True)

if __name__=="__main__":main()
