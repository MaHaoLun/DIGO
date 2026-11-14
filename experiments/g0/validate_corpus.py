"""Independent full-file audit of emitted paths, labels, counts and identities."""
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
from p0_inventory import atomic

def validate(run,city):
    started=time.monotonic();folder=run/"data"/city
    plan=json.loads((folder/"plan.json").read_text())
    groups=np.load(folder/"groups.npy");coords=np.load(folder/"coordinates.npy");cell64=np.load(folder/"cell64.npy")
    graph=np.load(folder/"routing_graph.npz");u=graph["u"];v=graph["v"];w=graph["knots"]
    n=len(coords);keys=u.astype(np.int64)*n+v
    seen=np.zeros(len(groups),bool);routes=0;max_edge_error=0.;max_progress_error=0.;source_count=0
    for path in sorted((folder/"sources").glob("s*.h5")):
        with h5py.File(path,"r") as f:
            assert f.attrs["status"]=="COMPLETE"
            source=int(f.attrs["source"]);gids=f["group_ids"][:];targets=f["targets"][:]
            assert not seen[gids].any() and np.all(groups[gids,0]==source)
            assert np.array_equal(groups[gids,1],targets)
            seen[gids]=True;nr=len(gids)*16
            po=f["path_offsets"][:];nodes=f["node_indices"][:];arr=f["arrival_times_float64"][:]
            co=f["cell_offsets"][:];cells=f["cells"][:];progress=f["progress"][:];cost=f["cost_seconds"][:]
        assert len(po)==nr+1 and len(co)==nr+1 and po[0]==co[0]==0
        assert po[-1]==len(nodes)==len(arr) and co[-1]==len(cells)==len(progress)
        assert np.all(np.diff(po)>0) and np.all(np.diff(co)>0)
        dep=np.repeat(np.arange(28800,43200,900),len(gids))
        dst=np.tile(targets,16)
        assert np.all(nodes[po[:-1]]==source) and np.array_equal(nodes[po[1:]-1],dst)
        assert np.array_equal(arr[po[:-1]],dep)
        assert np.max(np.abs(arr[po[1:]-1]-dep-cost),initial=0)<1e-7
        transitions=np.ones(len(nodes)-1,bool);transitions[po[1:-1]-1]=False
        before=nodes[:-1][transitions];after=nodes[1:][transitions]
        ix=np.searchsorted(keys,before.astype(np.int64)*n+after)
        assert np.array_equal(keys[ix],before.astype(np.int64)*n+after)
        entry=arr[:-1][transitions];actual=arr[1:][transitions]
        local=np.mod(entry,86400);slot=(local/900).astype(np.int64)
        alpha=(local-slot*900)/900
        travel=(1-alpha)*w[ix,slot]+alpha*w[ix,(slot+1)%96]
        edge_error=float(np.max(np.abs(actual-entry-travel),initial=0));max_edge_error=max(max_edge_error,edge_error)
        assert edge_error<1e-7 and np.all(actual>entry)
        # Independent last-visit reconstruction, from raw paths rather than
        # relying on any encoder-produced grouping or progress fields.
        for row in range(nr):
            visit=cell64[nodes[po[row]:po[row+1]]]
            uniq,rev=np.unique(visit[::-1],return_index=True)
            last=len(visit)-1-rev
            expected=2*(arr[po[row]+last]-dep[row])/cost[row]-1
            sl=slice(co[row],co[row+1])
            assert np.array_equal(cells[sl],uniq)
            error=float(np.max(np.abs(progress[sl]-expected),initial=0))
            assert error<1e-6 and np.all(np.isfinite(progress[sl]))
            max_progress_error=max(max_progress_error,error)
        routes+=nr;source_count+=1
        if source_count%200==0:
            print(json.dumps(dict(city=city,audited_sources=source_count,audited_routes=routes)),flush=True)
    assert seen.all() and routes==plan["points_to_label"]
    for role,expected in plan["sets"].items():
        ids=np.load(folder/(role+"_group_ids.npy"))
        assert len(ids)*16==expected["points"] and len(np.unique(ids))==len(ids) and seen[ids].all()
    report=dict(status="PASSED",city=city,points=routes,groups=len(groups),source_files=source_count,
        all_source_files_complete=True,all_routes_edge_replayed=True,all_native64_labels_reconstructed=True,
        all_role_counts_complete=True,max_edge_replay_error=max_edge_error,max_progress_error=max_progress_error,
        elapsed_seconds=time.monotonic()-started,no_content_hashes=True)
    atomic(folder/"independent_acceptance.json",report);print(json.dumps(report),flush=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--city",choices=["XA","BJ"],required=True);a=p.parse_args();validate(a.run,a.city)

if __name__=="__main__":main()
