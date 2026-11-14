"""Native continuous point labels, independent edge replay, sparse SSP64 data."""
import argparse
import ctypes as C
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
import os
from pathlib import Path
import resource
import time

import h5py
import numpy as np
from p0_inventory import BBOX, atomic

class Native:
    def __init__(self,root,city,stage="G0",snapshot=None,coordinates=None,box=None):
        self.lib=C.CDLL(str(Path(__file__).parent/"libe1_point.so"))
        lib=self.lib
        lib.e1_load.argtypes=[C.c_char_p,C.c_char_p,*([C.c_longlong]*4)]
        lib.e1_load.restype=C.c_void_p
        lib.e1_error.restype=C.c_char_p
        for key in ("e1_n","e1_m"):
            getattr(lib,key).argtypes=[C.c_void_p];getattr(lib,key).restype=C.c_int
        lib.e1_free.argtypes=[C.c_void_p]
        lib.e1_edges.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p]
        lib.e1_query.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_int]
        lib.e1_query.restype=C.c_void_p
        lib.e1_length.argtypes=[C.c_void_p];lib.e1_length.restype=C.c_longlong
        lib.e1_copy.argtypes=[C.c_void_p,*([C.c_void_p]*4)]
        lib.e1_result_free.argtypes=[C.c_void_p]
        folder=Path(root)/"project/digo_dataset_redesign_v1/data/staged"/(
            f"{city}_g0_offline_20260908" if stage=="G0" else f"{city}_g1_online_20260911")
        snapshot=Path(snapshot) if snapshot else folder/"snapshot.txt"
        coordinates=Path(coordinates) if coordinates else folder/"coordinates.txt"
        self.handle=lib.e1_load(os.fsencode(snapshot),os.fsencode(coordinates),*(box or BBOX[city]))
        if not self.handle:raise RuntimeError(lib.e1_error().decode())
        self.n=lib.e1_n(self.handle);m=lib.e1_m(self.handle)
        self.u=np.empty(m,np.int32);self.v=np.empty(m,np.int32)
        self.knots=np.empty((m,96),np.int64)
        lib.e1_edges(self.handle,self.u.ctypes.data,self.v.ctypes.data,self.knots.ctypes.data)
        self.keys=self.u.astype(np.int64)*self.n+self.v
        assert np.all(self.keys[1:]>self.keys[:-1])
        assert np.all(self.knots-np.roll(self.knots,-1,axis=1)<=900)

    def query(self,source,departure,destinations):
        targets=np.ascontiguousarray(destinations,dtype=np.int32)
        r=self.lib.e1_query(self.handle,int(source),int(departure),targets.ctypes.data,len(targets))
        if not r:raise RuntimeError(self.lib.e1_error().decode())
        try:
            length=self.lib.e1_length(r)
            offsets=np.empty(len(targets)+1,np.int64);nodes=np.empty(length,np.int32)
            arrivals=np.empty(length,np.float64);costs=np.empty(len(targets),np.float64)
            self.lib.e1_copy(r,offsets.ctypes.data,nodes.ctypes.data,arrivals.ctypes.data,costs.ctypes.data)
            return offsets,nodes,arrivals,costs
        finally:self.lib.e1_result_free(r)

    def replay(self,source,departure,targets,offsets,nodes,arrivals,costs):
        assert np.array_equal(nodes[offsets[:-1]],np.full(len(targets),source))
        assert np.array_equal(nodes[offsets[1:]-1],targets)
        assert np.array_equal(arrivals[offsets[:-1]],np.full(len(targets),departure))
        mask=np.ones(len(nodes)-1,bool);mask[offsets[1:-1]-1]=False
        before=nodes[:-1][mask];after=nodes[1:][mask]
        ix=np.searchsorted(self.keys,before.astype(np.int64)*self.n+after)
        assert np.array_equal(self.keys[ix],before.astype(np.int64)*self.n+after)
        t=arrivals[:-1][mask];actual=arrivals[1:][mask]
        phase=np.remainder(t,86400.);slot=(phase//900).astype(np.int64);r=phase-slot*900
        a=self.knots[ix,slot];b=self.knots[ix,(slot+1)%96]
        expected=t+a+(b-a)*r/900
        maxerr=float(np.max(np.abs(expected-actual),initial=0))
        assert maxerr<1e-7,("edge replay error",maxerr)
        assert np.all(actual>t)
        assert np.max(np.abs(arrivals[offsets[1:]-1]-departure-costs),initial=0)<1e-7
        return maxerr

    def rational_replay(self,nodes,departure):
        t=Fraction(departure)
        for u,v in zip(nodes[:-1],nodes[1:]):
            ix=int(np.searchsorted(self.keys,int(u)*self.n+int(v)))
            assert self.keys[ix]==int(u)*self.n+int(v)
            w=self.knots[ix];slot=int(t//900)%96;r=t-(t//900)*900
            t=t+int(w[slot])+Fraction(int(w[(slot+1)%96]-w[slot]),900)*r
        return t

def source_job(native,folder,source,gids,targets,cells64):
    final=folder/"sources"/f"s{source:07d}.h5"
    if final.exists():
        with h5py.File(final,"r") as f:
            assert f.attrs["status"]=="COMPLETE" and np.array_equal(f["group_ids"][:],gids)
            return dict(source=source,rows=int(f.attrs["rows"]),resumed=True)
    started=time.monotonic();routing=replay=encode=0.;maxerr=0.;sssp=0
    offsets=[0];sparse_offsets=[0];allnodes=[];allarr=[];allcells=[];allprogress=[];allcost=[]
    rational=[]
    for departure in range(28800,43200,900):
        t=time.monotonic();off,nodes,arr,cost=native.query(source,departure,targets)
        routing+=time.monotonic()-t;sssp+=1
        t=time.monotonic();maxerr=max(maxerr,native.replay(source,departure,targets,off,nodes,arr,cost))
        # Independently reconstruct one complete witness per SSSP with rational
        # edge arithmetic. Every emitted route also receives full edge replay.
        exact=native.rational_replay(nodes[off[0]:off[1]],departure)
        assert abs(float(exact)-arr[off[1]-1])<1e-7
        rational.append(dict(departure=departure,numerator=str(exact.numerator),denominator=str(exact.denominator)))
        replay+=time.monotonic()-t;t=time.monotonic()
        route=np.repeat(np.arange(len(targets)),np.diff(off))
        key=route*4096+cells64[nodes]
        unique,back=np.unique(key[::-1],return_index=True)
        last=len(key)-1-back
        counts=np.bincount(unique//4096,minlength=len(targets))
        progress=(2*(arr[last]-departure)/cost[unique//4096]-1).astype(np.float32)
        allcells.append((unique%4096).astype(np.uint16));allprogress.append(progress)
        sparse_offsets.extend((np.cumsum(counts)+sparse_offsets[-1]).tolist())
        offsets.extend((off[1:]+offsets[-1]).tolist())
        allnodes.append(nodes);allarr.append(arr);allcost.append(cost)
        encode+=time.monotonic()-t
    final.parent.mkdir(parents=True,exist_ok=True);tmp=final.with_suffix(".tmp.h5")
    t=time.monotonic()
    with h5py.File(tmp,"w") as f:
        arrays=dict(group_ids=gids,targets=targets,path_offsets=np.array(offsets,np.int64),
            node_indices=np.concatenate(allnodes),arrival_times_float64=np.concatenate(allarr),
            cost_seconds=np.concatenate(allcost),cell_offsets=np.array(sparse_offsets,np.int64),
            cells=np.concatenate(allcells),progress=np.concatenate(allprogress))
        for key,value in arrays.items():f.create_dataset(key,data=value,compression="lzf")
        f.attrs.update(status="COMPLETE",source=int(source),rows=len(targets)*16,
            shared_sssp=sssp,routing_seconds=routing,replay_seconds=replay,encoding_seconds=encode,
            max_edge_replay_error=maxerr,rational_witnesses=json.dumps(rational))
    tmp.replace(final)
    return dict(source=int(source),rows=len(targets)*16,sssp=sssp,routing_seconds=routing,
        replay_seconds=replay,encoding_seconds=encode,write_seconds=time.monotonic()-t,
        elapsed_seconds=time.monotonic()-started,max_edge_replay_error=maxerr)

def generate(root,run,city,workers):
    folder=run/"data"/city
    plan=json.loads((folder/"plan.json").read_text())
    native=Native(root,city)
    np.savez(folder/"routing_graph.npz",u=native.u,v=native.v,knots=native.knots)
    groups=np.load(folder/"groups.npy");cells64=np.load(folder/"cell64.npy")
    unique,start,count=np.unique(groups[:,0],return_index=True,return_counts=True)
    reports=[];started=time.monotonic()
    cpu_started=resource.getrusage(resource.RUSAGE_SELF)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(source_job,native,folder,int(s),np.arange(i,i+k,dtype=np.int64),
                             groups[i:i+k,1].astype(np.int32),cells64) for s,i,k in zip(unique,start,count)]
        for future in as_completed(futures):
            r=future.result();reports.append(r)
            if len(reports)%20==0 or len(reports)==len(futures):
                progress=dict(city=city,status="GENERATING",sources_complete=len(reports),
                    sources_total=len(futures),rows_saved=sum(x["rows"] for x in reports),
                    rows_required=len(groups)*16,elapsed_seconds=time.monotonic()-started)
                atomic(folder/"generation_progress.json",progress);print(json.dumps(progress),flush=True)
    assert sum(x["rows"] for x in reports)==len(groups)*16
    cpu_finished=resource.getrusage(resource.RUSAGE_SELF)
    report=dict(status="GENERATED_FULL",city=city,groups=len(groups),points=len(groups)*16,
        sources=len(unique),sssp_count=len(unique)*16,complete_edge_replay_routes=len(groups)*16,
        rational_complete_path_witnesses=len(unique)*16,
        rational_witness_scope="one actual route per shared SSSP; independent exact arithmetic replay",
        all_paths_reconstructable_from_node_indices_and_integer_knots=True,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        process_cpu_seconds=(cpu_finished.ru_utime+cpu_finished.ru_stime-
                             cpu_started.ru_utime-cpu_started.ru_stime),
        elapsed_seconds=time.monotonic()-started,source_reports=reports,
        corpus_plan=plan,no_content_hashes=True)
    atomic(folder/"generation_report.json",report)

class Dataset:
    """Expose a role without opening any other role's labels."""
    def __init__(self,run,city,role):
        self.folder=Path(run)/"data"/city;self.city=city;self.role=role
        self.group_ids=np.sort(np.load(self.folder/(role+"_group_ids.npy")))
        self.groups=np.load(self.folder/"groups.npy",mmap_mode="r")
        self.coords=np.load(self.folder/"coordinates.npy",mmap_mode="r")
        self.cells=np.load(self.folder/"cell64.npy",mmap_mode="r")
        self.road_support=np.bincount(self.cells,minlength=4096)>0
        self.metadata=dict(city=city,role=role,snapshot_id=city+"_G0_canonical_fifo_continuous",
            rows=len(self.group_ids)*16,groups=len(self.group_ids))
    def __len__(self):return len(self.group_ids)*16
    def iter_batches(self,batch_size=128):
        selected=self.groups[self.group_ids]
        sources,begin,counts=np.unique(selected[:,0],return_index=True,return_counts=True)
        box=BBOX[self.city];lo=np.array([box[0],box[2]]);span=np.array([box[1]-box[0],box[3]-box[2]])
        for source,i,k in zip(sources,begin,counts):
            ids=self.group_ids[i:i+k];pairs=selected[i:i+k]
            with h5py.File(self.folder/"sources"/f"s{source:07d}.h5","r") as f:
                assert f.attrs["status"]=="COMPLETE"
                allg=f["group_ids"][:];positions=np.searchsorted(allg,ids)
                assert np.array_equal(allg[positions],ids)
                off=f["cell_offsets"][:];cell=f["cells"][:];prog=f["progress"][:]
            nt=len(allg)
            for depidx in range(16):
                departure=28800+900*depidx
                for b in range(0,k,batch_size):
                    ends=min(k,b+batch_size);pp=pairs[b:ends];row=positions[b:ends]+depidx*nt
                    image=np.full((len(pp),2,4096),-1.,np.float32)
                    for j,r in enumerate(row):
                        sl=slice(off[r],off[r+1]);image[j,0,cell[sl]]=1.;image[j,1,cell[sl]]=prog[sl]
                    cond=np.empty((len(pp),5),np.float32)
                    cond[:,:2]=2*(self.coords[pp[:,0],1:]-lo)/span-1
                    cond[:,2:4]=2*(self.coords[pp[:,1],1:]-lo)/span-1
                    cond[:,4]=2*departure/86400-1
                    yield dict(image=image.reshape(-1,2,64,64),odt=cond,
                        query_id=ids[b:ends]*16+depidx,group_id=ids[b:ends],
                        endpoints=self.cells[pp],departure=departure)

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True);p.add_argument("--city",choices=["XA","BJ"],required=True)
    p.add_argument("--workers",type=int,default=8);a=p.parse_args()
    generate(a.root,a.run,a.city,a.workers)

if __name__=="__main__":main()
