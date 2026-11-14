"""Freeze model-blind candidate roles and three actual sampler configurations."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from p0_inventory import BBOX, atomic, cell_xy

SPLIT_SEED=20261024
SAMPLER_SEED=20261025

def role_codes(s,d,n):
    # Stateless seeded permutation of unordered node-pair indices. This is a
    # PRNG partition, unrelated to file/content verification.
    with np.errstate(over="ignore"):
        z=np.minimum(s,d).astype(np.uint64)*np.uint64(n)+np.maximum(s,d).astype(np.uint64)
        z += np.uint64(SPLIT_SEED)+np.uint64(0x9e3779b97f4a7c15)
        z=(z^(z>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        z=(z^(z>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        z=z^(z>>np.uint64(31))
    draw=z%np.uint64(1000)
    # train 85%, G0 select/cal 2.5% each, G0 test 5%, G1 test 5%.
    return np.searchsorted([850,875,900,950],draw,side="right").astype(np.int8)

def representatives(coords,box,cell64,cell128):
    rng=np.random.default_rng(SAMPLER_SEED)
    cluster=[]; uniform_s=[]; uniform_t=[]; hierarchical=[]
    xy=coords[:,1:]-[box[0],box[2]]
    for cell in np.unique(cell64):
        ix=np.flatnonzero(cell64==cell)
        # Preserve the legacy Cluster definition at the new 64 coarse scale:
        # k=4, five Lloyd assignment/update iterations, random centers with
        # replacement, first-cluster ties, one random member of largest cluster.
        centers=xy[rng.choice(ix,4,replace=True)].astype(np.float64)
        for _ in range(5):
            dist=((xy[ix,None,:]-centers[None,:,:])**2).sum(2)
            assignment=np.argmin(dist,axis=1)
            members=[ix[assignment==k] for k in range(4)]
            centers=np.array([xy[z].mean(0) if len(z) else xy[rng.choice(ix)] for z in members])
        best=max(range(4),key=lambda k:len(members[k]))
        cluster.append(rng.choice(members[best]))
    # Separate stream avoids consuming Cluster RNG to define Uniform.
    rng=np.random.default_rng(SAMPLER_SEED)
    for cell in np.unique(cell64):
        ix=np.flatnonzero(cell64==cell)
        chosen=rng.permutation(ix)[:4]
        uniform_s.append(chosen[0]);uniform_t.extend(chosen)
    for cell in np.unique(cell128):
        ix=np.flatnonzero(cell128==cell)
        cx,cy=int(cell%128),int(cell//128)
        dx=xy[ix,0]*256-(2*cx+1)*(box[1]-box[0])
        dy=xy[ix,1]*256-(2*cy+1)*(box[3]-box[2])
        score=dx*dx+dy*dy
        hierarchical.append(ix[np.argmin(score)])
    return {"cluster":(np.array(cluster),np.array(cluster)),
        "uniform_grid":(np.array(uniform_s),np.array(uniform_t)),
        "hierarchical_grid":(np.array(hierarchical),np.array(hierarchical))}

def quota(weights,total):
    raw=np.asarray(weights,dtype=np.float64)*total/sum(weights)
    result=np.floor(raw).astype(np.int64)
    order=np.argsort(-(raw-result),kind="stable")
    result[order[:total-int(result.sum())]]+=1
    return result

def candidates(sources,targets,cell32,components,n,exposed):
    chunks=[]
    for s in sources:
        dist=np.abs(cell32[targets]%32-cell32[s]%32)+np.abs(cell32[targets]//32-cell32[s]//32)
        good=(dist>=49)&(dist<=56)&(components[targets]==components[s])
        dest=targets[good]
        if exposed:
            keep=np.fromiter(((min(int(s),int(d)),max(int(s),int(d))) not in exposed
                              for d in dest),dtype=bool,count=len(dest))
            dest=dest[keep]
        src=np.full(len(dest),s,dtype=np.int64)
        roles=role_codes(src,dest,n)
        delta=np.abs(cell32[dest]%32-cell32[s]%32)+np.abs(cell32[dest]//32-cell32[s]//32)
        chunks.append(np.column_stack((src,dest,delta,roles)))
    return np.concatenate(chunks) if chunks else np.empty((0,4),dtype=np.int64)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); summaries=[]
    for city in ("XA","BJ"):
        folder=a.root/"project/digo_dataset_redesign_v1/data/staged"/f"{city}_g0_offline_20260908"
        coords=np.loadtxt(folder/"coordinates.txt",dtype=np.int64,skiprows=1)
        n=len(coords);box=BBOX[city];xy=coords[:,1:]
        x32=cell_xy(xy,box,32);x64=cell_xy(xy,box,64);x128=cell_xy(xy,box,128)
        c32=x32[:,1]*32+x32[:,0];c64=x64[:,1]*64+x64[:,0];c128=x128[:,1]*128+x128[:,0]
        # Actual directed topology; same-SCC capacity is guaranteed reachable.
        ends=np.loadtxt(folder/"snapshot.txt",dtype=np.int64,skiprows=1,usecols=(0,1))
        ei=np.searchsorted(coords[:,0],ends)
        graph=csr_matrix((np.ones(len(ei),np.uint8),(ei[:,0],ei[:,1])),shape=(n,n))
        _,comp=connected_components(graph,directed=True,connection="strong")
        exposed=set();exposure_files=[]
        for path in sorted((a.out/"development_exposure").glob("*.json")):
            qs=json.loads(path.read_text())
            if isinstance(qs,dict): qs=qs.get("queries",qs.get("intervals",[]))
            count=0
            for q in qs:
                if "source" not in q or "destination" not in q: continue
                ids=[int(q["source"]),int(q["destination"])]
                ix=np.searchsorted(coords[:,0],ids)
                if np.any(ix>=n) or not np.array_equal(coords[ix,0],ids):continue
                exposed.add(tuple(sorted(map(int,ix))));count+=1
            exposure_files.append(dict(path=str(path),matching_groups=count))
        inv=json.loads((a.out/f"{city}_G0_inventory.json").read_text())
        train_count=25000 if city=="XA" else 50000
        target=quota(inv["geometric_ordered_64_cell_pairs_by_d32"],train_count)
        pools={};report={}
        for method,(sources,targets) in representatives(coords,box,c64,c128).items():
            pool=candidates(sources,targets,c32,comp,n,exposed)
            train=pool[pool[:,3]==0]
            hist=np.bincount(train[:,2]-49,minlength=8)
            report[method]=dict(source_representatives=len(sources),target_representatives=len(targets),
                legal_unexposed_groups=len(pool),train_role_capacity=len(train),
                train_capacity_by_d32=hist.tolist(),target_by_d32=target.tolist(),
                feasible=bool(np.all(hist>=target)),deficit_by_d32=np.maximum(0,target-hist).tolist())
            pools[method]=train
            np.savez(a.out/f"{city}_{method}_train_candidates.npz",groups=train)
        r=dict(city=city,split_seed=SPLIT_SEED,sampler_seed=SAMPLER_SEED,
            reverse_od_same_role=True,role_probabilities=[.85,.025,.025,.05,.05],
            historical_exposure_unordered_od=len(exposed),exposure_files=exposure_files,
            quota_basis="Hamilton allocation proportional to all occupied ordered 64-cell-pair geometric capacity",
            required_train_groups=train_count,methods=report,
            status="SAMPLER_CAPACITY_PASSED" if all(v["feasible"] for v in report.values()) else "SAMPLER_CAPACITY_FAILED")
        atomic(a.out/f"{city}_sampler_capacity.json",r)
        print(json.dumps(r),flush=True);summaries.append(r)
    atomic(a.out/"sampler_capacity.json",dict(results=summaries,
        status="PASSED" if all(x["status"]=="SAMPLER_CAPACITY_PASSED" for x in summaries) else "FAILED"))

if __name__=="__main__": main()
