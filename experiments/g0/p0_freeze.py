"""Complete the common domain denominator and freeze E1 OD selections."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from p0_inventory import BBOX, atomic, cell_xy
from p0_plan import role_codes, quota, SAMPLER_SEED

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    for city in ("XA","BJ"):
        out=a.run/"data"/city;out.mkdir(parents=True,exist_ok=True)
        assert not (out/"plan.json").exists(),"Never overwrite a frozen plan"
        cap=json.loads((a.run/"p0"/f"{city}_sampler_capacity.json").read_text())
        assert cap["status"]=="SAMPLER_CAPACITY_PASSED"
        folder=a.root/"project/digo_dataset_redesign_v1/data/staged"/f"{city}_g0_offline_20260908"
        coords=np.loadtxt(folder/"coordinates.txt",dtype=np.int64,skiprows=1)
        ends=np.loadtxt(folder/"snapshot.txt",dtype=np.int64,skiprows=1,usecols=(0,1))
        ei=np.searchsorted(coords[:,0],ends);n=len(coords)
        arcs=csr_matrix((np.ones(len(ei),np.uint8),(ei[:,0],ei[:,1])),shape=(n,n))
        assert (arcs!=arcs.T).nnz==0,"SCC exclusion is exact only for reciprocal topology"
        _,comp=connected_components(arcs,directed=True,connection="strong")
        xy32=cell_xy(coords[:,1:],BBOX[city],32);xy64=cell_xy(coords[:,1:],BBOX[city],64)
        c32=xy32[:,1]*32+xy32[:,0];c64=xy64[:,1]*64+xy64[:,0]
        buckets=[np.flatnonzero(c32==c) for c in range(1024)]
        exposed=set()
        for f in cap["exposure_files"]:
            qs=json.loads(Path(f["path"]).read_text())
            for q in qs:
                if "source" not in q or "destination" not in q:continue
                ids=[int(q["source"]),int(q["destination"])];ix=np.searchsorted(coords[:,0],ids)
                if np.any(ix>=n) or not np.array_equal(coords[ix,0],ids):continue
                exposed.add(tuple(sorted(map(int,ix))))
        held={1:[],2:[],3:[],4:[]};den=np.zeros(4096*4096,dtype=bool)
        counts=np.zeros((5,8),np.int64)
        for sc in range(1024):
            near=[d for d in range(1024) if 49<=abs(d%32-sc%32)+abs(d//32-sc//32)<=56]
            targets=np.concatenate([buckets[d] for d in near]) if near else np.array([],np.int64)
            if not len(targets):continue
            for s in buckets[sc]:
                d=targets[comp[targets]==comp[s]]
                if exposed:
                    keep=np.fromiter(((min(int(s),int(z)),max(int(s),int(z))) not in exposed
                                      for z in d),bool,count=len(d));d=d[keep]
                src=np.full(len(d),s,np.int64);r=role_codes(src,d,n)
                dist=np.abs(c32[d]%32-sc%32)+np.abs(c32[d]//32-sc//32)
                np.add.at(counts,(r,dist-49),1)
                den[c64[s]*4096+c64[d[r==0]]]=True
                for role in held:
                    dest=d[r==role]
                    if len(dest):held[role].append(np.column_stack((np.full(len(dest),s),dest,dist[r==role])))
            if sc%128==0:print(city,"coarse cell",sc,flush=True)
        rng=np.random.default_rng(SAMPLER_SEED)
        weights=json.loads((a.run/"p0"/f"{city}_G0_inventory.json").read_text())["geometric_ordered_64_cell_pairs_by_d32"]
        wanted={"Vselect":(1,1562 if city=="XA" else 3125),
                "Vcal":(2,1563 if city=="XA" else 3125),
                "base_test":(3,3125 if city=="XA" else 6250)}
        selected={};summary={}
        for name,(role,total) in wanted.items():
            pool=np.concatenate(held[role]);target=quota(weights,total);parts=[]
            for j,k in enumerate(target):
                eligible=pool[pool[:,2]==49+j];assert len(eligible)>=k
                parts.append(eligible[rng.permutation(len(eligible))[:k],:2])
            selected[name]=np.concatenate(parts)
        for method in cap["methods"]:
            pool=np.load(a.run/"p0"/f"{city}_{method}_train_candidates.npz")["groups"]
            target=np.array(cap["methods"][method]["target_by_d32"]);parts=[]
            rng_method=np.random.default_rng(SAMPLER_SEED+1)
            for j,k in enumerate(target):
                eligible=pool[pool[:,2]==49+j];assert len(eligible)>=k
                parts.append(eligible[rng_method.permutation(len(eligible))[:k],:2])
            selected[method+"_train"]=np.concatenate(parts)
        combined=np.unique(np.concatenate(list(selected.values())),axis=0)
        codes=combined[:,0]*n+combined[:,1]
        assert np.all(codes[1:]>codes[:-1])
        for name,pairs in selected.items():
            ids=np.searchsorted(codes,pairs[:,0]*n+pairs[:,1])
            assert np.array_equal(combined[ids],pairs)
            np.save(out/(name+"_group_ids.npy"),ids)
            cells=c64[pairs[:,0]]*4096+c64[pairs[:,1]]
            summary[name]=dict(groups=len(pairs),points=len(pairs)*16,
                unique_directed_node_od=len(np.unique(pairs,axis=0)),
                ordered_64_cell_pairs=len(np.unique(cells)),
                od_coverage_percent=100*len(np.unique(cells))/int(den.sum()) if name.endswith("_train") else None,
                d32_histogram=np.bincount(np.abs(c32[pairs[:,0]]%32-c32[pairs[:,1]]%32)+
                    np.abs(c32[pairs[:,0]]//32-c32[pairs[:,1]]//32)-49,minlength=8).tolist())
        np.save(out/"groups.npy",combined)
        np.save(out/"coordinates.npy",coords)
        np.save(out/"cell64.npy",c64)
        np.save(out/"train_eligible_64_cell_pairs.npy",np.flatnonzero(den))
        r=dict(status="E1_COMMON_HOLDOUT_AND_TRAIN_PLANS_FROZEN",city=city,
            groups_to_label=len(combined),points_to_label=len(combined)*16,
            common_train_eligible_ordered_64_cell_pairs=int(den.sum()),
            candidate_role_capacity_by_d32=counts.tolist(),
            future_g1_test_role_reserved_groups=int(counts[4].sum()),
            topology_reciprocal=True,scc_reachability_is_exact=True,
            grid64=True,no_hash_verification=True,sets=summary,
            label_semantics="canonical_pair_periodic_fifo_continuous_pwl_no_ceil_v1",
            arithmetic="native long double labels; path and knots permit independent rational reconstruction",
            departures=list(range(28800,43200,900)))
        atomic(out/"plan.json",r);print(json.dumps(r),flush=True)

if __name__=="__main__":main()
