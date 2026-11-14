"""Build first-visit PiT and legal SSE features from accepted raw path witnesses."""
import argparse
import json
import resource
import time
from pathlib import Path
import sys
import h5py
import numpy as np
from common import RUN,E1,E2,parents,roles,atomic,BBOX


def registry(city,snapshot,role):
    if snapshot=='G0':
        from point_data import Dataset
        d=Dataset(E1,city,role);gids=d.group_ids;n=len(d.coords)
        pairs=np.asarray(d.groups[gids]);ppg=16
        tasks=[]
        for s in np.unique(pairs[:,0]):
            indices=np.flatnonzero(pairs[:,0]==s)
            tasks.append((d.folder/'sources'/f's{int(s):07d}.h5',indices,None))
    else:
        sys.path.insert(0,str(E2/'code'));from g1_data import Dataset
        d=Dataset(E2,city,role);gids=d.gids;n=len(d.coords);ppg=2
        pairs=np.column_stack((gids//n,gids%n));tasks=[]
        for name in np.unique(d.paths):
            indices=np.flatnonzero(d.paths==name)
            tasks.append((E2/str(name),indices,d.source_rows[indices]))
    return d,gids,pairs,ppg,tasks


def prepare(city,snapshot,role):
    parents();out=RUN/'data'/city/snapshot/role;out.mkdir(parents=True,exist_ok=True)
    if (out/'acceptance.json').exists():
        print(json.dumps(dict(resumed=True,**json.loads((out/'acceptance.json').read_text()))),flush=True);return
    started=time.monotonic();cpu=time.process_time();d,gids,pairs,ppg,tasks=registry(city,snapshot,role)
    n=len(gids)*ppg;cell_parts=[None]*n;day_parts=[None]*n;prog_parts=[None]*n;seen=np.zeros(n,bool)
    train=snapshot=='G0' and role=='hierarchical_grid_train'
    graph=np.load(E1/'data'/city/'routing_graph.npz')
    keys=graph['u'].astype(np.int64)*len(d.coords)+graph['v']
    freq=np.zeros(len(keys),np.int64);r_sum=np.zeros(4096*4096,np.float64);r_count=np.zeros(4096*4096,np.int64)
    raw_vertices=0;raw_edges=0;revisits=0
    for ti,(file,group_positions,provided_rows) in enumerate(tasks):
        with h5py.File(file,'r') as f:
            assert f.attrs['status']=='COMPLETE'
            allg=f['group_ids'][:];ng=len(allg)
            rows=np.searchsorted(allg,gids[group_positions]) if provided_rows is None else provided_rows
            assert np.array_equal(allg[rows],gids[group_positions])
            raw=f if snapshot=='G0' else f['G1']
            off=raw['path_offsets'][:];nodes=raw['node_indices'][:]
            arrivals=raw['arrival_times_float64' if snapshot=='G0' else 'arrival_times'][:]
            for dep in range(ppg):
                departure=28800+900*dep if ppg==16 else 43200+900*dep
                for gp,row in zip(group_positions,rows+dep*ng):
                    i=int(gp)*ppg+dep;assert not seen[i]
                    sl=slice(off[row],off[row+1]);path=nodes[sl];arr=arrivals[sl]
                    assert path[0]==pairs[gp,0] and path[-1]==pairs[gp,1]
                    assert arr[0]==departure and arr[-1]>departure
                    cells=d.cells[path];unique,first=np.unique(cells,return_index=True)
                    cell_parts[i]=unique.astype(np.uint16)
                    day_parts[i]=(2*np.remainder(arr[first],86400)/86400-1).astype(np.float32)
                    prog_parts[i]=(2*(arr[first]-departure)/(arr[-1]-departure)-1).astype(np.float32)
                    raw_vertices+=len(path);raw_edges+=len(path)-1
                    consecutive=np.r_[True,cells[1:]!=cells[:-1]]
                    revisits+=int(consecutive.sum()>len(unique))
                    if train:
                        edgekeys=path[:-1].astype(np.int64)*len(d.coords)+path[1:]
                        edgeix=np.searchsorted(keys,edgekeys);assert np.array_equal(keys[edgeix],edgekeys)
                        np.add.at(freq,edgeix,1)
                        # A crossing observation is the duration of its road edge.
                        # Region adjacency is induced by actual road crossings.
                        crossing=cells[:-1]!=cells[1:]
                        regionkeys=cells[:-1][crossing].astype(np.int64)*4096+cells[1:][crossing]
                        np.add.at(r_sum,regionkeys,np.diff(arr)[crossing]);np.add.at(r_count,regionkeys,1)
                    seen[i]=True
        if (ti+1)%100==0 or ti+1==len(tasks):
            state=dict(city=city,snapshot=snapshot,role=role,status='PREPARING',files=ti+1,
                total_files=len(tasks),points=int(seen.sum()),required=n,elapsed_seconds=time.monotonic()-started)
            atomic(out/'progress.json',state);print(json.dumps(state),flush=True)
    assert seen.all()
    lengths=np.asarray([len(x) for x in cell_parts],np.int64);offsets=np.r_[0,np.cumsum(lengths)]
    coords=d.coords;box=BBOX[city];lo=np.asarray([box[0],box[2]]);span=np.asarray([box[1]-box[0],box[3]-box[2]])
    cond=np.empty((len(pairs),5),np.float32)
    cond[:,:2]=2*(coords[pairs[:,0],1:]-lo)/span-1;cond[:,2:4]=2*(coords[pairs[:,1],1:]-lo)/span-1
    odt=np.repeat(cond,ppg,axis=0)
    departures=(28800+900*np.arange(ppg)) if ppg==16 else np.asarray([43200,44100])
    odt[:,4]=np.tile(2*departures/86400-1,len(pairs))
    arrays=dict(query_ids=(gids[:,None]*ppg+np.arange(ppg)).reshape(-1),
        pairs=np.repeat(pairs,ppg,axis=0).astype(np.int64),odt=odt,
        endpoints=np.repeat(d.cells[pairs],ppg,axis=0).astype(np.int64),
        offsets=offsets,cells=np.concatenate(cell_parts),daytime=np.concatenate(day_parts),
        progress=np.concatenate(prog_parts),support=d.road_support)
    assert np.all(np.diff(arrays['query_ids'])>0) and np.isfinite(odt).all()
    for name,value in arrays.items():
        with (out/(name+'.tmp.npy')).open('wb') as f:np.save(f,value)
        (out/(name+'.tmp.npy')).replace(out/(name+'.npy'))
    if train:
        feature=RUN/'data'/city/'SSE_features';feature.mkdir(exist_ok=True)
        selected=np.flatnonzero(freq);rsel=np.flatnonzero(r_count)
        assert int(freq.sum())==raw_edges
        np.savez(feature/'train_graphs.npz',node_u=graph['u'][selected],node_v=graph['v'][selected],
            node_frequency=freq[selected],region_u=rsel//4096,region_v=rsel%4096,
            region_ATT=r_sum[rsel]/r_count[rsel],region_count=r_count[rsel])
        atomic(feature/'legal_sources.json',dict(status='TRAIN_ONLY',city=city,role=role,
            points=n,path_edge_observations=int(freq.sum()),node_graph_edges=len(selected),
            region_crossing_observations=int(r_count.sum()),region_graph_edges=len(rsel),
            ATT_definition='mean crossing road-edge duration over accepted G0 training routes only',
            upstream_p_q_1_sampler='released RandomWalker deepwalk branch uses uniform neighbors, ignoring stored weights',
            no_heldout_path_features=True))
    report=dict(status='PREPARED_FULL',city=city,snapshot=snapshot,role=role,points=n,groups=len(gids),
        points_per_group=ppg,raw_vertices=raw_vertices,raw_edges=raw_edges,nonconsecutive_cell_revisit_routes=revisits,
        first_visit_semantics=True,source_files=len(tasks),elapsed_seconds=time.monotonic()-started,
        process_cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        parent_snapshot=str(E1 if snapshot=='G0' else E2),content_hashes=False)
    atomic(out/'acceptance.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--city',choices=['BJ','XA'],required=True)
    p.add_argument('--snapshot',choices=['G0','G1']);p.add_argument('--role');a=p.parse_args()
    for snapshot in ([a.snapshot] if a.snapshot else ['G0','G1']):
        for role in ([a.role] if a.role else roles(snapshot)):prepare(a.city,snapshot,role)

if __name__=='__main__':main()
