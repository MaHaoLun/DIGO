"""Full independent first-visit reconstruction and inherited role/count checks."""
import argparse
import json
import time
import h5py
import numpy as np
from common import RUN,E1,E2,SparseData,parents,roles,atomic
from prepare import registry


def validate(city):
    protocol=parents();started=time.monotonic();reports=[]
    e1=json.loads((E1/'code/protocol.json').read_text());e2=json.loads((E2/'protocol.json').read_text())
    for snapshot in ('G0','G1'):
        for role in roles(snapshot):
            cached=SparseData(city,snapshot,role);d,gids,pairs,ppg,tasks=registry(city,snapshot,role)
            expected=(e1['training_points_per_method'][city] if role=='hierarchical_grid_train' else
                e1[{'Vselect':'Vselect_points','Vcal':'Vcal_points','base_test':'base_test_points'}[role]][city]) if snapshot=='G0' else e2['budgets'][city][{'sdc_Afit':'Afit_points_per_FT','Aselect':'Aselect_points','Acal':'Acal_points','mixed_test':'mixed_test_points'}[role]]
            assert len(cached)==expected
            assert np.array_equal(cached.query_ids,(gids[:,None]*ppg+np.arange(ppg)).reshape(-1))
            assert np.array_equal(cached.pairs,np.repeat(pairs,ppg,axis=0))
            assert np.array_equal(cached.endpoints,np.repeat(d.cells[pairs],ppg,axis=0))
            seen=np.zeros(len(cached),bool);vertices=0;maxerr=0.
            for file,group_positions,provided_rows in tasks:
                with h5py.File(file,'r') as f:
                    allg=f['group_ids'][:];rows=np.searchsorted(allg,gids[group_positions]) if provided_rows is None else provided_rows
                    assert np.array_equal(allg[rows],gids[group_positions])
                    raw=f if snapshot=='G0' else f['G1'];off=raw['path_offsets'][:];nodes=raw['node_indices'][:]
                    arrival=raw['arrival_times_float64' if snapshot=='G0' else 'arrival_times'][:]
                    for dep in range(ppg):
                        departure=28800+900*dep if ppg==16 else 43200+900*dep
                        for gp,row in zip(group_positions,rows+dep*len(allg)):
                            index=int(gp)*ppg+dep;assert not seen[index]
                            sl=slice(off[row],off[row+1]);path=nodes[sl];times=arrival[sl]
                            first={}
                            for node,t in zip(path,times):
                                c=int(d.cells[node])
                                if c not in first:first[c]=float(t)
                            cells=np.asarray(sorted(first),np.uint16);t=np.asarray([first[int(c)] for c in cells])
                            st=slice(cached.offsets[index],cached.offsets[index+1])
                            assert np.array_equal(cached.cells[st],cells)
                            day=(2*(t%86400)/86400-1).astype(np.float32)
                            progress=(2*(t-departure)/(float(times[-1])-departure)-1).astype(np.float32)
                            assert np.array_equal(cached.daytime[st],day) and np.array_equal(cached.progress[st],progress)
                            assert set(map(int,cells))==set(map(int,d.cells[path]))
                            vertices+=len(path);seen[index]=True
            assert seen.all()
            reports.append(dict(snapshot=snapshot,role=role,points=expected,groups=len(gids),
                independently_reconstructed_first_visits=expected,raw_vertices=vertices,occupancy_matches_raw_paths=True,
                exact_float32_first_visit_channels=True))
            print(json.dumps(reports[-1]),flush=True)
    feature=RUN/'data'/city/'SSE_features'
    legal=json.loads((feature/'legal_sources.json').read_text())
    assert legal['role']=='hierarchical_grid_train' and legal['points']==e1['training_points_per_method'][city]
    graph=np.load(feature/'train_graphs.npz');assert int(graph['node_frequency'].sum())==legal['path_edge_observations']
    result=dict(status='PASSED',city=city,roles=reports,complete_roles=8,
        points=sum(x['points'] for x in reports),SSE_train_only_features=legal,
        elapsed_seconds=time.monotonic()-started,content_hashes=False)
    atomic(RUN/'audits'/f'{city}_prepared.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--city',choices=['BJ','XA'],required=True);validate(p.parse_args().city)
