"""Full point counts and initial-union directed-edge reduction, no repair."""
import argparse
import json
import time
import numpy as np
from common import RUN,E1,E2,SparseData,parents,atomic,ev


def paths(city,snapshot,method):
    if method=='SSP':
        model=(E1/'training'/f'{city}_hierarchical_grid_seed42') if snapshot=='G0' else (E2/'training'/f'{city}_sdc_mixed_ft_seed42')
    else:model=RUN/'training'/f'{city}_{snapshot}_{method}_seed42'
    return model,model/'test_predictions'


def compute(city,snapshot,method):
    parents();started=time.monotonic();role='base_test' if snapshot=='G0' else 'mixed_test'
    d=SparseData(city,snapshot,role);model,prediction=paths(city,snapshot,method)
    threshold=float(json.loads((model/'calibration.json').read_text())['threshold'])
    scores=np.load(prediction/'scores.npy',mmap_mode='r');qids=np.load(prediction/'query_ids.npy')
    assert scores.shape==(len(d),4096) and scores.dtype==np.float32 and np.array_equal(qids,d.query_ids)
    ppg=d.ppg;groups=len(d)//ppg;counts=np.empty((len(d),3),np.int64);unions=np.zeros((groups,4096),bool)
    gt=d.truth()
    for lo in range(0,len(d),512):
        hi=min(len(d),lo+512);block=scores[lo:hi]
        assert np.isfinite(block).all() and (block>=0).all() and (block<=1).all()
        pred=(block>=threshold)&d.support[None,:];pred[np.arange(hi-lo)[:,None],d.endpoints[lo:hi]]=True
        counts[lo:hi]=ev._count_rows(pred,gt[lo:hi])
        assert lo%ppg==0 and (hi-lo)%ppg==0
        unions[lo//ppg:hi//ppg]=pred.reshape(-1,ppg,4096).any(axis=1)
    source=E1/'data'/city/'routing_graph.npz' if snapshot=='G0' else E2/'data'/city/'G1_graph.npz'
    graph=np.load(source);cells=np.load(E1/'data'/city/'cell64.npy')
    cu,cv=cells[graph['u']],cells[graph['v']]
    keys,multiplicity=np.unique(cu.astype(np.int64)*4096+cv,return_counts=True)
    assert int(multiplicity.sum())==len(cu)
    allowed=np.empty(groups,np.int64)
    for lo in range(0,groups,256):
        part=unions[lo:lo+256];allowed[lo:lo+256]=(part[:,keys//4096]&part[:,keys%4096]).astype(np.int64)@multiplicity
    assert np.all((allowed>=0)&(allowed<=len(cu)))
    result=dict(status='METRICS_COMPUTED_FULL',city=city,snapshot=snapshot,method=method,seed=42,
        points=len(d),groups=groups,departures=ppg,threshold=threshold,**ev._micro(counts),
        edge_reduction_percent=float(np.mean(1-allowed.astype(np.float64)/len(cu))*100),
        allowed_directed_arcs_total=int(allowed.sum()),full_graph_directed_arcs=len(cu),
        unique_cell_pairs=len(keys),mean_initial_cells=float(unions.sum(1).mean()),
        support_and_endpoints=True,repair=False,interval_optimality_claim=False,
        raw_prediction_path=str(prediction),calibration_path=str(model/'calibration.json'),
        elapsed_seconds=time.monotonic()-started,SSP_reuses_accepted_outputs=method=='SSP')
    out=RUN/'artifacts'/f'{city}_{snapshot}_{method}';out.mkdir(parents=True,exist_ok=True)
    np.save(out/'point_counts.npy',counts);np.save(out/'group_allowed_arcs.npy',allowed)
    np.save(out/'initial_unions_packbits.npy',np.packbits(unions,axis=1))
    np.save(out/'query_ids.npy',qids)
    atomic(out/'metrics.json',result);print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--city',choices=['BJ','XA']);p.add_argument('--snapshot',choices=['G0','G1'])
    p.add_argument('--method',choices=['SSE','PiT','SSP'],default='SSP');a=p.parse_args()
    for city in ([a.city] if a.city else ['BJ','XA']):
        for snapshot in ([a.snapshot] if a.snapshot else ['G0','G1']):compute(city,snapshot,a.method)
