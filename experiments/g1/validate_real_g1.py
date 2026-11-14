"""Four same-clock real G1 point references against full rational Dijkstra."""
from fractions import Fraction
import argparse
import json
from pathlib import Path
import sys
import time
import h5py
import numpy as np
from gt_cache import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run
    config=json.loads((run/'protocol.json').read_text());parent=Path(config['parent_e1_run'])
    sys.path.insert(0,str(parent/'code'))
    from point_data import Native
    from validate_point import oracle
    started,cpu=time.monotonic(),time.process_time();results=[];requested=completed=0
    try:
        for city in ('XA','BJ'):
            folder=run/'data'/city
            assert json.loads((folder/'generation_report.json').read_text())['status']=='GT_QUALIFICATION_AND_SELECTED_LABELS_COMPLETE'
            native=Native(config['remote_root'],city,'G1')
            with np.load(folder/'G1_graph.npz') as graph:
                assert np.array_equal(native.u,graph['u']) and np.array_equal(native.v,graph['v'])
                assert np.array_equal(native.knots,graph['knots'])
            gid=int(np.load(folder/'sdc_Afit_ordinary_gids.npy')[0]);source,destination=divmod(gid,native.n)
            with np.load(folder/'cache_index.npz') as index:
                position=int(np.searchsorted(index['group_ids'],gid));assert int(index['group_ids'][position])==gid
                file=run/str(index['paths'][position]);row=int(index['rows'][position])
            with h5py.File(file,'r') as f:
                count=len(f['group_ids']);assert int(f['group_ids'][row])==gid
                for point,departure in enumerate((43200,44100)):
                    before=time.monotonic();expected=oracle(native,source,departure)[destination]
                    actual_row=row+point*count;offset=f['G1/path_offsets'][actual_row:actual_row+2]
                    cached=f['G1/node_indices'][offset[0]:offset[1]]
                    cached_exact=native.rational_replay(cached,departure)
                    assert cached_exact==expected,(city,gid,departure,'cached witness not exactly optimal')
                    requested+=1
                    atomic_json(run/'audits/G1_rational_reference_progress.json',dict(status='RUNNING',
                        requested_fresh_native_point_paths=requested,completed_fresh_native_point_paths=completed,cases=results))
                    off,nodes,arrival,cost=native.query(source,departure,np.asarray([destination],np.int32));completed+=1
                    exact=native.rational_replay(nodes,departure)
                    assert exact==expected and abs(float(expected)-arrival[-1])<1e-7
                    assert np.array_equal(nodes,cached),'Same native backend must retain the frozen witness independently of target sharing'
                    record=dict(city=city,group_id=gid,source_index=source,destination_index=destination,
                        departure=departure,status='PASSED',cached_and_fresh_witnesses_exactly_optimal=True,
                        target_sharing_does_not_change_witness=True,oracle_arrival_numerator=str(expected.numerator),
                        oracle_arrival_denominator=str(expected.denominator),elapsed_seconds=time.monotonic()-before)
                    results.append(record);print(json.dumps(record),flush=True)
            native.lib.e1_free(native.handle)
        result=dict(status='PASSED',cases=results,independent_exact_reference_points=4,
            additional_fresh_native_point_paths=requested,completed_fresh_native_point_paths=completed,
            elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
            scope='four sampled real G1 full-graph point references; not exhaustive exact-arithmetic optimality of all labels',
            training_labels_unchanged=True,formal_training_updates=0,no_hashes=True)
        atomic_json(run/'audits/G1_real_rational_reference.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        atomic_json(run/'audits/G1_rational_reference_failure.json',dict(status='FAILED',error=repr(exc),
            requested_fresh_native_point_paths=requested,completed_fresh_native_point_paths=completed,
            completed_cases=results,elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu))
        raise


if __name__=='__main__':main()
