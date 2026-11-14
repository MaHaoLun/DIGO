"""Independent rational shortest-point references on frozen real-graph ODs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from point_data import Native
from validate_point import oracle
from p0_inventory import atomic

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True);a=p.parse_args();results=[]
    for city in ("XA","BJ"):
        native=Native(a.root,city);groups=np.load(a.run/"data"/city/"groups.npy")
        for index,departure in ((0,28800),(len(groups)//2,42300)):
            source,destination=map(int,groups[index]);started=time.monotonic()
            truth=oracle(native,source,departure)[destination]
            off,nodes,arr,cost=native.query(source,departure,np.array([destination],np.int32))
            replay=native.rational_replay(nodes,departure)
            assert truth==replay,(city,index,"native witness is not exactly optimal under rational reference")
            assert abs(float(truth)-arr[-1])<1e-7
            r=dict(city=city,group_id=index,departure=departure,status="PASSED",
                exact_optimal_witness=True,oracle_arrival_numerator=str(truth.numerator),
                oracle_arrival_denominator=str(truth.denominator),elapsed_seconds=time.monotonic()-started)
            results.append(r);print(json.dumps(r),flush=True)
    atomic(a.run/"p1/real_rational_point_checks.json",dict(status="PASSED",cases=results,
        scope="four frozen real-graph point references; not exhaustive exact-arithmetic optimality of all labels",
        separate_from_complete_route_and_label_replay=True,no_content_hashes=True))

if __name__=="__main__":main()
