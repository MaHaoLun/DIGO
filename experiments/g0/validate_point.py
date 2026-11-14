"""Small mathematical checks; these are not formal experiment rows."""
import argparse
from fractions import Fraction as F
import heapq
import json
from pathlib import Path
import tempfile
import numpy as np
from p0_inventory import atomic
from p0_plan import role_codes
from point_data import Native

def oracle(native,source,departure):
    adjacency=[[] for _ in range(native.n)]
    for u,v,w in zip(native.u,native.v,native.knots):adjacency[u].append((v,w))
    dist={source:F(departure)};heap=[(F(departure),source)]
    while heap:
        t,u=heapq.heappop(heap)
        if t!=dist[u]:continue
        for v,w in adjacency[u]:
            slot=int(t//900)%96;r=t-(t//900)*900
            z=t+int(w[slot])+F(int(w[(slot+1)%96]-w[slot]),900)*r
            if v not in dist or z<dist[v]:dist[v]=z;heapq.heappush(heap,(z,int(v)))
    return dist

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    cases=0;fractional=False
    with tempfile.TemporaryDirectory(dir=a.run) as temp:
        folder=Path(temp);coords=folder/"coordinates.txt";snap=folder/"snapshot.txt"
        coords.write_text("3\n1 0 0\n2 500000 500000\n3 1000000 1000000\n")
        w12=[5]*96;w23=[5+(i%2) for i in range(96)];w13=[11]*96
        with snap.open("w") as f:
            f.write("3 6\n")
            for u,v,w in ((1,2,w12),(1,3,w13),(2,1,w12),(2,3,w23),(3,1,w13),(3,2,w23)):
                f.write(f"{u} {v} "+" ".join(map(str,w))+"\n")
        native=Native(a.root,"XA",snapshot=snap,coordinates=coords,box=(0,1000000,0,1000000))
        for source in range(3):
            for dep in (0,1,899,900,1799,28800,43200,86399):
                targets=np.arange(3,dtype=np.int32)
                off,nodes,arr,cost=native.query(source,dep,targets)
                truth=oracle(native,source,dep)
                for j,target in enumerate(targets):
                    exact=native.rational_replay(nodes[off[j]:off[j+1]],dep)
                    assert exact==truth[target]
                    assert abs(float(exact)-arr[off[j+1]-1])<1e-9
                    fractional |= exact.denominator>1
                    cases+=1
        assert fractional,"Fixture must distinguish continuous labels from per-edge ceil"
    z=np.arange(10000,dtype=np.int64);d=(z*17+3)%10000
    assert np.array_equal(role_codes(z,d,10000),role_codes(d,z,10000))
    report=dict(status="PASSED",formal_points=0,rational_oracle_point_cases=cases,
        fractional_nonceil_case=True,midnight_periodicity=True,reverse_od_roles_equal=True)
    atomic(a.run/"p1"/"continuous_point_checks.json",report);print(json.dumps(report))

if __name__=="__main__":main()
