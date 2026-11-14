"""Released SSE Node2Vec with explicit epoch parameter and seeded walk streams."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import random
import resource
import sys
import time
import numpy as np
import networkx as nx
from common import RUN,E1,parents,atomic


def walk_stream(args):
    graph,stream,num_walks=args
    seed=int(np.random.SeedSequence([42,stream]).generate_state(1)[0])
    random.seed(seed);np.random.seed(seed)
    sys.path.insert(0,str(RUN/'external_src/JIT-TP/SSE/NodeEmb'))
    from ge.walker import RandomWalker
    walker=RandomWalker(graph,p=1.,q=1.)
    # The released p=q=1 branch is uniform-neighbor DeepWalk. Its alias tables
    # are unused; skipping their construction preserves every walk transition.
    return walker._simulate_walks(list(graph.nodes()),num_walks,30)


def generate(city,kind):
    parents();folder=RUN/'data'/city/'SSE_features';target=folder/(kind+'_embedding.npy')
    receipt=folder/(kind+'_embedding_receipt.json')
    if receipt.exists():print(receipt.read_text(),flush=True);return
    source=json.loads((folder/'legal_sources.json').read_text());assert source['status']=='TRAIN_ONLY'
    graphs=np.load(folder/'train_graphs.npz');G=nx.DiGraph()
    u,v=graphs[kind+'_u'],graphs[kind+'_v']
    weights=graphs['node_frequency' if kind=='node' else 'region_ATT']
    G.add_weighted_edges_from((int(a),int(b),float(w)) for a,b,w in zip(u,v,weights))
    count=len(np.load(E1/'data'/city/'coordinates.npy')) if kind=='node' else 4096
    started=time.monotonic();cpu0=resource.getrusage(resource.RUSAGE_SELF);child0=resource.getrusage(resource.RUSAGE_CHILDREN)
    walk_file=folder/(kind+'_walks.npy')
    # Variable walk lengths are retained. A compact CSR file checkpoints the
    # complete 25-walk-per-node corpus before the ten Word2Vec epochs.
    if not walk_file.exists():
        with ProcessPoolExecutor(max_workers=32) as pool:
            parts=list(pool.map(walk_stream,[(G,i,1) for i in range(25)]))
        walks=[walk for part in parts for walk in part]
        assert len(walks)==25*len(G)
        lengths=np.asarray([len(w) for w in walks],np.int64)
        with walk_file.with_suffix('.tmp.npy').open('wb') as handle:np.save(handle,np.concatenate(walks).astype(np.int32))
        walk_file.with_suffix('.tmp.npy').replace(walk_file)
        np.save(folder/(kind+'_walk_offsets.npy'),np.r_[0,np.cumsum(lengths)])
        del walks,parts
    tokens=np.load(walk_file,mmap_mode='r');off=np.load(folder/(kind+'_walk_offsets.npy'))
    assert len(off)-1==25*len(G) and np.all((np.diff(off)>=1)&(np.diff(off)<=30))
    class Corpus:
        def __iter__(self):
            for lo,hi in zip(off[:-1],off[1:]):yield tokens[lo:hi].tolist()
    from ge.models.node2vec import Node2Vec
    # Use the released train()/get_embeddings(), but supply its actual `epoch`
    # argument. Passing `epochs=10` upstream gets overwritten by default epoch5.
    model=Node2Vec.__new__(Node2Vec);model.graph=G;model.sentences=Corpus();model._embeddings={}
    random.seed(42);np.random.seed(42)
    model.train(embed_size=64,window_size=5,workers=32,epoch=10,seed=42)
    model.w2v_model.save(str(folder/(kind+'_word2vec.model')))
    learned=model.get_embeddings();fallback=np.random.default_rng(42)
    matrix=np.empty((count,64),np.float32);missing=0
    for i in range(count):
        if i in learned:matrix[i]=learned[i]
        else:matrix[i]=fallback.normal(0,1,64);missing+=1
    assert np.isfinite(matrix).all();np.save(target,matrix)
    cpu1=resource.getrusage(resource.RUSAGE_SELF);child1=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=dict(status='PRETRAINED_FULL',city=city,kind=kind,shape=list(matrix.shape),graph_nodes=len(G),
        graph_edges=len(G.edges),walks=len(off)-1,tokens=len(tokens),walk_length=30,num_walks=25,
        epoch_argument=10,word2vec_actual_epochs=model.w2v_model.epochs,workers=32,master_seed=42,
        derived_walk_streams=25,missing_fallback=missing,weights_ignored_by_released_p_q_1_branch=True,
        legal_sources=source,elapsed_seconds=time.monotonic()-started,
        process_cpu_seconds=cpu1.ru_utime+cpu1.ru_stime-cpu0.ru_utime-cpu0.ru_stime,
        walk_child_cpu_seconds=child1.ru_utime+child1.ru_stime-child0.ru_utime-child0.ru_stime,
        threaded_word2vec_not_bitwise_reproducible=True,content_hashes=False)
    assert result['word2vec_actual_epochs']==10
    atomic(receipt,result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--city',choices=['BJ','XA'],required=True);p.add_argument('--kind',choices=['node','region'])
    a=p.parse_args();sys.path.insert(0,str(RUN/'external_src/JIT-TP/SSE/NodeEmb'))
    for kind in ([a.kind] if a.kind else ['node','region']):generate(a.city,kind)

if __name__=='__main__':main()
