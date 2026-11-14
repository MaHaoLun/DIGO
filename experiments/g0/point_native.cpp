// Continuous point labels share the migrated canonical FIFO graph loader.
#include "corpus/graph.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <vector>

using corpus::Graph;
struct Result { std::vector<long long> offsets; std::vector<int> nodes;
                std::vector<double> arrivals; std::vector<double> costs; };
static thread_local std::string error;
extern "C" const char* e1_error(){return error.c_str();}
extern "C" void* e1_load(const char* snapshot,const char* coordinates,
                          long long xmin,long long xmax,long long ymin,long long ymax){
    try{return new Graph(Graph::load(snapshot,coordinates,{xmin,xmax,ymin,ymax},corpus::RoutingPolicy::CanonicalFifoLpf));}
    catch(const std::exception& e){error=e.what();return nullptr;}
}
extern "C" void e1_free(void* p){delete static_cast<Graph*>(p);}
extern "C" int e1_n(void* p){return static_cast<Graph*>(p)->nodes.size();}
extern "C" int e1_m(void* p){int n=0;for(auto& a:static_cast<Graph*>(p)->adjacency)n+=a.size();return n;}
extern "C" void e1_edges(void* p,int* u,int* v,long long* knots){
    auto& g=*static_cast<Graph*>(p);int k=0;
    for(size_t i=0;i<g.nodes.size();++i)for(const auto& edge:g.adjacency[i]){
        u[k]=i;v[k]=edge.to;for(int j=0;j<96;++j)knots[k*96+j]=edge.weights[j];++k;
    }
}
static long double arrive(const corpus::Edge& edge,long double t){
    const long double phase=std::fmod(t,86400.L);
    const int slot=static_cast<int>(phase/900.L);
    const long double r=phase-slot*900.L;
    return t+edge.weights[slot]+(edge.weights[(slot+1)%96]-edge.weights[slot])*r/900.L;
}
extern "C" void* e1_query(void* pointer,int source,int departure,const int* targets,int nt){
    try{
        const auto& g=*static_cast<Graph*>(pointer);const size_t n=g.nodes.size();
        if(source<0||source>=int(n)||departure<0||departure>=86400||nt<1)throw std::runtime_error("bad query");
        std::vector<long double> dist(n,std::numeric_limits<long double>::infinity());
        std::vector<int> parent(n,-1);std::vector<bool> settled(n,false),wanted(n,false);
        int remaining=0;
        for(int j=0;j<nt;++j){if(targets[j]<0||targets[j]>=int(n))throw std::runtime_error("bad destination");
            if(!wanted[targets[j]]){wanted[targets[j]]=true;++remaining;}}
        using State=std::pair<long double,int>;
        std::priority_queue<State,std::vector<State>,std::greater<State>> heap;
        dist[source]=departure;heap.push({dist[source],source});
        while(!heap.empty()){
            auto [t,u]=heap.top();heap.pop();if(settled[u]||t!=dist[u])continue;
            settled[u]=true;if(wanted[u]&&--remaining==0)break;
            for(const auto& edge:g.adjacency[u]){
                int v=edge.to;if(settled[v])continue;long double candidate=arrive(edge,t);
                if(candidate<dist[v]){dist[v]=candidate;parent[v]=u;heap.push({candidate,v});}
            }
        }
        auto* r=new Result;r->offsets.push_back(0);
        for(int j=0;j<nt;++j){
            const int d=targets[j];if(!settled[d]){delete r;throw std::runtime_error("unreachable frozen target");}
            std::vector<int> path;int v=d;
            while(v!=source){if(v<0||path.size()>=n){delete r;throw std::runtime_error("bad predecessor chain");}
                path.push_back(v);v=parent[v];}path.push_back(source);std::reverse(path.begin(),path.end());
            for(int node:path){r->nodes.push_back(node);r->arrivals.push_back(static_cast<double>(dist[node]));}
            r->offsets.push_back(r->nodes.size());r->costs.push_back(static_cast<double>(dist[d]-departure));
        }
        return r;
    }catch(const std::exception& e){error=e.what();return nullptr;}
}
extern "C" long long e1_length(void* p){return static_cast<Result*>(p)->nodes.size();}
extern "C" void e1_copy(void* p,long long* offsets,int* nodes,double* arrivals,double* costs){
    auto& r=*static_cast<Result*>(p);std::copy(r.offsets.begin(),r.offsets.end(),offsets);
    std::copy(r.nodes.begin(),r.nodes.end(),nodes);std::copy(r.arrivals.begin(),r.arrivals.end(),arrivals);
    std::copy(r.costs.begin(),r.costs.end(),costs);
}
extern "C" void e1_result_free(void* p){delete static_cast<Result*>(p);}
