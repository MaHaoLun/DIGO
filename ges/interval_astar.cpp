#include "pwl.hpp"
#include <array>

namespace continuous_interval_astar_v1 {
constexpr long long INF=std::numeric_limits<long long>::max()/4;
struct Stats {
 std::vector<unsigned char> seen;
 std::array<long long,4> scans{},unique{};
 std::array<double,4> seconds{};
 long long compositions=0,envelopes=0,minima=0,pops=0,reopens=0,stale=0,updates=0,pruned=0,max_pieces=0;
 explicit Stats(size_t m):seen(m){}
 void scan(int e,int stage){++scans[stage];unsigned char bit=1<<stage;if(!(seen[e]&bit)){seen[e]|=bit;++unique[stage];}}
};
struct Handle {Graph g;std::string output;explicit Handle(const char*p):g(p){
 long long largest=0;for(const auto&e:g.edges)largest=std::max(largest,e.low);
 need((__int128)largest*g.ids.size()<INF,"unsupported static distance range");
}};
struct Bounds {std::vector<long long>d;std::vector<int>edge;};
static Bounds bounds(const Graph&g,int destination,Stats&z){
 Bounds b{std::vector<long long>(g.ids.size(),INF),std::vector<int>(g.ids.size(),-1)};
 using Entry=std::pair<long long,int>;std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>>q;
 b.d[destination]=0;q.emplace(0,destination);
 while(!q.empty()){
  auto [dist,v]=q.top();q.pop();if(dist!=b.d[v])continue;
  for(int e:g.in[v]){z.scan(e,0);const auto&a=g.edges[e];
   if(dist+a.low<b.d[a.u]){b.d[a.u]=dist+a.low;b.edge[a.u]=e;q.emplace(b.d[a.u],a.u);}}
 }return b;
}
struct Result {bool found=false;Q cost,departure;std::vector<int>edges;std::vector<Q>arrivals;
 std::string certificate;bool has_remaining_key=false;Q remaining_key;};
static Result recover(const Graph&g,int source,int destination,const Q&departure,Stats&z){
 std::vector<Q>arrival(g.ids.size());std::vector<unsigned char>finite(g.ids.size());std::vector<int>pred(g.ids.size(),-1);
 using Entry=std::pair<Q,int>;std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>>q;
 finite[source]=1;arrival[source]=departure;q.emplace(departure,source);
 while(!q.empty()){
  auto [at,u]=q.top();q.pop();if(at!=arrival[u])continue;if(u==destination)break;
  for(int e:g.out[u]){z.scan(e,3);int v=g.edges[e].v;Q next=g.arrival(e,at);
   if(!finite[v]||next<arrival[v]){finite[v]=1;arrival[v]=next;pred[v]=e;q.emplace(next,v);}}
 }
 Result r;if(!finite[destination])return r;r.found=true;r.departure=departure;r.cost=arrival[destination]-departure;
 for(int v=destination;v!=source;){int e=pred[v];need(e>=0,"missing recovery predecessor");r.edges.push_back(e);v=g.edges[e].u;need(r.edges.size()<=g.ids.size(),"recovery cycle");}
 std::reverse(r.edges.begin(),r.edges.end());r.arrivals={departure};
 for(int e:r.edges){z.scan(e,3);r.arrivals.push_back(g.arrival(e,r.arrivals.back()));}
 need(r.arrivals.back()-departure==r.cost,"replay differs from point cost");return r;
}
static Result solve(const Graph&g,int source,int destination,const Q&lo,const Q&hi,Stats&z){
 auto begin=Clock::now();auto h=bounds(g,destination,z);
 z.seconds[0]=std::chrono::duration<double>(Clock::now()-begin).count();
 if(h.d[source]==INF){Result r;r.certificate="full_graph_topologically_unreachable";return r;}
 begin=Clock::now();P seed{{lo,hi,1,0}};size_t length=0;
 for(int u=source;u!=destination;){int e=h.edge[u];need(e>=0,"missing seed predecessor");
  z.scan(e,1);seed=g.compose(seed,e);++z.compositions;u=g.edges[e].v;need(++length<=g.ids.size(),"seed cycle");}
 auto [upper,earliest]=minimum(seed);++z.minima;
 z.seconds[1]=std::chrono::duration<double>(Clock::now()-begin).count();
 begin=Clock::now();std::vector<P>labels(g.ids.size());std::vector<unsigned long long>versions(g.ids.size());
 std::vector<unsigned char>expanded(g.ids.size());
 using Entry=std::tuple<Q,int,unsigned long long>;std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>>open;
 labels[source]={{lo,hi,1,0}};open.emplace(Q(h.d[source]),source,++versions[source]);
 bool bound_stop=false;Q remaining;
 while(!open.empty()){
  auto [key,u,version]=open.top();open.pop();
  if(version!=versions[u]){++z.stale;continue;}
  if(key>upper){bound_stop=true;remaining=key;break;}
  ++z.pops;if(expanded[u])++z.reopens;expanded[u]=1;
  if(u==destination)continue;
  Q prefix_min=key-h.d[u];
  // Nonnegative FIFO self-loops cannot improve labels[u] during expansion.
  for(int e:g.out[u]){
   z.scan(e,2);int v=g.edges[e].v;
   if(h.d[v]==INF||prefix_min+g.edges[e].low+h.d[v]>upper){++z.pruned;continue;}
   P candidate=g.compose(labels[u],e);++z.compositions;
   auto [merged,changed]=lower(labels[v],candidate);++z.envelopes;
   if(!changed)continue;
   labels[v]=std::move(merged);++z.updates;z.max_pieces=std::max(z.max_pieces,(long long)labels[v].size());
   auto [cost,departure]=minimum(labels[v]);++z.minima;
   if(v==destination){if(cost<upper||(cost==upper&&departure<earliest)){upper=cost;earliest=departure;}}
   else open.emplace(Q(cost+h.d[v]),v,++versions[v]);
  }
 }
 z.seconds[2]=std::chrono::duration<double>(Clock::now()-begin).count();
 begin=Clock::now();auto result=recover(g,source,destination,earliest,z);
 z.seconds[3]=std::chrono::duration<double>(Clock::now()-begin).count();
 need(result.found&&result.cost==upper,"continuous certificate differs from recovered route");
 result.certificate=bound_stop?"exact_full_interval_open_lower_bound_exceeds_incumbent":"exact_full_interval_open_exhausted";
 result.has_remaining_key=bound_stop;result.remaining_key=remaining;return result;
}
static thread_local std::string error;
}
extern "C" {
void* ca_create(const char*p){try{return new continuous_interval_astar_v1::Handle(p);}catch(const std::exception&e){continuous_interval_astar_v1::error=e.what();return nullptr;}}
void ca_destroy(void*p){delete static_cast<continuous_interval_astar_v1::Handle*>(p);}
const char* ca_error(){return continuous_interval_astar_v1::error.c_str();}
const char* ca_query(void*p,long long source,long long destination,long long lo,long long hi){
 using namespace continuous_interval_astar_v1;
 try{
  auto&h=*static_cast<Handle*>(p);const auto&g=h.g;need(lo>=0&&lo<hi,"invalid closed departure interval");
  auto begin=Clock::now();Stats z(g.edges.size());auto r=solve(g,g.index.at(source),g.index.at(destination),Q(lo),Q(hi),z);
  double seconds=std::chrono::duration<double>(Clock::now()-begin).count();
  std::ostringstream o;o.precision(17);o<<"{\"solver\":\"Continuous-Interval-AStar-v1\",\"version\":\"1.0.0\",\"status\":\""<<(r.found?"OK":"NO_PATH")<<"\",\"certificate\":\""<<r.certificate<<"\",\"scope\":\"full_graph_closed_continuous_interval\"";
  if(r.found){o<<",\"cost\":\""<<r.cost<<"\",\"departure\":\""<<r.departure<<"\",\"route\":["<<source;
   for(int e:r.edges)o<<","<<g.ids[g.edges[e].v];o<<"],\"edge_ids\":[";
   for(size_t i=0;i<r.edges.size();++i){if(i)o<<",";o<<r.edges[i];}o<<"],\"arrivals\":[";
   for(size_t i=0;i<r.arrivals.size();++i){if(i)o<<",";o<<"\""<<r.arrivals[i]<<"\"";}o<<"]";
   if(r.has_remaining_key)o<<",\"remaining_open_lower_bound\":\""<<r.remaining_key<<"\"";
  }
  long long unique=0;for(auto b:z.seen)if(b)++unique;
  o<<",\"edge_unique_all\":"<<unique<<",\"stage_names\":[\"bounds\",\"seed\",\"propagation\",\"recovery\"],\"stage_seconds\":[";
  for(int i=0;i<4;++i){if(i)o<<",";o<<z.seconds[i];}o<<"],\"stage_scans\":[";
  for(int i=0;i<4;++i){if(i)o<<",";o<<z.scans[i];}o<<"],\"stage_unique\":[";
  for(int i=0;i<4;++i){if(i)o<<",";o<<z.unique[i];}o<<"]";
  o<<",\"compositions\":"<<z.compositions<<",\"envelopes\":"<<z.envelopes<<",\"minima\":"<<z.minima<<",\"pops\":"<<z.pops<<",\"reopens\":"<<z.reopens<<",\"stale_entries\":"<<z.stale<<",\"updates\":"<<z.updates<<",\"pruned\":"<<z.pruned<<",\"max_pieces\":"<<z.max_pieces<<",\"native_seconds\":"<<seconds<<"}";
  h.output=o.str();return h.output.c_str();
 }catch(const std::exception&e){error=e.what();return nullptr;}
}
}
