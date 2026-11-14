// Compact formal region adapter; same exact core as the full-graph baseline.
#include "interval_astar.cpp"
namespace exact=continuous_interval_astar_v1;
using Mask=std::array<unsigned char,4096>;
struct Regional:exact::Handle {
 std::array<std::vector<int>,4096>nodes;
 explicit Regional(const char*p):exact::Handle(p){for(int u=0;u<(int)g.ids.size();++u){need(g.cells[u]>=0&&g.cells[u]<4096,"bad cell");nodes[g.cells[u]].push_back(u);}}
};
struct Meter {
 std::vector<unsigned char>seen;std::array<long long,5>scans{},unique{};std::array<double,5>seconds{};
 explicit Meter(size_t m):seen(m){}
 void scan(int e,int stage){++scans[stage];unsigned char bit=1<<stage;if(!(seen[e]&bit)){seen[e]|=bit;++unique[stage];}}
 void fold(const exact::Stats&other,int source_bits,int target_stage){
  for(int i=0;i<4;++i)if(source_bits&(1<<i))scans[target_stage]+=other.scans[i];
  for(size_t e=0;e<seen.size();++e)if(other.seen[e]&source_bits){unsigned char bit=1<<target_stage;if(!(seen[e]&bit)){seen[e]|=bit;++unique[target_stage];}}
 }
};
static bool connected(const Graph&g,int s,int d,const Mask&mask,Meter&z){
 std::vector<unsigned char>seen(g.ids.size());std::vector<int>queue{s};seen[s]=1;
 for(size_t i=0;i<queue.size();++i){int u=queue[i];if(u==d)return true;
  for(int e:g.out[u]){z.scan(e,0);int v=g.edges[e].v;if(mask[g.cells[v]]&&!seen[v]){seen[v]=1;queue.push_back(v);}}}
 return false;
}
static bool endpoint_bridge(const Graph&g,int s,int d,long long lo,long long hi,Mask&mask,Meter&z){
 exact::Stats repair_stats(g.edges.size());bool reachable=true;
 for(long long t:{lo,hi}){
  auto path=exact::recover(g,s,d,Q(t),repair_stats);if(!path.found){reachable=false;break;}
  for(int e:path.edges){mask[g.cells[g.edges[e].u]]=1;mask[g.cells[g.edges[e].v]]=1;}
 }
 z.fold(repair_stats,8,1);return reachable;
}
struct Domain {
 Graph&g;std::vector<std::vector<int>>old_out,old_in;
 Domain(Graph&graph,std::vector<std::vector<int>>&&view):g(graph){
  old_out.swap(g.out);old_in.swap(g.in);g.out=std::move(view);g.in.resize(g.ids.size());
  for(auto&edges:g.out)for(int e:edges)g.in[g.edges[e].v].push_back(e);
 }
 ~Domain(){g.out.swap(old_out);g.in.swap(old_in);}
};
static thread_local std::string region_error;
extern "C" {
void* e4_create(const char*p){try{return new Regional(p);}catch(const std::exception&e){region_error=e.what();return nullptr;}}
void e4_destroy(void*p){delete static_cast<Regional*>(p);}
const char* e4_error(){return region_error.c_str();}
const char* e4_query(void*p,long long source,long long destination,long long lo,long long hi,int method,const unsigned char*input){
 try{
  need(lo>=0&&lo<hi,"bad interval");need(method==0||method==1||method==4||method==5||method==6||method==8,"bad method");
  auto&h=*static_cast<Regional*>(p);auto&g=h.g;int s=g.index.at(source),d=g.index.at(destination);
  auto begin=Clock::now();Mask active{};
  if(method==0)active.fill(1);
  else if(method==6){active[g.cells[s]]=1;active[g.cells[d]]=1;}
  else{need(input,"missing mask");for(int i=0;i<4096;++i)active[i]=input[i]!=0;}
  need(active[g.cells[s]]&&active[g.cells[d]],"inactive endpoint");
  Meter z(g.edges.size());exact::Stats core(g.edges.size());exact::Result result;
  auto mark=Clock::now();bool reachable=method==0||connected(g,s,d,active,z);
  z.seconds[0]=std::chrono::duration<double>(Clock::now()-mark).count();
  bool initial_connected=reachable;bool repaired=false;
  mark=Clock::now();
  if(method==5||(!reachable&&(method==4||method==6||method==8))){repaired=true;reachable=endpoint_bridge(g,s,d,lo,hi,active,z);}
  z.seconds[1]=std::chrono::duration<double>(Clock::now()-mark).count();
  if(reachable){
   mark=Clock::now();std::vector<std::vector<int>>view(g.ids.size());
   for(int cell=0;cell<4096;++cell)if(active[cell])for(int u:h.nodes[cell])for(int e:g.out[u]){z.scan(e,2);if(active[g.cells[g.edges[e].v]])view[u].push_back(e);}
   Domain scope(g,std::move(view));z.seconds[2]=std::chrono::duration<double>(Clock::now()-mark).count();
   mark=Clock::now();result=exact::solve(g,s,d,Q(lo),Q(hi),core);
   z.fold(core,7,3);z.fold(core,8,4);z.seconds[4]=core.seconds[3];
   z.seconds[3]=std::chrono::duration<double>(Clock::now()-mark).count()-z.seconds[4];
  }
  double elapsed=std::chrono::duration<double>(Clock::now()-begin).count();
  std::ostringstream o;o.precision(17);o<<"{\"solver\":\"GES-AStar-full-v2\",\"status\":\""<<(result.found?"OK":"NO_PATH")<<"\",\"method\":"<<method<<",\"initial_connected\":"<<(initial_connected?"true":"false")<<",\"repaired\":"<<(repaired?"true":"false");
  if(result.found){o<<",\"cost\":\""<<result.cost<<"\",\"departure\":\""<<result.departure<<"\",\"route\":["<<source;
   for(int e:result.edges)o<<","<<g.ids[g.edges[e].v];o<<"],\"edge_ids\":[";
   for(size_t i=0;i<result.edges.size();++i){if(i)o<<",";o<<result.edges[i];}o<<"],\"arrivals\":[";
   for(size_t i=0;i<result.arrivals.size();++i){if(i)o<<",";o<<"\""<<result.arrivals[i]<<"\"";}o<<"]";
  }
  o<<",\"active_cells\":[";bool first=true;for(int i=0;i<4096;++i)if(active[i]){if(!first)o<<",";first=false;o<<i;}o<<"]";
  long long unique=0;for(auto b:z.seen)if(b)++unique;
  o<<",\"edge_unique_all\":"<<unique<<",\"stage_scans\":[";
  for(int i=0;i<5;++i){if(i)o<<",";o<<z.scans[i];}o<<"],\"stage_unique\":[";
  for(int i=0;i<5;++i){if(i)o<<",";o<<z.unique[i];}o<<"],\"stage_seconds\":[";
  for(int i=0;i<5;++i){if(i)o<<",";o<<z.seconds[i];}o<<"]";
  o<<",\"compositions\":"<<core.compositions<<",\"updates\":"<<core.updates<<",\"pops\":"<<core.pops<<",\"max_pieces\":"<<core.max_pieces<<",\"reopens\":"<<core.reopens<<",\"native_seconds\":"<<elapsed<<"}";
  h.output=o.str();return h.output.c_str();
 }catch(const std::exception&e){region_error=e.what();return nullptr;}
}
}
