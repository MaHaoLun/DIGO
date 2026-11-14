// PWL algebra extracted from continuous_exact_20261009/src/exact.cpp.
// No candidate search, pruning, interval partitioning or full-graph continuation.
#pragma once
#include <boost/multiprecision/cpp_int.hpp>
#ifdef EXACT_GMP
#include <boost/multiprecision/gmp.hpp>
#endif
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <thread>
#include <vector>
#ifdef EXACT_GMP
using Q=boost::multiprecision::mpq_rational;
#elif defined(EXACT_FIXED_BITS)
using FixedInt=boost::multiprecision::cpp_int_backend<EXACT_FIXED_BITS,EXACT_FIXED_BITS,boost::multiprecision::signed_magnitude,boost::multiprecision::checked,void>;
using Q=boost::multiprecision::number<boost::multiprecision::rational_adaptor<FixedInt>>;
#else
using Q=boost::multiprecision::cpp_rational;
#endif
using Clock=std::chrono::steady_clock;
#ifdef E4_PROFILE_ALGEBRA
// Diagnostic only: these non-nested operations partition algebra wall time.
static thread_local double algebra_seconds[3]{};
static thread_local long long algebra_calls[3]{};
struct AlgebraTimer {
 int kind;Clock::time_point begin;
 explicit AlgebraTimer(int k):kind(k),begin(Clock::now()){++algebra_calls[kind];}
 ~AlgebraTimer(){algebra_seconds[kind]+=std::chrono::duration<double>(Clock::now()-begin).count();}
};
#endif
struct Seg {Q l,r,m,b; Q at(const Q&t)const{return m*t+b;}};
using P=std::vector<Seg>;
static void need(bool v,const char*s){if(!v)throw std::runtime_error(s);}
static void add(P&p,const Q&l,const Q&r,const Q&m,const Q&b){
    if(l>=r)return;
    if(!p.empty() && p.back().r==l && p.back().m==m && p.back().b==b)p.back().r=r;
    else p.push_back({l,r,m,b});
}
struct Edge {int u,v;std::vector<long long>w;long long low;bool constant;};
struct Graph {
 int k=0;long long step=0,period=0;std::vector<long long>ids;std::vector<int>cells;
 std::map<long long,int>index;std::vector<Edge>edges;std::vector<std::vector<int>>out,in;
 explicit Graph(const char*file){
  std::ifstream f(file);int n,m;f>>n>>m>>k>>step;need(f.good()&&n>0&&m>=0&&k>0&&step>0,"bad graph header");period=k*step;
  out.resize(n);in.resize(n);
  for(int i=0;i<n;i++){long long id;int cell;f>>id>>cell;need(!index.count(id),"duplicate vertex");index[id]=i;ids.push_back(id);cells.push_back(cell);}
  for(int i=0;i<m;i++){long long a,b;f>>a>>b;need(index.count(a)&&index.count(b),"missing endpoint");Edge e{index[a],index[b],{},0,true};
   for(int j=0;j<k;j++){long long w;f>>w;need(w>=0,"negative edge");e.w.push_back(w);}
   e.low=*std::min_element(e.w.begin(),e.w.end());
   for(int j=0;j<k;j++){need(e.w[j]-e.w[(j+1)%k]<=step,"non FIFO");if(e.w[j]!=e.w[0])e.constant=false;}
   out[e.u].push_back(i);in[e.v].push_back(i);edges.push_back(e);
  }need(bool(f),"truncated graph");
 }
 long long slot(const Q&t)const{need(t>=0,"negative entry");Q z=t/step;return (numerator(z)/denominator(z)).convert_to<long long>();}
 std::pair<Q,Q> line(const Edge&e,long long j)const{
  int i=j%k;Q slope=Q(e.w[(i+1)%k]-e.w[i])/step;
  return {Q(1)+slope,Q(e.w[i])-slope*(j*step)};
 }
 Q arrival(int eid,const Q&t)const{const auto&e=edges[eid];if(e.constant)return t+e.w[0];auto [m,b]=line(e,slot(t));return m*t+b;}
 long long window_low(int eid,long long lo,long long hi)const{
  // All knot times share step, so integer entry bounds have edge minima with
  // denominator step. Keep that exact scale instead of flooring every edge.
  const auto&e=edges[eid];if(e.constant||hi-lo>=period)return e.low*step;
  auto val=[&](long long t){long long j=t/step,r=t%step;int i=j%k;__int128 n=(__int128)e.w[i]*(step-r)+(__int128)e.w[(i+1)%k]*r;need(n<=std::numeric_limits<long long>::max()/4,"scaled weight overflow");return (long long)n;};
  long long low=std::min(val(lo),val(hi));for(long long j=lo/step+1;j<=hi/step;j++)low=std::min(low,e.w[j%k]*step);return low;
 }
 P compose(const P&p,int eid)const{
#ifdef E4_PROFILE_ALGEBRA
  AlgebraTimer timer(0);
#endif
  P result;const auto&e=edges[eid];
  if(e.constant){for(const auto&s:p)add(result,s.l,s.r,s.m,s.b+e.w[0]);return result;}
  for(const auto&s:p){Q t=s.l;
   if(s.m==0){add(result,s.l,s.r,0,arrival(eid,s.b));continue;}
   while(t<s.r){long long j=slot(s.at(t));Q end=std::min(s.r,Q((Q((j+1)*step)-s.b)/s.m));need(end>t,"nonadvancing compose");auto [m,b]=line(e,j);add(result,t,end,Q(m*s.m),Q(m*s.b+b));t=end;}
  }return result;
 }
};
static std::pair<P,bool> lower(const P&a,const P&b){
#ifdef E4_PROFILE_ALGEBRA
 AlgebraTimer timer(1);
#endif
 if(a.empty())return {b,!b.empty()};if(b.empty())return {a,false};
 P out;bool changed=false;size_t i=0,j=0;
 while(i<a.size()&&j<b.size()){
  const auto&x=a[i];const auto&y=b[j];Q l=std::max(x.l,y.l),r=std::min(x.r,y.r);
#ifdef E4_FAST_ENVELOPE
  // Exact affine difference: avoid divisions and midpoint evaluations when
  // one segment dominates throughout this overlap. No approximation/pruning.
  if(l<r){
   if(x.m==y.m){
    if(y.b<x.b){add(out,l,r,y.m,y.b);changed=true;}
    else add(out,l,r,x.m,x.b);
   }else{
    Q dm=y.m-x.m,db=y.b-x.b,dl=dm*l+db,dr=dm*r+db;
    if(dl>=0&&dr>=0)add(out,l,r,x.m,x.b);
    else if(dl<=0&&dr<=0){add(out,l,r,y.m,y.b);changed=true;}
    else{
     Q cross=-db/dm;need(l<cross&&cross<r,"bad envelope crossing");
     if(dl<0){add(out,l,cross,y.m,y.b);add(out,cross,r,x.m,x.b);}
     else{add(out,l,cross,x.m,x.b);add(out,cross,r,y.m,y.b);}
     changed=true;
    }
   }
  }
#else
  auto part=[&](const Q&v,const Q&w){if(v>=w)return;Q t=(v+w)/2;if(y.at(t)<x.at(t)){add(out,v,w,y.m,y.b);changed=true;}else add(out,v,w,x.m,x.b);};
  if(l<r){if(x.m!=y.m){Q cross=(y.b-x.b)/(x.m-y.m);if(l<cross&&cross<r){part(l,cross);part(cross,r);}else part(l,r);}else part(l,r);}
#endif
  if(x.r==r)++i;if(y.r==r)++j;
 }need(i==a.size()&&j==b.size(),"profile domain mismatch");return {out,changed};
}
static std::pair<Q,Q> minimum(const P&p){
#ifdef E4_PROFILE_ALGEBRA
 AlgebraTimer timer(2);
#endif
 need(!p.empty(),"empty profile minimum");Q cost=p[0].at(p[0].l)-p[0].l,time=p[0].l;
 for(const auto&s:p)for(const Q&t:{s.l,s.r}){Q c=s.at(t)-t;if(c<cost||(c==cost&&t<time)){cost=c;time=t;}}
 return {cost,time};
}
