#include "corpus/online.hpp"
#include "corpus/routing.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace corpus {
namespace {
using Json = sdc::io::Json;
constexpr std::array<std::int64_t,3> monitor{43200,44100,45000};
Json decimal(double value) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite online diagnostic");
    std::ostringstream out; out << std::setprecision(17) << value;
    return Json::number(out.str());
}
OdPair canonical(std::int64_t a, std::int64_t b) { return {std::min(a,b),std::max(a,b)}; }
const Edge& edge_to(const Graph& graph, std::size_t from, std::size_t to) {
    const auto& edges=graph.adjacency.at(from);
    auto found=std::lower_bound(edges.begin(),edges.end(),to,
        [](const Edge& edge,std::size_t value){return edge.to<value;});
    if(found==edges.end() || found->to!=to) throw std::runtime_error("online graph topology or route edge mismatch");
    return *found;
}
void compatible(const Graph& old_graph,const Graph& new_graph) {
    if(old_graph.policy!=RoutingPolicy::CanonicalFifoLpf || new_graph.policy!=RoutingPolicy::CanonicalFifoLpf)
        throw std::invalid_argument("online detection requires canonical FIFO models");
    if(old_graph.fifo_integer_violations || new_graph.fifo_integer_violations)
        throw std::invalid_argument("online detection refuses FIFO violations");
    if(old_graph.nodes.size()!=new_graph.nodes.size() || old_graph.adjacency.size()!=old_graph.nodes.size() ||
       new_graph.adjacency.size()!=new_graph.nodes.size()) throw std::invalid_argument("online graph registry size mismatch");
    const auto& a=old_graph.bbox; const auto& b=new_graph.bbox;
    if(a.longitude_min!=b.longitude_min || a.longitude_max!=b.longitude_max ||
       a.latitude_min!=b.latitude_min || a.latitude_max!=b.latitude_max)
        throw std::invalid_argument("online graph bbox mismatch");
    for(std::size_t i=0;i<old_graph.nodes.size();++i) {
        const auto& a_node=old_graph.nodes[i]; const auto& b_node=new_graph.nodes[i];
        if(a_node.id!=b_node.id || a_node.x!=b_node.x || a_node.y!=b_node.y ||
           a_node.coarse_cell!=b_node.coarse_cell || a_node.fine_cell!=b_node.fine_cell ||
           a_node.longitude_condition!=b_node.longitude_condition || a_node.latitude_condition!=b_node.latitude_condition)
            throw std::invalid_argument("online graph node registry mismatch");
        if(old_graph.adjacency[i].size()!=new_graph.adjacency[i].size())
            throw std::invalid_argument("online graph topology size mismatch");
        for(std::size_t j=0;j<old_graph.adjacency[i].size();++j)
            if(old_graph.adjacency[i][j].to!=new_graph.adjacency[i][j].to)
                throw std::invalid_argument("online graph topology mismatch");
    }
}
}

G0Membership read_g0_membership(const std::filesystem::path& path) {
    std::ifstream input(path); if(!input) throw std::runtime_error("cannot read G0 OD membership");
    std::string line; if(!std::getline(input,line)) throw std::runtime_error("empty G0 OD membership");
    if(!line.empty() && line.back()=='\r') line.pop_back();
    if(line!="source,destination,split") throw std::runtime_error("G0 OD membership requires source,destination,split header");
    G0Membership result;
    while(std::getline(input,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        std::istringstream row(line); std::string source,destination,role,extra;
        if(!std::getline(row,source,',') || !std::getline(row,destination,',') || !std::getline(row,role) || role.find(',')!=std::string::npos)
            throw std::runtime_error("invalid G0 OD membership row");
        const auto integer=[](const std::string& value){std::size_t used=0; auto n=std::stoll(value,&used);
            if(used!=value.size() || value.empty()) throw std::runtime_error("invalid G0 OD identifier");
            return n;};
        Split split;
        if(role=="train") split=Split::train;
        else if(role=="validation") split=Split::validation;
        else if(role=="base_test") split=Split::base_test;
        else throw std::runtime_error("unsupported G0 OD split");
        const OdPair pair{integer(source),integer(destination)};
        if(pair.first==pair.second || !result.emplace(pair,split).second)
            throw std::runtime_error("duplicate or trivial G0 OD membership");
    }
    if(!input.eof()) throw std::runtime_error("failed reading G0 OD membership");
    return result;
}

ShiftDetection detect_shifts(const Graph& old_graph,const Graph& new_graph) {
    compatible(old_graph,new_graph); ShiftDetection result;
    for(std::size_t u=0;u<new_graph.nodes.size();++u) {
        for(std::size_t j=0;j<new_graph.adjacency[u].size();++j) {
            const auto& edge=new_graph.adjacency[u][j];
            if(new_graph.nodes[u].id>new_graph.nodes[edge.to].id) continue;
            const auto& old_edge=old_graph.adjacency[u][j]; ++result.canonical_pairs;
            std::uint64_t changed=0;
            for(std::size_t k=0;k<96;++k) changed+=edge.weights[k]!=old_edge.weights[k];
            result.effective_changed_anchors+=changed; result.effective_changed_anchor_pairs+=changed!=0;
            __int128_t absolute_sum=0,old_sum=0;
            for(auto time:monitor) {
                const auto before=old_graph.travel_time(old_edge,time),after=new_graph.travel_time(edge,time);
                __int128_t difference=static_cast<__int128_t>(after)-before;
                absolute_sum+=difference<0?-difference:difference; old_sum+=before;
            }
            result.monitoring_changed_pairs+=absolute_sum!=0;
            const auto denominator=std::max<__int128_t>(old_sum,3);
            if(absolute_sum<15 || absolute_sum*10<denominator) continue;
            ShiftPair shifted{static_cast<double>(absolute_sum)/3.0,static_cast<double>(absolute_sum)/static_cast<double>(denominator)};
            result.shifted_pairs.emplace(canonical(new_graph.nodes[u].id,new_graph.nodes[edge.to].id),shifted);
            const auto first=new_graph.nodes[u].coarse_cell,second=new_graph.nodes[edge.to].coarse_cell;
            result.cell_scores.at(static_cast<std::size_t>(first))+=shifted.relative_change;
            if(second!=first) result.cell_scores.at(static_cast<std::size_t>(second))+=shifted.relative_change;
        }
    }
    std::vector<int> order;
    for(int cell=0;cell<1024;++cell) if(result.cell_scores[static_cast<std::size_t>(cell)]>0) order.push_back(cell);
    std::sort(order.begin(),order.end(),[&](int a,int b){
        const auto x=result.cell_scores[static_cast<std::size_t>(a)],y=result.cell_scores[static_cast<std::size_t>(b)];
        return x!=y?x>y:a<b;
    });
    const auto count=(order.size()+4)/5;
    result.hotspots.insert(order.begin(),order.begin()+static_cast<std::ptrdiff_t>(count));
    return result;
}
Json ShiftDetection::report() const {
    Json::Array pairs,cells;
    for(const auto& item:shifted_pairs) pairs.emplace_back(Json::Object{
        {"source",item.first.first},{"destination",item.first.second},
        {"absolute_change",decimal(item.second.absolute_change)},{"relative_change",decimal(item.second.relative_change)}});
    for(int cell=0;cell<1024;++cell) cells.emplace_back(Json::Object{{"grid_id",cell},
        {"score",decimal(cell_scores[static_cast<std::size_t>(cell)])},{"selected",hotspots.count(cell)!=0}});
    return Json::Object{{"canonical_pairs",canonical_pairs},{"effective_changed_anchor_pairs",effective_changed_anchor_pairs},
        {"effective_changed_anchors",effective_changed_anchors},{"monitoring_changed_pairs",monitoring_changed_pairs},
        {"monitoring_times",Json::Array{43200,44100,45000}},{"absolute_threshold_seconds",5},
        {"relative_threshold",decimal(0.1)},{"epsilon_time_seconds",1},{"comparison","greater_than_or_equal_AND"},
        {"hotspot_fraction",decimal(0.2)},{"hotspot_rule","ceil(positive_cell_count/5); score descending then grid ID ascending"},
        {"shifted_pair_count",static_cast<std::uint64_t>(shifted_pairs.size())},
        {"hotspot_count",static_cast<std::uint64_t>(hotspots.size())},{"shifted_pairs",pairs},{"cells",cells},
        {"anchor_counts_semantics","effective repaired FIFO anchors; raw/fallback attribution requires source audit"}};
}
PointEvidence analyze_point(const Graph& old_graph,const Graph& new_graph,const ShiftDetection& detection,
                            const RouteRecord& route,const RouteRecord* old_route) {
    if(route.status!="ok" || route.node_ids.empty() || route.node_ids.size()!=route.arrival_times.size())
        throw std::invalid_argument("online evidence requires successful complete G1 route");
    PointEvidence result;
    bool endpoint=false;
    for(std::size_t i=0;i<route.node_ids.size();++i) {
        const auto u=new_graph.index(route.node_ids[i]);
        if(detection.hotspots.count(new_graph.nodes[u].coarse_cell)) {
            if(i==0 || i+1==route.node_ids.size()) endpoint=true;
            else result.internal_hotspot_hit=true;
        }
        if(i+1==route.node_ids.size()) continue;
        const auto pair=canonical(route.node_ids[i],route.node_ids[i+1]);
        if(!detection.shifted_pairs.count(pair)) continue;
        result.shifted_pair_hit=true;
        const auto v=new_graph.index(route.node_ids[i+1]);
        const auto& before=edge_to(old_graph,u,v); const auto& after=edge_to(new_graph,u,v);
        result.changed_edge_at_entry |= old_graph.travel_time(before,route.arrival_times[i])!=new_graph.travel_time(after,route.arrival_times[i]);
    }
    result.endpoint_only_hit=endpoint && !result.internal_hotspot_hit;
    if(old_route) {
        if(old_route->status!="ok" || old_route->query.source!=route.query.source || old_route->query.destination!=route.query.destination ||
           old_route->query.departure!=route.query.departure || old_route->node_ids.empty() ||
           old_route->node_ids.size()!=old_route->arrival_times.size())
            throw std::invalid_argument("online counterfactual must be successful same-clock same-OD G0 route");
        result.has_old_route=true;
        result.old_new_path_changed=old_route->node_ids!=route.node_ids;
        result.old_new_duration_changed=old_route->travel_time!=route.travel_time;
    }
    return result;
}
Json PointEvidence::json() const {
    return Json::Object{{"shifted_pair_hit",shifted_pair_hit},{"changed_edge_at_entry",changed_edge_at_entry},
        {"internal_hotspot_hit",internal_hotspot_hit},{"endpoint_only_hit",endpoint_only_hit},
        {"has_old_route",has_old_route},{"old_new_path_changed",has_old_route?Json(old_new_path_changed):Json()},
        {"old_new_duration_changed",has_old_route?Json(old_new_duration_changed):Json()},{"eligible",eligible()}};
}
OnlineConfig::OnlineConfig() {
    sampling.start_second=43200; sampling.interval_seconds=1800; sampling.point_step_seconds=900;
    sampling.split_rows={0,0,0};
}
Json OnlinePlan::capacity_report() const {
    Json::Array splits;
    for(std::size_t i=0;i<2;++i) splits.emplace_back(Json::Object{
        {"split",i==0?"adaptation":"post_update_test"},{"legal_groups",legal_groups[i]},
        {"ordinary_groups",ordinary_groups[i]},{"targeted_groups",targeted_groups[i]},
        {"screened_groups",screened_groups[i]},{"eligible_screened_groups",eligible_screened_groups[i]}});
    return Json::Object{{"representative_count",representative_count},{"candidate_groups",candidate_groups},
        {"reachable_groups",reachable_groups},{"new_od_groups",new_od_groups},{"new_od_adaptation_groups",new_od_adaptation_groups},
        {"new_od_assignment","canonical-rank-ordered new ODs; Random(seed) mt19937_64 rejection/Fisher-Yates shuffle; first floor(4*N/5) adaptation"},
        {"qualification_old_sssp",qualification_old_sssp},{"qualification_new_sssp",qualification_new_sssp},
        {"qualification_sssp",qualification_old_sssp+qualification_new_sssp},{"splits",splits}};
}

OnlinePlan make_online_plan(const Graph& old_graph,const Graph& new_graph,const OnlineConfig& config,
                           const G0Membership& membership,const std::string& city,
                           const std::string& new_model_id,const std::string& old_model_id,const ShiftDetection& detection) {
    compatible(old_graph,new_graph);
    if(city.empty() || new_model_id.empty() || old_model_id.empty() || new_model_id==old_model_id)
        throw std::invalid_argument("online queries require distinct nonempty model identities");
    if(config.sampling.method!="hierarchical_grid" || config.sampling.start_second!=43200 ||
       config.sampling.interval_seconds!=1800 || config.sampling.point_step_seconds!=900 || config.sampling.points_per_group()!=2)
        throw std::invalid_argument("online sampling requires hierarchical noon two-point window");
    if(!config.qualification_chunk_groups) throw std::invalid_argument("qualification chunk must be positive");
    auto sampling=config.sampling; sampling.split_rows={0,0,0};
    auto reps=hierarchical_representatives(new_graph,sampling);
    std::sort(reps.begin(),reps.end(),[&](auto a,auto b){return new_graph.nodes[a].id<new_graph.nodes[b].id;});
    struct Group {std::size_t source,destination;std::uint64_t rank;std::size_t split=0;};
    std::vector<Group> groups; std::vector<std::size_t> new_groups;
    OnlinePlan result; result.representative_count=reps.size();
    std::size_t matched_membership=0;
    for(auto source:reps) for(auto destination:reps) {
        if(source==destination) continue;
        const auto sc=new_graph.nodes[source].coarse_cell,dc=new_graph.nodes[destination].coarse_cell;
        const auto distance=std::abs(sc%32-dc%32)+std::abs(sc/32-dc/32);
        if(sc==dc || distance<sampling.min_distance || distance>sampling.max_distance) continue;
        if(result.candidate_groups>(std::numeric_limits<std::uint64_t>::max()-1)/2)
            throw std::overflow_error("online canonical rank overflows query ID");
        const auto rank=result.candidate_groups++;
        if(new_graph.component[source]!=new_graph.component[destination]) continue;
        Group group{source,destination,rank,0};
        const OdPair pair{new_graph.nodes[source].id,new_graph.nodes[destination].id};
        const auto known=membership.find(pair);
        if(known==membership.end()) new_groups.push_back(groups.size());
        else {
            ++matched_membership;
            if(known->second==Split::train || known->second==Split::validation) group.split=0;
            else if(known->second==Split::base_test) group.split=1;
            else throw std::invalid_argument("G0 membership contains non-G0 role");
        }
        groups.push_back(group);
    }
    if(matched_membership!=membership.size()) throw std::invalid_argument("G0 membership outside unchanged reachable OD domain");
    result.reachable_groups=groups.size(); result.new_od_groups=new_groups.size();
    sampling_detail::Random partition_random(sampling.seed); partition_random.shuffle(new_groups);
    const auto adaptation_count=new_groups.size()/5*4+(new_groups.size()%5)*4/5;
    result.new_od_adaptation_groups=adaptation_count;
    for(std::size_t i=0;i<new_groups.size();++i) groups[new_groups[i]].split=i<adaptation_count?0:1;
    std::array<std::vector<std::size_t>,2> pools;
    for(std::size_t i=0;i<groups.size();++i) pools[groups[i].split].push_back(i);
    sampling_detail::Random selection_random(sampling.seed);
    for(std::size_t split=0;split<2;++split) {
        result.legal_groups[split]=pools[split].size();
        if(config.split_rows[split]%10 || !config.split_rows[split])
            throw std::invalid_argument("each online split must have positive rows divisible by 10 for exact 80/20 two-point groups");
        const auto needed=config.split_rows[split]/2;
        result.targeted_groups[split]=needed/5; result.ordinary_groups[split]=needed-result.targeted_groups[split];
        if(needed>pools[split].size()) throw std::runtime_error("insufficient legal online split capacity; no quota reduction permitted");
        selection_random.shuffle(pools[split]);
    }
    if(config.split_rows[0]>std::numeric_limits<std::uint64_t>::max()-config.split_rows[1] ||
       config.split_rows[0]+config.split_rows[1]>result.queries.max_size())
        throw std::length_error("online query row count exceeds capacity");
    result.queries.reserve(static_cast<std::size_t>(config.split_rows[0]+config.split_rows[1]));
    result.targeted.reserve(result.queries.capacity()); result.qualification_evidence.reserve(result.queries.capacity());
    const auto query_for=[&](const Group& group,std::size_t point) {
        QueryRecord query; query.query_id=group.rank*2+point;query.interval_id=group.rank;
        query.city=city;query.snapshot_id=new_model_id;query.source=new_graph.nodes[group.source].id;
        query.destination=new_graph.nodes[group.destination].id;query.departure=43200+static_cast<std::int64_t>(point)*900;
        query.split=group.split==0?Split::adaptation:Split::post_update_test; return query;
    };
    std::array<std::uint64_t,2> row_indices{};
    const auto append=[&](const Group& group,bool targeted,const std::array<PointEvidence,2>& evidence) {
        for(std::size_t point=0;point<2;++point) {
            auto query=query_for(group,point);query.row_index=row_indices[group.split]++;
            result.queries.push_back(std::move(query));result.targeted.push_back(targeted?1:0);
            result.qualification_evidence.push_back(evidence[point]);
        }
    };
    TdDijkstra old_solver(old_graph),new_solver(new_graph);
    for(std::size_t split=0;split<2;++split) {
        auto& pool=pools[split]; const auto ordinary=static_cast<std::size_t>(result.ordinary_groups[split]);
        for(std::size_t i=0;i<ordinary;++i) append(groups[pool[i]],false,{});
        // Shuffle ties remain random; only endpoint score changes target priority.
        const auto priority=[&](std::size_t index){const auto& g=groups[index];
            return detection.cell_scores[static_cast<std::size_t>(new_graph.nodes[g.source].coarse_cell)]+
                   detection.cell_scores[static_cast<std::size_t>(new_graph.nodes[g.destination].coarse_cell)];};
        std::stable_sort(pool.begin()+static_cast<std::ptrdiff_t>(ordinary),pool.end(),
                         [&](auto a,auto b){return priority(a)>priority(b);});
        std::uint64_t retained=0;
        for(std::size_t first=ordinary;first<pool.size() && retained<result.targeted_groups[split];) {
            const auto count=std::min(config.qualification_chunk_groups,pool.size()-first);
            const auto last=first+count;
            std::vector<std::array<PointEvidence,2>> evidence(count);
            std::map<std::pair<std::int64_t,std::int64_t>,std::vector<std::size_t>> work;
            for(std::size_t i=first;i<last;++i) for(std::size_t point=0;point<2;++point) {
                const auto query=query_for(groups[pool[i]],point);
                work[{query.source,query.departure}].push_back(i);
            }
            for(const auto& entry:work) {
                std::vector<std::int64_t> targets;targets.reserve(entry.second.size());
                for(auto i:entry.second) targets.push_back(new_graph.nodes[groups[pool[i]].destination].id);
                old_solver.run(entry.first.first,entry.first.second,targets);++result.qualification_old_sssp;
                new_solver.run(entry.first.first,entry.first.second,targets);++result.qualification_new_sssp;
                const auto point=static_cast<std::size_t>((entry.first.second-43200)/900);
                for(auto i:entry.second) {
                    auto query=query_for(groups[pool[i]],point);
                    const auto route=new_solver.route(query); query.snapshot_id=old_model_id;
                    const auto old_route=old_solver.route(query);
                    evidence[i-first][point]=analyze_point(old_graph,new_graph,detection,route,&old_route);
                }
            }
            result.screened_groups[split]+=count;
            for(std::size_t i=first;i<last;++i) {
                const auto& flags=evidence[i-first];
                if(!flags[0].eligible() && !flags[1].eligible()) continue;
                ++result.eligible_screened_groups[split];
                if(retained<result.targeted_groups[split]) {append(groups[pool[i]],true,flags);++retained;}
            }
            first=last;
        }
        if(retained!=result.targeted_groups[split]) throw std::runtime_error("insufficient eligible online target capacity for split "+
            std::to_string(split)+": required="+std::to_string(result.targeted_groups[split])+" retained="+std::to_string(retained)+
            " screened="+std::to_string(result.screened_groups[split])+"; no threshold relaxation, duplicate or quota reduction permitted");
    }
    return result;
}
} // namespace corpus
