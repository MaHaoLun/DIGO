#include "corpus/graph.hpp"
#include "corpus/routing.hpp"
#include "corpus/sampling.hpp"
#include "corpus/writer.hpp"
#include "sdc/io.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <sys/resource.h>
#include <tuple>

namespace fs = std::filesystem;
using sdc::io::Json;
using Clock = std::chrono::steady_clock;
namespace {
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
Json real(double x) { return Json::number(std::to_string(x)); }
std::uint64_t peak_rss() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("cannot read peak RSS");
#ifdef __APPLE__
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
}
std::uint64_t uint_arg(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("expected unsigned integer");
    return std::stoull(text);
}
struct Cli {
    sdc::io::Options paths;
    std::string config="configs/offline.json", mode="generate";
    std::uint64_t fixture_groups=10;
    std::int64_t start_override=-1;
    bool fixture_groups_given=false;
};
Cli parse(int argc, char** argv) {
    Cli c;
    for (int i=1;i<argc;++i) {
        std::string key=argv[i];
        if (key=="--help") {
            std::cout << "sdc_offline --workspace-root ROOT --input-dir data/staged/PACKAGE --city BJ|XA --run-id NEW [--config configs/offline.json] [--mode generate|probe|fixture|qualify] [--start-second SECONDS (probe only)] [--fixture-groups COUNT (fixture only)]\n";
            std::exit(0);
        }
        if (i+1==argc) throw std::runtime_error("missing argument value: "+key);
        std::string value=argv[++i];
        if (key=="--workspace-root") c.paths.workspace_root=value;
        else if (key=="--input-dir") c.paths.input_dir=value;
        else if (key=="--city") c.paths.city=value;
        else if (key=="--run-id") c.paths.run_id=value;
        else if (key=="--config") c.config=value;
        else if (key=="--mode") c.mode=value;
        else if (key=="--start-second") { auto n=uint_arg(value); if(n>=86400)throw std::runtime_error("departure outside day"); c.start_override=static_cast<std::int64_t>(n); }
        else if (key=="--fixture-groups") {c.fixture_groups=uint_arg(value);c.fixture_groups_given=true;}
        else throw std::runtime_error("unknown option: "+key);
    }
    if (c.paths.city!="BJ" && c.paths.city!="XA") throw std::runtime_error("city must be BJ or XA");
    if (c.mode!="generate" && c.mode!="probe" && c.mode!="fixture" && c.mode!="qualify") throw std::runtime_error("invalid mode");
    if (c.start_override>=0 && c.mode!="probe") throw std::runtime_error("start override is probe-only; freeze production configuration first");
    if (c.fixture_groups_given && c.mode!="fixture") throw std::runtime_error("fixture budget cannot alter production or probe counts");
    if (c.fixture_groups<5 || c.fixture_groups>10000) throw std::runtime_error("fixture groups must be 5..10000");
    return c;
}
int integer(const Json& c,const std::string& key) {
    auto n=c.at(key).unsigned_integer();
    if(n>static_cast<std::uint64_t>(std::numeric_limits<int>::max())) throw std::runtime_error("configuration integer overflow");
    return static_cast<int>(n);
}
Json summary(std::vector<std::int64_t> v) {
    if(v.empty())throw std::runtime_error("empty probe statistics");
    std::sort(v.begin(),v.end());
    return Json::Object{{"min",v.front()},{"median",v[v.size()/2]},{"p95",v[(v.size()-1)*95/100]},{"max",v.back()}};
}
Json lpf_audit(const corpus::GraphAudit& a) {
    return Json::Object{{"source_varying_rows",a.source_varying_rows},{"canonical_pair_count",a.canonical_pair_count},
        {"canonical_varying_pairs_before",a.canonical_varying_pairs_before},{"canonical_varying_pairs_after",a.canonical_varying_pairs_after},
        {"changed_pairs",a.changed_pairs},{"changed_knots",a.changed_knots},{"sum_raise",a.sum_raise},{"max_raise",a.max_raise},
        {"raw_lpf_violations",a.raw_lpf_violations},{"repaired_continuous_violations",a.repaired_continuous_violations},
        {"repaired_integer_violations",a.repaired_integer_violations},{"reverse_rows_ignored",a.reverse_rows_ignored},{"duplicate_rows",a.duplicate_rows}};
}
Json::Object graph_record(const corpus::Graph& graph, const sdc::io::Boundaries& boundary,
                          const Json& provenance, const std::string& city) {
    const bool fifo=graph.policy==corpus::RoutingPolicy::CanonicalFifoLpf;
    const std::string policy=fifo?corpus::fifo_policy_name:corpus::legacy_policy_name;
    const auto source=provenance.at("snapshot_id").string();
    if(source.empty())throw std::runtime_error("source snapshot identity is empty");
    const auto identity=fifo?source+"::"+policy:source;
    std::uint64_t arcs=0,varying=0;
    for(const auto& edges:graph.adjacency)for(const auto& edge:edges){
        ++arcs;
        if(std::any_of(edge.weights.begin()+1,edge.weights.end(),[&](auto w){return w!=edge.weights.front();}))++varying;
    }
    Json::Object r{{"city",city},{"date","20160101"},{"graph_state","G0"},{"snapshot_id",identity},
        {"source_snapshot_id",source},{"routing_model_id",identity},{"routing_graph_policy",policy},
        {"inputs",Json::Object{{"snapshot",Json::Object{{"path",boundary.relative(boundary.input("snapshot.txt"))},{"bytes",static_cast<std::uint64_t>(fs::file_size(boundary.input("snapshot.txt")))}}},
            {"coordinates",Json::Object{{"path",boundary.relative(boundary.input("coordinates.txt"))},{"bytes",static_cast<std::uint64_t>(fs::file_size(boundary.input("coordinates.txt")))}}},
            {"provenance",boundary.relative(boundary.input("provenance.json"))}}},
        {"graph",Json::Object{{"nodes",static_cast<std::uint64_t>(graph.nodes.size())},{"input_arcs",graph.input_arc_count},
            {"effective_arcs",arcs},{"input_time_varying_arcs",graph.input_time_varying_arc_count},{"effective_time_varying_arcs",varying}}},
        {"time",Json::Object{{"departure_domain","[0,86400) UTC+08 observation-day seconds"},{"slot_seconds",900},{"slot_count",96},
            {"out_of_day_lookup",fifo?"periodic same-day LPF, exact-ceil integer evaluation; not G1":"floor(unwrapped_arrival/900)%96; periodic same-day slots, not G1"},
            {"arrival_unit","unwrapped integer seconds"},{"arr_unit","minutes"}}},
        {"fifo",Json::Object{{"deferred",!fifo},{"optimality_certified",fifo},{"source_snapshot",provenance.contains("source_fifo")?provenance.at("source_fifo"):Json()},
            {"effective_routing_graph_integer_violations",graph.fifo_integer_violations},
            {"scope",fifo?"all LPF segments including periodic midnight; fixed-departure optimum in the constructed integer-time model, not raw traffic truth or interval-optimal departure":"source: 95 within-day boundaries; effective slots: 96 boundaries including midnight; optimality uncertified"},
            {"source_snapshot_modified",false},{"weights_modified",fifo&&graph.audit.changed_knots>0},{"waiting_allowed",false}}},
        {"dynamics",Json::Object{{"all_effective_weights_constant",varying==0},{"temporal_variation_preserved",varying!=0},
            {"interpretation",fifo?"canonical endpoint-pair dynamic observations shared in both directions and conservatively FIFO-corrected; all source files preserved":"historical last-row slot interpretation; any erased dynamics are not repaired"}}}};
    if(fifo) {
        if(graph.fifo_integer_violations || graph.audit.repaired_continuous_violations || graph.audit.repaired_integer_violations)
            throw std::runtime_error("constructed LPF failed FIFO qualification");
        std::ostringstream out;out<<graph.audit.canonical_pair_count<<" 96 900 86400\n";
        std::uint64_t pairs=0;
        for(std::size_t u=0;u<graph.nodes.size();++u)for(const auto& edge:graph.adjacency[u])if(graph.nodes[u].id<=graph.nodes[edge.to].id){
            out<<graph.nodes[u].id<<' '<<graph.nodes[edge.to].id;
            for(auto w:edge.weights)out<<' '<<w;
            out<<'\n';++pairs;
        }
        if(pairs!=graph.audit.canonical_pair_count)throw std::runtime_error("LPF artifact pair count mismatch");
        sdc::io::write_text_exclusive(boundary.output()/"lpf_knots.txt",out.str());
        r["lpf"]=Json::Object{{"knots_path","lpf_knots.txt"},{"bytes",static_cast<std::uint64_t>(fs::file_size(boundary.output()/"lpf_knots.txt"))},
            {"correction_method","minimal_upward_periodic_majorant_v1"},{"knot_count",96},{"knot_step_seconds",900},{"period_seconds",86400},
            {"rounding","ceil_exact_integer"},{"audit",lpf_audit(graph.audit)}};
    }
    r["mapping"]=Json::Object{{"registry","grid_registry.tsv"},{"bbox_microdegrees",Json::Object{{"lon_min",graph.bbox.longitude_min},{"lon_max",graph.bbox.longitude_max},{"lat_min",graph.bbox.latitude_min},{"lat_max",graph.bbox.latitude_max}}},{"coarse_dim",32},{"fine_dim",64},{"cell_order","latitude row-major y*dim+x; floor; inclusive max clipped"},{"odt","normalized exact o_lon,o_lat,d_lon,d_lat,departure; [-1,1]"}};
    std::ostringstream registry;registry<<"node_id\tlongitude_microdegrees\tlatitude_microdegrees\tcoarse_cell\tfine_cell\n";
    for(const auto& node:graph.nodes)registry<<node.id<<'\t'<<node.x<<'\t'<<node.y<<'\t'<<node.coarse_cell<<'\t'<<node.fine_cell<<'\n';
    sdc::io::write_text_exclusive(boundary.output()/"grid_registry.tsv",registry.str());
    return r;
}
int execute(const Cli& cli) {
    auto total_start=Clock::now();
    sdc::io::Boundaries boundary(cli.paths);
    fs::path config_raw(cli.config);
    if(config_raw.is_absolute())throw std::runtime_error("config must be workspace-relative");
    for(const auto& part:config_raw)if(part=="..")throw std::runtime_error("config parent traversal forbidden");
    auto config_path=boundary.root()/config_raw;
    boundary.verify_regular(config_path);
    auto config=sdc::io::read_json(config_path);
    auto provenance=sdc::io::read_json(boundary.input("provenance.json"));
    if(provenance.at("city").string()!=cli.paths.city || provenance.at("date").string()!="20160101")throw std::runtime_error("city/date provenance mismatch");
    bool synthetic=provenance.at("synthetic").boolean();
    if((cli.mode=="fixture"&&!synthetic)||(synthetic&&cli.mode!="fixture"&&cli.mode!="qualify"))throw std::runtime_error("synthetic input must use fixture/qualify mode; real input cannot masquerade as fixture");
    if(!synthetic && provenance.at("status").string()!="CORPUS_INPUTS_STAGED")throw std::runtime_error("input staging incomplete");
    auto snapshot_path=boundary.input("snapshot.txt"), coordinates_path=boundary.input("coordinates.txt");
    std::uint64_t snapshot_size=fs::file_size(snapshot_path),coordinates_size=fs::file_size(coordinates_path);
    if(snapshot_size!=provenance.at("files").at("snapshot.txt").at("bytes").unsigned_integer() || coordinates_size!=provenance.at("files").at("coordinates.txt").at("bytes").unsigned_integer())throw std::runtime_error("input byte count mismatch");
    auto snapshot_time=fs::last_write_time(snapshot_path), coordinates_time=fs::last_write_time(coordinates_path);
    const auto& b=provenance.at("bbox_microdegrees").array();
    if(b.size()!=4)throw std::runtime_error("bbox must have four microdegree bounds");
    // BJ and XA have positive geographic coordinates; reject unexpected signs/overflow explicitly.
    corpus::BBox bbox;
    std::int64_t* bounds[]={&bbox.longitude_min,&bbox.longitude_max,&bbox.latitude_min,&bbox.latitude_max};
    for(std::size_t i=0;i<4;++i){auto n=b[i].unsigned_integer();if(n>180000000)throw std::runtime_error("bbox geographic range");*bounds[i]=static_cast<std::int64_t>(n);}
    auto mapping_start=Clock::now();
    auto policy_name=config.contains("routing_graph_policy")?config.at("routing_graph_policy").string():std::string(corpus::legacy_policy_name);
    if(policy_name!=corpus::legacy_policy_name && policy_name!=corpus::fifo_policy_name)throw std::runtime_error("unsupported routing_graph_policy");
    const bool fifo=policy_name==corpus::fifo_policy_name;
    auto graph=corpus::Graph::load(snapshot_path,coordinates_path,bbox,fifo?corpus::RoutingPolicy::CanonicalFifoLpf:corpus::RoutingPolicy::LegacySlots);
    auto mapping_seconds=seconds(mapping_start);
    if(fs::file_size(snapshot_path)!=snapshot_size || fs::file_size(coordinates_path)!=coordinates_size || fs::last_write_time(snapshot_path)!=snapshot_time || fs::last_write_time(coordinates_path)!=coordinates_time)throw std::runtime_error("staged input changed during read");
    if(cli.mode=="qualify") {
        if(!fifo)throw std::runtime_error("qualify requires the FIFO LPF policy");
        boundary.create_output();auto r=graph_record(graph,boundary,provenance,cli.paths.city);
        r["schema_version"]=Json(2);r["status"]=Json("FIFO_GRAPH_QUALIFIED");r["mode"]=Json("qualify");r["production"]=Json(false);
        r["run_id"]=Json(cli.paths.run_id);r["config"]=config;
        r["scope"]=Json("complete city graph and FIFO LPF construction; no corpus rows generated");
        r["metrics"]=Json::Object{{"total_seconds",real(seconds(total_start))},{"peak_rss_bytes",peak_rss()}};
        sdc::io::write_text_exclusive(boundary.output()/"manifest.json",Json(r).dump()+"\n");
        std::cout<<Json(Json::Object{{"status","FIFO_GRAPH_QUALIFIED"},{"city",cli.paths.city},{"audit",lpf_audit(graph.audit)}}).dump()<<'\n';return 0;
    }
    corpus::SamplingConfig sampling;
    sampling.method=config.at("sampler").string(); sampling.seed=config.at("seed").unsigned_integer();
    sampling.cluster_count=integer(config,"cluster_count");sampling.candidates_per_grid=integer(config,"candidates_per_grid");
    sampling.fine_grid_dim=integer(config,"fine_grid_dim");sampling.min_distance=integer(config,"min_distance");sampling.max_distance=integer(config,"max_distance");
    sampling.start_second=cli.start_override>=0 ? cli.start_override:integer(config,"start_second");
    sampling.interval_seconds=integer(config,"interval_seconds");sampling.point_step_seconds=integer(config,"point_step_seconds");
    const auto points=sampling.points_per_group();
    if(sampling.seed!=20260905)throw std::runtime_error("fixed seed must be 20260905");
    const bool production=cli.mode=="generate";
    if(production) {
        if(fifo) {
            if(config.at("start_time_status").string()!="fixed_by_adr_0010" || sampling.start_second!=28800 || sampling.interval_seconds!=14400 || sampling.point_step_seconds!=900 || points!=16)
                throw std::runtime_error("FIFO production requires fixed 08:00-12:00, 16 points per group");
        } else if(config.at("start_time_status").string()!="locked_after_current_snapshot_probe" || sampling.interval_seconds!=3600 || sampling.point_step_seconds!=900)
            throw std::runtime_error("historical slot production requires its original locked four-point contract");
        if(sampling.min_distance!=2 || sampling.max_distance!=4 || sampling.fine_grid_dim!=2)throw std::runtime_error("production spatial contract is fixed");
        sampling.split_rows=cli.paths.city=="BJ" ? std::array<std::uint64_t,3>{800000,100000,100000}:std::array<std::uint64_t,3>{400000,50000,50000};
    } else if(cli.mode=="probe") {
        auto groups=config.at("probe_groups").unsigned_integer();
        if(groups<1||groups>10000)throw std::runtime_error("probe groups must be 1..10000");
        sampling.split_rows={groups*points,0,0};
    } else sampling.split_rows={(cli.fixture_groups-4)*points,2*points,2*points};
    auto snapshot_id=provenance.at("snapshot_id").string();
    if(snapshot_id.empty())throw std::runtime_error("snapshot identity is empty");
    if(fifo)snapshot_id+="::"+policy_name;
    auto plan_start=Clock::now();
    auto plan=corpus::make_query_plan(graph,sampling,cli.paths.city,snapshot_id);
    auto plan_seconds=seconds(plan_start);
    std::uint64_t expected=0;for(auto n:sampling.split_rows)expected+=n;
    if(plan.queries.size()!=expected)throw std::runtime_error("query plan row count mismatch");
    std::vector<std::size_t> execution_order(plan.queries.size());std::iota(execution_order.begin(),execution_order.end(),0);
    std::sort(execution_order.begin(),execution_order.end(),[&](auto a,auto b){const auto& x=plan.queries[a];const auto& y=plan.queries[b];return std::tie(x.source,x.departure,x.destination,x.query_id)<std::tie(y.source,y.departure,y.destination,y.query_id);});
    boundary.create_output();
    Json::Object query_counts;for(std::size_t i=0;i<3;++i)query_counts[corpus::split_name(static_cast<corpus::Split>(i))]=Json(sampling.split_rows[i]);
    auto model=graph_record(graph,boundary,provenance,cli.paths.city);
    Json::Object plan_info{{"status","QUERY_PLAN_FROZEN"},{"plan_id",cli.paths.run_id},{"point_rows",expected},{"split_rows",query_counts},{"candidate_groups",plan.candidate_groups},{"reachable_groups",plan.reachable_groups},{"representatives",plan.representative_count},{"seed",sampling.seed},{"method",sampling.method},{"start_second",sampling.start_second},{"interval_seconds",sampling.interval_seconds},{"point_step_seconds",sampling.point_step_seconds},{"points_per_group",points}};
    sdc::io::write_text_exclusive(boundary.output()/"query_plan.json",Json(plan_info).dump()+"\n");
    auto budget=config.at("shard_byte_budget").unsigned_integer();
    if(budget<1048576 || budget>268435456)throw std::runtime_error("shard byte budget must be 1..256 MiB");
    std::unique_ptr<corpus::CorpusWriter> writer;
    if(cli.mode!="probe")writer=std::make_unique<corpus::CorpusWriter>(boundary.output(),graph,budget);
    corpus::TdDijkstra router(graph);
    std::uint64_t calls=0,written=0,max_route_bytes=0;
    double routing_seconds=0,route_handoff_seconds=0;
    std::vector<std::int64_t> route_lengths,travel_times,occupied_cells;
    std::size_t group_end=0;
    for(std::size_t position=0;position<execution_order.size();++position) {
        const auto index=execution_order[position];
        const auto& query=plan.queries[index];
        if(position==group_end) {
            std::vector<std::int64_t> targets;
            for(group_end=position;group_end<execution_order.size();++group_end) {
                const auto& next=plan.queries[execution_order[group_end]];
                if(next.source!=query.source || next.departure!=query.departure)break;
                targets.push_back(next.destination);
            }
            auto t=Clock::now();router.run(query.source,query.departure,targets);routing_seconds+=seconds(t);++calls;
            if(calls%256==0)std::cerr<<Json(Json::Object{{"event","routing_progress"},{"sssp_calls",calls},{"written_rows",written},{"expected_rows",expected},{"elapsed_seconds",real(seconds(total_start))}}).dump()<<'\n';
        }
        auto t=Clock::now();auto route=router.route(query);route_handoff_seconds+=seconds(t);
        if(route.status!="ok")throw std::runtime_error("frozen query did not produce a valid route: "+route.status);
        max_route_bytes=std::max(max_route_bytes,static_cast<std::uint64_t>(route.node_ids.capacity()*sizeof(std::int64_t)+route.arrival_times.capacity()*sizeof(std::int64_t)));
        if(writer)writer->append(route);
        else {
            route_lengths.push_back(static_cast<std::int64_t>(route.node_ids.size()));travel_times.push_back(route.travel_time);
            std::array<bool,1024> occupied{};for(auto id:route.node_ids)occupied[graph.nodes[graph.index(id)].coarse_cell]=true;
            occupied_cells.push_back(std::count(occupied.begin(),occupied.end(),true));
        }
        ++written;
    }
    if(written!=expected)throw std::runtime_error("route or conversion dropped rows");
    Json::Object result=writer?writer->finish().object():Json::Object{};
    auto config_record=config.object();config_record["departure_start_seconds"]=Json(sampling.start_second);
    result["schema_version"]=Json(fifo?2:1);result["status"]=Json(production?"CORPUS_GENERATED":cli.mode=="probe"?"PROBE_COMPLETED":"FIXTURE_GENERATED");
    result["production"]=Json(production);result["mode"]=Json(cli.mode);result["city"]=Json(cli.paths.city);result["date"]=Json("20160101");result["graph_state"]=Json("G0");
    result["run_id"]=Json(cli.paths.run_id);result["plan_id"]=Json(cli.paths.run_id);result["snapshot_id"]=Json(snapshot_id);result["method"]=Json(sampling.method);result["router"]=Json("td_dijkstra");result["config"]=Json(config_record);
    result["point_rows"]=Json(written);result["counts"]=Json(query_counts);result["query_plan"]=Json(plan_info);
    result["route_context"]=Json::Object{{"city",cli.paths.city},{"snapshot_id",snapshot_id},{"status","ok"},{"status_encoding",Json::Object{{"0","ok"}}}};
    result["identity_scope"]=Json("numeric query_id and interval_id are local to plan_id; references include city,snapshot_id,plan_id; semantic point key is city,snapshot_id,source,destination,departure");
    result["encoding_version"]=Json("fixed_bbox_floor32_last_visit_progress_v1");
    result["metrics"]=Json::Object{{"id_mapping_seconds",real(mapping_seconds)},{"query_plan_seconds",real(plan_seconds)},{"routing_seconds",real(routing_seconds)},{"route_reconstruction_seconds",real(route_handoff_seconds)},{"sssp_calls",calls},{"total_seconds",real(seconds(total_start))},{"peak_rss_bytes",peak_rss()}};
    result["memory_accounting"]=Json::Object{{"writer_split_budget_bytes",budget},{"writer_all_split_budget_bytes",budget*3},{"max_live_route_array_capacity_bytes",max_route_bytes},{"description","at most one reconstructed RouteRecord handed to encoder/writer; graph, fixed scalar query plan, SSSP arrays and that one route are separate from the three writer buffers; no global retained routes or padded tensors"}};
    for(auto& item:model)result[item.first]=std::move(item.second);
    result["routing_execution"]=Json("one TD Dijkstra per source/departure; stop after all requested targets settled; same requested-target answers as full search");
    if(cli.mode=="probe") {
        Json::Array interval_ids;
        for(std::size_t i=0;i<plan.queries.size();i+=points)interval_ids.emplace_back(plan.queries[i].interval_id);
        result["probe"]=Json::Object{{"scope","fixed-window integration/capacity probe on a final-train OD prefix; not window selection or production acceptance"},{"interval_ids",interval_ids},{"route_node_count",summary(route_lengths)},{"travel_seconds",summary(travel_times)},{"occupied_cells",summary(occupied_cells)},{"all_selected_routes_valid",true}};
    }
    sdc::io::write_text_exclusive(boundary.output()/"manifest.json",Json(result).dump()+"\n");
    std::cout<<Json(Json::Object{{"status",result.at("status")},{"city",cli.paths.city},{"point_rows",written},{"sssp_calls",calls},{"manifest",boundary.relative(boundary.output()/"manifest.json")},{"elapsed_seconds",real(seconds(total_start))}}).dump()<<'\n';
    return 0;
}
}
int main(int argc,char** argv) {
    try{return execute(parse(argc,argv));}
    catch(const sdc::io::BoundaryViolation& e){std::cerr<<"BOUNDARY_ERROR: "<<e.what()<<'\n';return 3;}
    catch(const std::exception& e){std::cerr<<"OFFLINE_ERROR: "<<e.what()<<'\n';return 2;}
}
