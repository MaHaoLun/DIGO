#include "corpus/online.hpp"
#include "corpus/routing.hpp"
#include "corpus/writer.hpp"
#include "sdc/io.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
    rusage usage{}; if(getrusage(RUSAGE_SELF,&usage))throw std::runtime_error("cannot read peak RSS");
#ifdef __APPLE__
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss)*1024;
#endif
}
std::uint64_t uint_arg(const std::string& x) { if(x.empty()||x.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("unsigned integer expected");return std::stoull(x); }
struct Cli {
    sdc::io::Options paths;
    std::string config="configs/online.json",previous_input,membership_input,mode="generate";
    std::uint64_t fixture_groups=25;
    bool fixture_given=false;
};
Cli parse(int argc,char** argv) {
    Cli c;
    for(int i=1;i<argc;++i) {
        std::string k=argv[i];
        if(k=="--help") { std::cout<<"sdc_online --workspace-root ROOT --input-dir G1_PACKAGE --previous-input-dir G0_PACKAGE --membership-input STAGED_CSV --city BJ|XA --run-id NEW [--config configs/online.json] [--mode generate|fixture] [--fixture-groups 25 (fixture only)]\n";std::exit(0); }
        if(i+1==argc)throw std::runtime_error("missing value: "+k);
        std::string v=argv[++i];
        if(k=="--workspace-root")c.paths.workspace_root=v;
        else if(k=="--input-dir")c.paths.input_dir=v;
        else if(k=="--previous-input-dir")c.previous_input=v;
        else if(k=="--membership-input")c.membership_input=v;
        else if(k=="--city")c.paths.city=v;
        else if(k=="--run-id")c.paths.run_id=v;
        else if(k=="--config")c.config=v;
        else if(k=="--mode")c.mode=v;
        else if(k=="--fixture-groups"){c.fixture_given=true;c.fixture_groups=uint_arg(v);}
        else throw std::runtime_error("unknown option: "+k);
    }
    if(c.paths.city!="BJ"&&c.paths.city!="XA")throw std::runtime_error("city must be BJ or XA");
    if(c.mode!="generate"&&c.mode!="fixture")throw std::runtime_error("mode must be generate or fixture");
    if(c.previous_input.empty()||c.membership_input.empty())throw std::runtime_error("previous graph and membership inputs required");
    if(c.fixture_given&&c.mode!="fixture")throw std::runtime_error("fixture budget cannot alter production counts");
    if(c.fixture_groups<25||c.fixture_groups>10000||c.fixture_groups%25)throw std::runtime_error("fixture groups must be a multiple of 25 in 25..10000");
    return c;
}
Json lpf_audit(const corpus::GraphAudit& a) {
    return Json::Object{{"source_varying_rows",a.source_varying_rows},{"canonical_pair_count",a.canonical_pair_count},
        {"canonical_varying_pairs_before",a.canonical_varying_pairs_before},{"canonical_varying_pairs_after",a.canonical_varying_pairs_after},
        {"changed_pairs",a.changed_pairs},{"changed_knots",a.changed_knots},{"sum_raise",a.sum_raise},{"max_raise",a.max_raise},
        {"raw_lpf_violations",a.raw_lpf_violations},{"repaired_continuous_violations",a.repaired_continuous_violations},
        {"repaired_integer_violations",a.repaired_integer_violations},{"reverse_rows_ignored",a.reverse_rows_ignored},{"duplicate_rows",a.duplicate_rows}};
}
Json::Object graph_record(const corpus::Graph& graph, const sdc::io::Boundaries& boundary,
                          const Json& provenance, const std::string& city, const std::string& date, const std::string& state, const std::string& prefix) {
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
    Json::Object r{{"city",city},{"date",date},{"graph_state",state},{"snapshot_id",identity},
        {"source_snapshot_id",source},{"routing_model_id",identity},{"routing_graph_policy",policy},
        {"inputs",Json::Object{{"snapshot",Json::Object{{"path",boundary.relative(boundary.input("snapshot.txt"))},{"bytes",static_cast<std::uint64_t>(fs::file_size(boundary.input("snapshot.txt")))}}},
            {"coordinates",Json::Object{{"path",boundary.relative(boundary.input("coordinates.txt"))},{"bytes",static_cast<std::uint64_t>(fs::file_size(boundary.input("coordinates.txt")))}}},
            {"provenance",boundary.relative(boundary.input("provenance.json"))}}},
        {"graph",Json::Object{{"nodes",static_cast<std::uint64_t>(graph.nodes.size())},{"input_arcs",graph.input_arc_count},
            {"effective_arcs",arcs},{"input_time_varying_arcs",graph.input_time_varying_arc_count},{"effective_time_varying_arcs",varying}}},
        {"time",Json::Object{{"departure_domain","[0,86400) UTC+08 observation-day seconds"},{"slot_seconds",900},{"slot_count",96},
            {"out_of_day_lookup",fifo?"periodic observation-day LPF, exact-ceil integer evaluation":"floor(unwrapped_arrival/900)%96; periodic same-day slots, not G1"},
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
        sdc::io::write_text_exclusive(boundary.output()/(prefix+"lpf_knots.txt"),out.str());
        r["lpf"]=Json::Object{{"knots_path",(prefix+"lpf_knots.txt")},{"bytes",static_cast<std::uint64_t>(fs::file_size(boundary.output()/(prefix+"lpf_knots.txt")))},
            {"correction_method","minimal_upward_periodic_majorant_v1"},{"knot_count",96},{"knot_step_seconds",900},{"period_seconds",86400},
            {"rounding","ceil_exact_integer"},{"audit",lpf_audit(graph.audit)}};
    }
    r["mapping"]=Json::Object{{"registry",(prefix+"grid_registry.tsv")},{"bbox_microdegrees",Json::Object{{"lon_min",graph.bbox.longitude_min},{"lon_max",graph.bbox.longitude_max},{"lat_min",graph.bbox.latitude_min},{"lat_max",graph.bbox.latitude_max}}},{"coarse_dim",32},{"fine_dim",64},{"cell_order","latitude row-major y*dim+x; floor; inclusive max clipped"},{"odt","normalized exact o_lon,o_lat,d_lon,d_lat,departure; [-1,1]"}};
    std::ostringstream registry;registry<<"node_id\tlongitude_microdegrees\tlatitude_microdegrees\tcoarse_cell\tfine_cell\n";
    for(const auto& node:graph.nodes)registry<<node.id<<'\t'<<node.x<<'\t'<<node.y<<'\t'<<node.coarse_cell<<'\t'<<node.fine_cell<<'\n';
    sdc::io::write_text_exclusive(boundary.output()/(prefix+"grid_registry.tsv"),registry.str());
    return r;
}
struct InputStamp {
    fs::path path; std::uint64_t bytes; fs::file_time_type time;
    explicit InputStamp(fs::path p):path(std::move(p)),bytes(fs::file_size(path)),time(fs::last_write_time(path)){}
    void verify()const {if(fs::file_size(path)!=bytes||fs::last_write_time(path)!=time)throw std::runtime_error("staged input changed during execution");}
};
Json provenance(const sdc::io::Boundaries& b,const Cli& cli,const std::string& date,const std::string& state,std::vector<InputStamp>& stamps) {
    auto p=b.input("provenance.json");stamps.emplace_back(p);auto v=sdc::io::read_json(p);
    if(v.at("city").string()!=cli.paths.city||v.at("date").string()!=date||v.at("graph_state").string()!=state)throw std::runtime_error("city/date/graph state provenance mismatch");
    if(v.at("synthetic").boolean()!=(cli.mode=="fixture"))throw std::runtime_error("fixture mode requires synthetic inputs; production requires real inputs");
    if(v.at("snapshot_id").string().empty())throw std::runtime_error("empty snapshot identity");
    for(const auto* name:{"snapshot.txt","coordinates.txt"}){
        auto path=b.input(name);stamps.emplace_back(path);
        if(fs::file_size(path)!=v.at("files").at(name).at("bytes").unsigned_integer())throw std::runtime_error("input byte count mismatch");
    }
    if(cli.mode=="generate") {
        if(v.at("status").string()!="CORPUS_INPUTS_STAGED")throw std::runtime_error("incomplete corpus input staging");
        auto path=b.input("source_snapshot_manifest.json");stamps.emplace_back(path);
        if(fs::file_size(path)!=v.at("files").at("source_snapshot_manifest.json").at("bytes").unsigned_integer())throw std::runtime_error("source manifest byte count mismatch");
        auto source=sdc::io::read_json(path);const auto& c=source.at("completeness");
        if(source.at("status").string()!="SNAPSHOT_GENERATED"||source.at("city").string()!=cli.paths.city||source.at("observation_date").string()!=date||source.at("graph_state").string()!=state||source.at("snapshot_id").string()!=v.at("snapshot_id").string()||!c.at("declared_complete").boolean()||!c.at("selected_city_day_has_observations").boolean()||c.at("expected_raw_files").unsigned_integer()!=30||c.at("processed_raw_files").unsigned_integer()!=30)throw std::runtime_error("production requires a complete matching 30-shard source snapshot");
        if(source.at("outputs").at("snapshot.txt").at("bytes").unsigned_integer()!=fs::file_size(b.input("snapshot.txt")))throw std::runtime_error("source/staged snapshot byte count mismatch");
        const auto& coordinate_source=source.at("city_static_inputs").at("coordinates").string();
        if(v.at("files").at("coordinates.txt").at("source").string()!=coordinate_source)throw std::runtime_error("source/staged coordinate provenance path mismatch");
        std::uint64_t coordinate_records=0;
        for(const auto& item:source.at("inputs").array())if(item.at("path").string()==coordinate_source){
            ++coordinate_records;if(item.at("bytes").unsigned_integer()!=fs::file_size(b.input("coordinates.txt")))throw std::runtime_error("source/staged coordinate byte count mismatch");
        }
        if(coordinate_records!=1)throw std::runtime_error("source coordinate provenance requires exactly one input record");
    }
    return v;
}
corpus::BBox read_bbox(const Json& p) {
    const auto& a=p.at("bbox_microdegrees").array();if(a.size()!=4)throw std::runtime_error("bbox must have four bounds");
    corpus::BBox b;std::int64_t* v[]={&b.longitude_min,&b.longitude_max,&b.latitude_min,&b.latitude_max};
    for(std::size_t i=0;i<4;++i){auto n=a[i].unsigned_integer();if(n>180000000)throw std::runtime_error("bbox geographic range");*v[i]=static_cast<std::int64_t>(n);}
    return b;
}
Json ints(const std::vector<std::int64_t>& values) {Json::Array a;for(auto v:values)a.emplace_back(v);return a;}
int execute(const Cli& cli) {
    auto start=Clock::now();const bool production=cli.mode=="generate";
    sdc::io::Boundaries boundary(cli.paths);auto old_options=cli.paths;old_options.input_dir=cli.previous_input;
    sdc::io::Boundaries old_boundary(old_options);
    fs::path config_raw(cli.config);if(config_raw.is_absolute())throw std::runtime_error("config must be workspace relative");
    for(const auto& p:config_raw)if(p=="..")throw std::runtime_error("config parent traversal forbidden");
    auto config_path=boundary.root()/config_raw;boundary.verify_regular(config_path);auto config=sdc::io::read_json(config_path);
    auto budget=config.at("shard_byte_budget").unsigned_integer();if(budget<1048576||budget>268435456)throw std::runtime_error("shard budget must be 1..256 MiB");
    std::vector<InputStamp> stamps;stamps.emplace_back(config_path);
    auto current=provenance(boundary,cli,"20160102","G1",stamps),previous=provenance(old_boundary,cli,"20160101","G0",stamps);
    if(current.at("bbox_microdegrees").dump()!=previous.at("bbox_microdegrees").dump())throw std::runtime_error("G0/G1 bbox differs");
    if(config.at("routing_graph_policy").string()!=corpus::fifo_policy_name||config.at("sampler").string()!="hierarchical_grid"||config.at("seed").unsigned_integer()!=20260905||config.at("fine_grid_dim").unsigned_integer()!=2||config.at("min_distance").unsigned_integer()!=2||config.at("max_distance").unsigned_integer()!=4||config.at("start_second").unsigned_integer()!=43200||config.at("interval_seconds").unsigned_integer()!=1800||config.at("point_step_seconds").unsigned_integer()!=900||config.at("start_time_status").string()!="fixed_by_adr_0013")throw std::runtime_error("fixed G1 sampling/FIFO contract differs");
    if(config.at("absolute_shift_seconds").unsigned_integer()!=5||config.at("relative_shift_threshold").dump()!="0.1"||config.at("hotspot_fraction").dump()!="0.2"||config.at("targeted_fraction").dump()!="0.2")throw std::runtime_error("fixed G1 detector/mixture contract differs");
    auto mapping_start=Clock::now();auto bbox=read_bbox(current);
    auto graph=corpus::Graph::load(boundary.input("snapshot.txt"),boundary.input("coordinates.txt"),bbox,corpus::RoutingPolicy::CanonicalFifoLpf);
    auto old_graph=corpus::Graph::load(old_boundary.input("snapshot.txt"),old_boundary.input("coordinates.txt"),bbox,corpus::RoutingPolicy::CanonicalFifoLpf);
    auto mapping_seconds=seconds(mapping_start);
    auto new_id=current.at("snapshot_id").string()+"::"+corpus::fifo_policy_name,old_id=previous.at("snapshot_id").string()+"::"+corpus::fifo_policy_name;
    if(fs::path(cli.membership_input).filename()!="g0_od_membership.csv")throw std::runtime_error("membership filename must be g0_od_membership.csv");
    auto membership_path=boundary.input(cli.membership_input);stamps.emplace_back(membership_path);
    auto membership_provenance_path=boundary.input(boundary.relative(membership_path.parent_path()/"provenance.json"));stamps.emplace_back(membership_provenance_path);
    auto mp=sdc::io::read_json(membership_provenance_path);
    if(mp.at("city").string()!=cli.paths.city||mp.at("date").string()!="20160101"||mp.at("graph_state").string()!="G0"||mp.at("snapshot_id").string()!=old_id||mp.at("status").string()!="G0_OD_MEMBERSHIP_STAGED")throw std::runtime_error("G0 membership provenance mismatch");
    if(mp.at("synthetic").boolean()!=!production)throw std::runtime_error("membership synthetic provenance mismatch");
    if(mp.at("files").at("g0_od_membership.csv").at("bytes").unsigned_integer()!=fs::file_size(membership_path))throw std::runtime_error("G0 membership byte count mismatch");
    auto membership=corpus::read_g0_membership(membership_path);
    if(mp.at("od_groups").unsigned_integer()!=membership.size())throw std::runtime_error("membership total OD count mismatch");
    std::array<std::uint64_t,3> old_groups{};for(const auto& kv:membership)++old_groups.at(static_cast<std::size_t>(kv.second));
    if(production){const auto want=cli.paths.city=="BJ"?std::array<std::uint64_t,3>{50000,6250,6250}:std::array<std::uint64_t,3>{25000,3125,3125};if(old_groups!=want)throw std::runtime_error("full G0 OD membership counts required");
        std::uint64_t total=0;for(std::size_t i=0;i<3;++i){auto n=old_groups[i]*16;total+=n;if(mp.at("split_rows").at(corpus::split_name(static_cast<corpus::Split>(i))).unsigned_integer()!=n)throw std::runtime_error("G0 full point count provenance mismatch");}
        if(mp.at("point_rows").unsigned_integer()!=total)throw std::runtime_error("G0 total point count provenance mismatch");}
    for(std::size_t i=0;i<3;++i)if(mp.at("split_groups").at(corpus::split_name(static_cast<corpus::Split>(i))).unsigned_integer()!=old_groups[i])throw std::runtime_error("membership OD count provenance mismatch");
    corpus::OnlineConfig online_config;
    online_config.split_rows=cli.paths.city=="BJ"?std::array<std::uint64_t,2>{80000,20000}:std::array<std::uint64_t,2>{40000,10000};
    if(!production)online_config.split_rows={cli.fixture_groups*8/5,cli.fixture_groups*2/5};
    online_config.qualification_chunk_groups=config.at("qualification_chunk_groups").unsigned_integer();
    if(online_config.qualification_chunk_groups<1||online_config.qualification_chunk_groups>65536)throw std::runtime_error("qualification chunk must be 1..65536 groups");
    auto detection_start=Clock::now();auto detection=corpus::detect_shifts(old_graph,graph);auto detection_seconds=seconds(detection_start);
    auto plan_start=Clock::now();auto plan=corpus::make_online_plan(old_graph,graph,online_config,membership,cli.paths.city,new_id,old_id,detection);auto plan_seconds=seconds(plan_start);
    std::uint64_t expected=online_config.split_rows[0]+online_config.split_rows[1];
    if(plan.queries.size()!=expected||plan.targeted.size()!=expected||plan.qualification_evidence.size()!=expected)throw std::runtime_error("frozen plan size mismatch");
    for(const auto& s:stamps)s.verify();
    boundary.create_output();auto model=graph_record(graph,boundary,current,cli.paths.city,"20160102","G1","");
    auto previous_model=graph_record(old_graph,old_boundary,previous,cli.paths.city,"20160101","G0","previous_");
    Json::Object counts{{"adaptation",online_config.split_rows[0]},{"post_update_test",online_config.split_rows[1]}};
    auto plan_info=plan.capacity_report().object();
    plan_info["status"]="QUERY_PLAN_FROZEN";plan_info["plan_id"]=cli.paths.run_id;plan_info["point_rows"]=expected;plan_info["split_rows"]=counts;plan_info["seed"]=online_config.sampling.seed;plan_info["method"]="hierarchical_grid";plan_info["start_second"]=43200;plan_info["interval_seconds"]=1800;plan_info["point_step_seconds"]=900;plan_info["points_per_group"]=2;
    sdc::io::write_text_exclusive(boundary.output()/"query_plan.json",Json(plan_info).dump()+"\n");
    auto detection_report=detection.report();sdc::io::write_text_exclusive(boundary.output()/"shift_detection.json",detection_report.dump()+"\n");
    corpus::CorpusWriter writer(boundary.output(),graph,budget,{corpus::Split::adaptation,corpus::Split::post_update_test});
    auto evidence_path=boundary.output()/"online_evidence.jsonl";sdc::io::write_text_exclusive(evidence_path,"");std::ofstream evidence_out(evidence_path,std::ios::app);
    if(!evidence_out)throw std::runtime_error("cannot open evidence output");
    std::vector<std::size_t> order(expected);std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](auto a,auto b){const auto& x=plan.queries[a];const auto& y=plan.queries[b];return std::tie(x.source,x.departure,x.destination,x.query_id)<std::tie(y.source,y.departure,y.destination,y.query_id);});
    corpus::TdDijkstra router(graph),old_router(old_graph);std::uint64_t calls=0,old_calls=0,written=0,max_route_bytes=0;
    double routing_seconds=0,old_routing_seconds=0,handoff_seconds=0;std::size_t group_end=0;
    for(std::size_t position=0;position<order.size();++position){
        auto index=order[position];const auto& query=plan.queries[index];
        if(position==group_end){
            std::vector<std::int64_t> targets,old_targets;
            for(group_end=position;group_end<order.size();++group_end){auto next_index=order[group_end];const auto& next=plan.queries[next_index];if(next.source!=query.source||next.departure!=query.departure)break;targets.push_back(next.destination);if(plan.targeted[next_index])old_targets.push_back(next.destination);}
            auto t=Clock::now();router.run(query.source,query.departure,targets);routing_seconds+=seconds(t);++calls;
            if(!old_targets.empty()){t=Clock::now();old_router.run(query.source,query.departure,old_targets);old_routing_seconds+=seconds(t);++old_calls;}
            if(calls%256==0)std::cerr<<Json(Json::Object{{"event","online_routing_progress"},{"sssp_calls",calls},{"written_rows",written},{"expected_rows",expected},{"elapsed_seconds",real(seconds(start))}}).dump()<<'\n';
        }
        auto t=Clock::now();auto route=router.route(query);corpus::RouteRecord old_route;
        if(route.status!="ok")throw std::runtime_error("frozen G1 query route failed");
        if(plan.targeted[index]){auto old_query=query;old_query.snapshot_id=old_id;old_route=old_router.route(old_query);if(old_route.status!="ok")throw std::runtime_error("frozen target G0 evidence route failed");}
        auto evidence=corpus::analyze_point(old_graph,graph,detection,route,plan.targeted[index]?&old_route:nullptr);
        if(plan.targeted[index]&&evidence.json().dump()!=plan.qualification_evidence[index].json().dump())throw std::runtime_error("final target evidence changed since qualification");
        handoff_seconds+=seconds(t);
        max_route_bytes=std::max(max_route_bytes,static_cast<std::uint64_t>((route.node_ids.capacity()+route.arrival_times.capacity()+old_route.node_ids.capacity()+old_route.arrival_times.capacity())*sizeof(std::int64_t)));
        auto row=evidence.json().object();row["query_id"]=query.query_id;row["interval_id"]=query.interval_id;row["row_index"]=query.row_index;row["source"]=query.source;row["destination"]=query.destination;row["departure"]=query.departure;row["split"]=corpus::split_name(query.split);row["stratum"]=plan.targeted[index]?"targeted":"ordinary";
        row["old_node_ids"]=plan.targeted[index]?ints(old_route.node_ids):Json();row["old_arrival_times"]=plan.targeted[index]?ints(old_route.arrival_times):Json();row["old_travel_time"]=plan.targeted[index]?Json(old_route.travel_time):Json();
        evidence_out<<Json(row).dump()<<'\n';if(!evidence_out)throw std::runtime_error("evidence write failed");writer.append(route);++written;
    }
    evidence_out.close();if(!evidence_out)throw std::runtime_error("evidence close failed");
    if(written!=expected)throw std::runtime_error("online generation dropped rows");
    for(const auto& s:stamps)s.verify();
    auto result=writer.finish().object();for(auto& kv:model)result[kv.first]=std::move(kv.second);
    result["schema_version"]=2;result["status"]=production?"CORPUS_GENERATED":"FIXTURE_GENERATED";result["production"]=production;result["mode"]=cli.mode;result["run_id"]=cli.paths.run_id;result["plan_id"]=cli.paths.run_id;result["point_rows"]=written;result["counts"]=counts;result["query_plan"]=plan_info;result["config"]=config;result["method"]="hierarchical_grid";result["router"]="td_dijkstra";result["previous_graph"]=previous_model;
    result["route_context"]=Json::Object{{"city",cli.paths.city},{"snapshot_id",new_id},{"status","ok"},{"status_encoding",Json::Object{{"0","ok"}}}};
    result["encoding_version"]="fixed_bbox_floor32_last_visit_progress_v1";
    result["identity_scope"]="numeric query_id and interval_id are local to plan_id; references include city,snapshot_id,plan_id; semantic point key is city,snapshot_id,source,destination,departure";
    result["online"]=Json::Object{{"detection_path","shift_detection.json"},{"detection",detection_report},{"selection_plan",plan_info},{"evidence_path","online_evidence.jsonl"},{"evidence_bytes",static_cast<std::uint64_t>(fs::file_size(evidence_path))},{"g0_membership",Json::Object{{"path",boundary.relative(membership_path)},{"bytes",static_cast<std::uint64_t>(fs::file_size(membership_path))},{"provenance",boundary.relative(membership_provenance_path)}}},{"config",config},{"scope","historical full-day snapshot refresh; not a strictly causal real-time service; no SSP/replay training"},{"ordinary_old_route_comparison",false}};
    result["metrics"]=Json::Object{{"id_mapping_seconds",real(mapping_seconds)},{"detection_seconds",real(detection_seconds)},{"query_plan_seconds",real(plan_seconds)},{"routing_seconds",real(routing_seconds)},{"old_evidence_routing_seconds",real(old_routing_seconds)},{"route_reconstruction_and_evidence_seconds",real(handoff_seconds)},{"sssp_calls",calls},{"old_evidence_sssp_calls",old_calls},{"qualification_old_sssp_calls",plan.qualification_old_sssp},{"qualification_new_sssp_calls",plan.qualification_new_sssp},{"total_seconds",real(seconds(start))},{"peak_rss_bytes",peak_rss()}};
    result["memory_accounting"]=Json::Object{{"writer_split_budget_bytes",budget},{"writer_all_split_budget_bytes",budget*2},{"max_live_route_array_capacity_bytes",max_route_bytes},{"description","two graph models, scalar frozen plan, two SSSP workspaces, at most one current route and one old target route, two bounded writer buffers; evidence streamed"}};
    result["routing_execution"]="qualify target candidates in grouped searches, freeze all identities, then regenerate every G1 label with exact target-set TD Dijkstra; solve same-clock G0 again only for selected targets";
    sdc::io::write_text_exclusive(boundary.output()/"manifest.json",Json(result).dump()+"\n");
    std::cout<<Json(Json::Object{{"status",result.at("status")},{"city",cli.paths.city},{"point_rows",written},{"sssp_calls",calls},{"manifest",boundary.relative(boundary.output()/"manifest.json")},{"elapsed_seconds",real(seconds(start))}}).dump()<<'\n';return 0;
}
}
int main(int argc,char** argv){try{return execute(parse(argc,argv));}catch(const sdc::io::BoundaryViolation& e){std::cerr<<"BOUNDARY_ERROR: "<<e.what()<<'\n';return 3;}catch(const std::exception& e){std::cerr<<"ONLINE_ERROR: "<<e.what()<<'\n';return 2;}}
