#include "sdc/io.hpp"
#include "sdc/snapshot.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>

namespace sdc::io {
namespace fs = std::filesystem;
namespace {
using Object = Json::Object;
using Array = Json::Array;

class Output {
public:
    explicit Output(const fs::path& path) {
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0644);
        if (fd_ < 0) throw std::runtime_error("exclusive output open failed: " + path.filename().string());
        buffer_.reserve(1024 * 1024 + 4096);
    }
    ~Output() { if (fd_ >= 0) ::close(fd_); }
    void append(const std::string& text) { buffer_ += text; if (buffer_.size() >= 1024 * 1024) flush(); }
    void finish() {
        flush();
        if (::fsync(fd_) != 0) throw std::runtime_error("output fsync failed");
        int fd = fd_; fd_ = -1; if (::close(fd) != 0) throw std::runtime_error("output close failed");
    }
private:
    int fd_ = -1;
    std::string buffer_;
    void flush() {
        std::size_t offset = 0;
        while (offset < buffer_.size()) {
            auto n = ::write(fd_, buffer_.data() + offset, buffer_.size() - offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("output write failed: " + std::string(std::strerror(errno)));
            offset += static_cast<std::size_t>(n);
        }
        buffer_.clear();
    }
};

struct Stamp {
    std::uintmax_t size;
    fs::file_time_type modified;
    std::uint64_t inode;
    std::uint64_t device;
};
Stamp stamp(const fs::path& path) {
    struct stat info{};
    if (::lstat(path.c_str(), &info) != 0) throw InputError("cannot stat staged input");
    return {fs::file_size(path), fs::last_write_time(path), static_cast<std::uint64_t>(info.st_ino), static_cast<std::uint64_t>(info.st_dev)};
}

struct Prepared {
    fs::path graph, coordinates, roads, raw_list, provenance_path;
    std::vector<fs::path> raw;
    Json provenance;
    std::map<fs::path, Stamp> stamps;
    bool declared_complete = false;
};

Prepared prepare(const Boundaries& boundaries, const Options& options) {
    Prepared input;
    input.graph = boundaries.input("graph.txt");
    input.coordinates = boundaries.input("coordinates.txt");
    input.roads = boundaries.input("roads.tsv");
    input.raw_list = boundaries.input("raw_files.txt");
    input.provenance_path = boundaries.input("provenance.json");
    for (const auto& path : {input.graph, input.coordinates, input.roads, input.raw_list, input.provenance_path}) input.stamps.emplace(path, stamp(path));
    input.provenance = read_json(input.provenance_path);
    const auto& p = input.provenance;
    if (p.at("schema_version").unsigned_integer() != 1 || p.at("city").string() != options.city || p.at("observation_date").string() != options.date || p.at("input_kind").string() != "raw_trajectory") throw InputError("provenance schema/city/date/input_kind mismatch");
    const auto& semantics = p.at("reference_semantics");
    if (semantics.at("policy").string() != "legacy_exact_v1") throw InputError("reference policy must be legacy_exact_v1");
    for (const auto& file : semantics.at("files").array()) {
        if (file.at("file").string().empty()) throw InputError("invalid reference source registration");
    }
    std::ifstream list(input.raw_list);
    if (!list) throw InputError("cannot read raw_files.txt");
    std::string line; std::set<fs::path> raw_seen;
    while (std::getline(list, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) throw InputError("raw_files.txt has an empty record");
        const auto path = boundaries.input(line);
        if (!raw_seen.insert(path).second) throw InputError("duplicate raw path");
        input.raw.push_back(path);
    }
    if (list.bad()) throw InputError("raw_files.txt read failed");
    std::set<fs::path> required{input.graph, input.coordinates, input.roads, input.raw_list};
    for (const auto& raw : input.raw) if (!required.insert(raw).second) throw InputError("raw file aliases a static input");
    if (raw_seen.count(input.provenance_path)) throw InputError("provenance cannot be raw input");
    const auto& completeness = p.at("completeness");
    input.declared_complete = completeness.at("declared_complete").boolean();
    if (completeness.at("expected_raw_files").unsigned_integer() != input.raw.size()) throw InputError("declared raw file count differs from ordered list");
    if (input.declared_complete && input.raw.empty()) throw InputError("a complete day cannot have an empty raw file set");
    std::set<fs::path> registered;
    struct Registered { fs::path path; std::uint64_t expected_bytes; };
    std::vector<Registered> records;
    for (const auto& source : p.at("sources").array()) {
        auto path = boundaries.input(source.at("path").string());
        if (!required.count(path)) throw InputError("provenance registers an unexpected runtime input");
        if (!registered.insert(path).second) throw InputError("duplicate provenance source path");
        records.push_back({path, source.at("bytes").unsigned_integer()});
    }
    if (registered != required) throw InputError("provenance does not cover the exact runtime input set");
    // All path decisions precede raw data reads and output creation.
    for (const auto& record : records) {
        input.stamps.emplace(record.path, stamp(record.path));
        if (input.stamps.at(record.path).size != record.expected_bytes) throw InputError("provenance byte mismatch: " + boundaries.relative(record.path));
    }
    input.stamps.emplace(input.provenance_path, stamp(input.provenance_path));
    return input;
}

template<std::size_t N> Array array_json(const std::array<std::uint64_t, N>& values) {
    Array out; out.reserve(N); for (auto n : values) out.emplace_back(n); return out;
}
Json statistics_json(const Graph& graph, const AggregationResult& result) {
    Object counters; for (const auto& pair : result.stats.counters) counters.emplace(pair.first, pair.second);
    return Object{{"schema_version", 1}, {"semantics_policy", "legacy_exact_v1"}, {"aggregation_algorithm", kAggregationPolicy},
        {"base_node_count", static_cast<std::uint64_t>(graph.node_count)}, {"base_arc_count", static_cast<std::uint64_t>(graph.edges.size())},
        {"registered_roads", static_cast<std::uint64_t>(graph.roads.size())}, {"multiple_roads_same_arc", graph.multiple_roads_same_arc},
        {"counters", counters}, {"slot_observations", array_json(result.stats.slot_observations)},
        {"slot_timestamp_samples", array_json(result.stats.slot_timestamp_samples)}, {"slot_updates", array_json(result.stats.slot_updates)},
        {"min_selected_epoch", result.stats.min_selected_epoch}, {"max_selected_epoch", result.stats.max_selected_epoch}};
}
Json fifo_json(const Graph& graph, const AggregationResult& result) {
    std::uint64_t continuous = 0, integer = 0, max_drop = 0;
    Array samples;
    for (std::size_t index = 0; index < graph.edges.size(); ++index) {
        for (std::size_t slot = 1; slot < kSlotCount; ++slot) {
            auto old = snapshot_weight(graph, result, index, slot-1), next = snapshot_weight(graph, result, index, slot);
            if (old <= next) continue;
            const auto drop = static_cast<std::uint64_t>(old) - next;
            ++continuous; if (drop > 1) ++integer; max_drop = std::max(max_drop, drop);
            if (samples.size() < 20) samples.emplace_back(Object{{"src", graph.edges[index].src}, {"dst", graph.edges[index].dst},
                {"from_slot", static_cast<std::uint64_t>(slot-1)}, {"to_slot", static_cast<std::uint64_t>(slot)},
                {"weight_drop_seconds", drop}, {"integer_seconds_violation", drop > 1}});
        }
    }
    return Object{{"schema_version", 1}, {"model", "piecewise_constant_edge_travel_time_seconds"},
        {"boundary_scope", "95 within-day boundaries; next-day rollover is not evaluated"},
        {"boundaries_checked", static_cast<std::uint64_t>(graph.edges.size()) * (kSlotCount - 1)},
        {"continuous_time", Object{{"violation_rule", "w_next < w_previous"}, {"violations", continuous}, {"passed", continuous == 0}}},
        {"integer_seconds", Object{{"violation_rule", "w_next + 1 < w_previous"}, {"violations", integer}, {"passed", integer == 0}}},
        {"maximum_weight_drop_seconds", max_drop}, {"weights_modified", false}, {"violation_examples", samples}};
}

void write_outputs(const Boundaries& boundaries, const Graph& graph, const AggregationResult& result) {
    std::vector<std::size_t> order(graph.edges.size()); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return std::tie(graph.edges[left].src, graph.edges[left].dst) < std::tie(graph.edges[right].src, graph.edges[right].dst);
    });
    Output snapshot(boundaries.output() / "snapshot.txt");
    snapshot.append(std::to_string(graph.node_count) + " " + std::to_string(graph.edges.size()) + "\n");
    for (auto index : order) {
        std::ostringstream row; row << graph.edges[index].src << ' ' << graph.edges[index].dst;
        for (std::size_t slot = 0; slot < kSlotCount; ++slot) row << ' ' << snapshot_weight(graph, result, index, slot);
        row << '\n'; snapshot.append(row.str());
    }
    snapshot.finish();
    Output updates(boundaries.output() / "batchUpdates.txt"); updates.append("96 900\n");
    for (std::size_t slot = 0; slot < kSlotCount; ++slot) {
        updates.append(std::to_string(result.updates[slot].size()));
        for (auto index : order) {
            auto it = result.updates[slot].find(index); if (it == result.updates[slot].end()) continue;
            updates.append(" " + std::to_string(graph.edges[index].src) + " " + std::to_string(graph.edges[index].dst) + " " + std::to_string(it->second));
        }
        updates.append("\n");
    }
    updates.finish();
    write_text_exclusive(boundaries.output() / "stats.json", statistics_json(graph, result).dump() + "\n");
    write_text_exclusive(boundaries.output() / "fifo_report.json", fifo_json(graph, result).dump() + "\n");
}

std::uint64_t counter(const Statistics& stats, const std::string& name) {
    auto it = stats.counters.find(name); return it == stats.counters.end() ? 0 : it->second;
}
} // namespace

int build_snapshot(const Options& options) {
    if (options.city != "BJ" && options.city != "XA") throw InputError("city must be BJ or XA");
    if (options.date != "20160101" && options.date != "20160102") throw InputError("date must be 20160101 or 20160102");
    auto day = day_window(options.date);
    Boundaries boundaries(options);
    auto input = prepare(boundaries, options);
    std::cerr << "Validated staged input metadata; parsing " << input.raw.size() << " ordered raw files.\n";
    Graph graph;
    AggregationResult result;
    try {
        graph = load_city_inputs({input.graph, input.coordinates, input.roads});
        auto raw = accumulate_raw_trajectories(graph, input.raw, day);
        result = finalize_legacy_updates(graph, std::move(raw), day);
    } catch (const std::runtime_error& error) {
        throw InputError(error.what());
    }
    for (const auto& item : input.stamps) {
        boundaries.verify_regular(item.first);
        auto current = stamp(item.first);
        if (current.size != item.second.size || current.modified != item.second.modified || current.inode != item.second.inode || current.device != item.second.device) throw InputError("staged input changed during processing: " + boundaries.relative(item.first));
    }
    const bool observations = counter(result.stats, "selected_day_observations") > 0;
    if (input.declared_complete && !observations) throw InputError("declared complete input contains no observations for the selected city/day");
    boundaries.create_output();
    write_outputs(boundaries, graph, result);
    Array files;
    for (const auto& item : input.stamps) files.emplace_back(Object{{"path", boundaries.relative(item.first)}, {"bytes", static_cast<std::uint64_t>(item.second.size)}});
    Array ordered_raw;
    for (const auto& path : input.raw) ordered_raw.emplace_back(Object{{"path", boundaries.relative(path)}, {"bytes", static_cast<std::uint64_t>(input.stamps.at(path).size)}});
    Object outputs;
    for (const char* name : {"snapshot.txt", "batchUpdates.txt", "stats.json", "fifo_report.json"}) {
        outputs.emplace(name, Object{{"path", name}, {"bytes", static_cast<std::uint64_t>(fs::file_size(boundaries.output() / name))}});
    }
    const auto snapshot_id = options.city + "_" + options.date + "_" + options.run_id;
    Json manifest(Object{{"schema_version", 1}, {"status", "SNAPSHOT_GENERATED"}, {"city", options.city}, {"observation_date", options.date},
        {"snapshot_id", snapshot_id}, {"graph_state", options.date == "20160101" ? "G0" : "G1"}, {"run_id", options.run_id},
        {"time", Object{{"timezone", "UTC+08:00"}, {"start_epoch_inclusive", day.start_epoch}, {"end_epoch_exclusive", day.end_epoch}, {"slot_seconds", kSlotSeconds}, {"slot_count", static_cast<std::uint64_t>(kSlotCount)}, {"weight_unit", "seconds"}, {"slot_assignment", "floor((entry_epoch - day_start_epoch) / 900)"}}},
        {"semantics", Object{{"policy", "legacy_exact_v1"}, {"aggregation_algorithm", kAggregationPolicy}, {"unobserved_slot", "base_edge_weight"}, {"carry_forward", false}, {"new_arcs", false}, {"output_order", "ascending_original_src_then_dst"}, {"raw_file_order", "raw_files.txt explicit order"}, {"reference_semantics", input.provenance.at("reference_semantics")}}},
        {"city_static_inputs", Object{{"graph", boundaries.relative(input.graph)}, {"coordinates", boundaries.relative(input.coordinates)}, {"road_registry", boundaries.relative(input.roads)}}},
        {"inputs", files}, {"ordered_raw_files", ordered_raw}, {"outputs", outputs}, {"input_stability_check", "size, mtime, device, inode unchanged from metadata validation through completed parse"},
        {"completeness", Object{{"declared_complete", input.declared_complete}, {"expected_raw_files", static_cast<std::uint64_t>(input.raw.size())}, {"processed_raw_files", static_cast<std::uint64_t>(input.raw.size())}, {"selected_city_day_has_observations", observations}}},
        {"scope", "daily graph snapshot; downstream OD queries, routes, encoding, and training data are separate stages"},
        {"fifo", fifo_json(graph, result)}, {"statistics", statistics_json(graph, result)}});
    // The success marker is written last, after every output has been flushed.
    write_text_exclusive(boundaries.output() / "manifest.json", manifest.dump() + "\n");
    std::cout << "SNAPSHOT_GENERATED " << boundaries.relative(boundaries.output()) << " " << snapshot_id << '\n';
    return 0;
}
} // namespace sdc::io
