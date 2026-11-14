#include "sdc/snapshot.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace sdc {
namespace {

struct PairHash {
    std::size_t operator()(const std::pair<NodeId, NodeId>& value) const noexcept {
        const auto a = std::hash<NodeId>{}(value.first);
        const auto b = std::hash<NodeId>{}(value.second);
        return a ^ (b + 0x9e3779b9U + (a << 6U) + (a >> 2U));
    }
};

class InputRows {
public:
    explicit InputRows(const std::filesystem::path& path) : input_(path), path_(path.string()) {
        if (!input_) fail("cannot open input");
    }
    std::vector<std::string> next() {
        std::string line;
        while (std::getline(input_, line)) {
            ++line_;
            std::istringstream row(line);
            std::vector<std::string> tokens;
            for (std::string token; row >> token;) tokens.push_back(std::move(token));
            if (!tokens.empty()) return tokens;
        }
        if (input_.bad()) fail("input read failure");
        return {};
    }
    std::vector<std::string> require(std::size_t count) {
        auto tokens = next();
        if (tokens.size() != count) fail("wrong field count or premature end of input");
        return tokens;
    }
    template <typename T> T integer(const std::string& token) const {
        T number{};
        const auto result = std::from_chars(token.data(), token.data() + token.size(), number);
        if (result.ec != std::errc{} || result.ptr != token.data() + token.size())
            fail("invalid or overflowing integer");
        return number;
    }
    void finish() {
        if (!next().empty()) fail("extra rows after declared count");
    }
    [[noreturn]] void fail(const std::string& reason) const {
        throw std::runtime_error(path_ + ":" + std::to_string(line_) + ": " + reason);
    }
private:
    std::ifstream input_;
    std::string path_;
    std::uint64_t line_ = 0;
};

Weight positive_weight(InputRows& input, const std::string& token) {
    auto value = input.integer<std::uint64_t>(token);
    if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
        input.fail("weight must be a positive signed-32-bit number of seconds");
    return static_cast<Weight>(value);
}

} // namespace

DayWindow day_window(const std::string& date) {
    if (date == "20160101") return {date, 1451577600, 1451664000};
    if (date == "20160102") return {date, 1451664000, 1451750400};
    throw std::runtime_error("only observation dates 20160101 and 20160102 are supported");
}

Graph load_city_inputs(const CityInputs& inputs) {
    Graph graph;
    InputRows coordinates(inputs.coordinates);
    const auto coordinate_header = coordinates.require(1);
    graph.node_count = coordinates.integer<std::size_t>(coordinate_header[0]);
    if (graph.node_count == 0) coordinates.fail("node count must be positive");
    std::unordered_set<NodeId> nodes;
    for (std::size_t i = 0; i < graph.node_count; ++i) {
        const auto row = coordinates.require(3);
        Coordinate coordinate{coordinates.integer<NodeId>(row[0]),
            coordinates.integer<std::int64_t>(row[1]), coordinates.integer<std::int64_t>(row[2])};
        if (coordinate.longitude_microdegrees < -180000000 || coordinate.longitude_microdegrees > 180000000 ||
            coordinate.latitude_microdegrees < -90000000 || coordinate.latitude_microdegrees > 90000000)
            coordinates.fail("coordinate outside geographic microdegree range");
        if (!nodes.insert(coordinate.node_id).second) coordinates.fail("duplicate coordinate node ID");
        graph.coordinates.push_back(coordinate);
    }
    coordinates.finish();

    InputRows base(inputs.base_graph);
    const auto header = base.require(2);
    if (base.integer<std::size_t>(header[0]) != graph.node_count)
        base.fail("node count disagrees with coordinates");
    const auto edge_count = base.integer<std::size_t>(header[1]);
    std::unordered_map<std::pair<NodeId, NodeId>, std::size_t, PairHash> arcs;
    std::unordered_map<std::pair<NodeId, NodeId>, Weight, PairHash> last_undirected_weight;
    for (std::size_t i = 0; i < edge_count; ++i) {
        const auto row = base.require(3);
        const Edge edge{base.integer<NodeId>(row[0]), base.integer<NodeId>(row[1]), positive_weight(base, row[2])};
        if (nodes.count(edge.src) == 0 || nodes.count(edge.dst) == 0)
            base.fail("edge endpoint missing from coordinates");
        if (!arcs.emplace(std::make_pair(edge.src, edge.dst), i).second)
            base.fail("duplicate directed base edge");
        const std::pair<NodeId, NodeId> canonical = std::minmax(edge.src, edge.dst);
        // The reference overwrote the road's original weight while reading both directions.
        last_undirected_weight[canonical] = edge.base_seconds;
        graph.edges.push_back(edge);
    }
    base.finish();

    InputRows registry(inputs.road_registry);
    const auto road_header = registry.require(1);
    const auto road_count = registry.integer<std::size_t>(road_header[0]);
    std::unordered_set<std::size_t> mapped_arcs;
    for (std::size_t i = 0; i < road_count; ++i) {
        const auto row = registry.next();
        if (row.size() != 4 && row.size() != 5) registry.fail("road row must have four or five fields");
        Road road;
        road.road_id = registry.integer<RoadId>(row[0]);
        road.original_src = registry.integer<NodeId>(row[1]);
        road.original_dst = registry.integer<NodeId>(row[2]);
        road.distance_m = registry.integer<std::uint64_t>(row[3]);
        if (road.road_id == 0 || road.road_id > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            registry.fail("road ID must be a positive signed-64-bit integer");
        if (road.distance_m == 0 || road.distance_m > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
            registry.fail("road distance must be positive and fit signed 32 bits");
        if (!nodes.count(road.original_src) || !nodes.count(road.original_dst))
            registry.fail("road endpoint missing from coordinates");
        const std::pair<NodeId, NodeId> canonical = std::minmax(road.original_src, road.original_dst);
        const auto arc = arcs.find(canonical);
        if (arc == arcs.end()) registry.fail("road has no existing canonical directed base arc");
        road.edge_index = arc->second;
        road.legacy_base_seconds = row.size() == 5 ? positive_weight(registry, row[4]) : last_undirected_weight.at(canonical);
        if (!graph.road_index.emplace(road.road_id, i).second) registry.fail("duplicate road ID");
        if (!mapped_arcs.insert(road.edge_index).second) ++graph.multiple_roads_same_arc;
        graph.roads.push_back(road);
    }
    registry.finish();
    return graph;
}

void increment_counter(Statistics& stats, const std::string& name, std::uint64_t increment) {
    auto& counter = stats.counters[name];
    if (increment > std::numeric_limits<std::uint64_t>::max() - counter)
        throw std::overflow_error("statistics counter overflow: " + name);
    counter += increment;
}

} // namespace sdc
