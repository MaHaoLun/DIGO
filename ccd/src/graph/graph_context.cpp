#include "corpus/graph.hpp"
#include "corpus/lpf.hpp"
#include <map>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace corpus {
namespace {

class Rows {
public:
    explicit Rows(const std::filesystem::path& path) : input_(path), path_(path.string()) {
        if (!input_) fail("cannot open input");
    }

    std::vector<std::string_view> require(std::size_t fields) {
        auto row = next();
        if (row.size() != fields) fail("wrong field count or premature end of input");
        return row;
    }

    template <typename T> T integer(std::string_view text) const {
        T result{};
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
            fail("invalid or overflowing integer");
        return result;
    }

    void finish() {
        if (!next().empty()) fail("extra rows after declared count");
    }

    [[noreturn]] void fail(const std::string& reason) const {
        throw std::runtime_error(path_ + ":" + std::to_string(line_number_) + ": " + reason);
    }

private:
    std::vector<std::string_view> next() {
        while (std::getline(input_, line_)) {
            ++line_number_;
            std::vector<std::string_view> fields;
            fields.reserve(98);
            std::size_t position = 0;
            while (position < line_.size()) {
                while (position < line_.size() &&
                       std::isspace(static_cast<unsigned char>(line_[position]))) ++position;
                const auto start = position;
                while (position < line_.size() &&
                       !std::isspace(static_cast<unsigned char>(line_[position]))) ++position;
                if (position > start) fields.emplace_back(line_.data() + start, position - start);
            }
            if (!fields.empty()) return fields;
        }
        if (input_.bad()) fail("input read failure");
        return {};
    }

    std::ifstream input_;
    std::string path_, line_;
    std::uint64_t line_number_ = 0;
};

void validate_bbox(const BBox& box) {
    if (box.longitude_min < -180000000 || box.longitude_max > 180000000 ||
        box.latitude_min < -90000000 || box.latitude_max > 90000000 ||
        box.longitude_min >= box.longitude_max || box.latitude_min >= box.latitude_max)
        throw std::invalid_argument("bbox must have positive geographic microdegree extents");
}

int grid_axis(std::int64_t value, std::int64_t low, std::int64_t high, int dimension) {
    // All inputs are inside geographic bounds: the integer product fits int64.
    // Division implements floor because both operands are nonnegative.
    return static_cast<int>(std::min<std::int64_t>(dimension - 1,
        ((value - low) * dimension) / (high - low)));
}

float condition(std::int64_t value, std::int64_t low, std::int64_t high) {
    return static_cast<float>(2.0 * static_cast<double>(value - low) /
                              static_cast<double>(high - low) - 1.0);
}

struct PairHash {
    std::size_t operator()(const std::pair<std::size_t, std::size_t>& pair) const noexcept {
        const auto left = std::hash<std::size_t>{}(pair.first);
        const auto right = std::hash<std::size_t>{}(pair.second);
        return left ^ (right + static_cast<std::size_t>(0x9e3779b9U) + (left << 6U) + (left >> 2U));
    }
};

} // namespace

std::int64_t Graph::travel_time(const Edge& edge, std::int64_t arrival) const {
    if(arrival<0) throw std::invalid_argument("negative graph traversal time");
    return policy==RoutingPolicy::CanonicalFifoLpf ? lpf::evaluate(edge.weights,arrival) : edge.weights[(arrival/900)%96];
}

std::size_t Graph::index(std::int64_t original_id) const {
    const auto found = indices_.find(original_id);
    if (found == indices_.end())
        throw std::out_of_range("unknown original node ID " + std::to_string(original_id));
    return found->second;
}

Graph Graph::load(const std::filesystem::path& snapshot,
                  const std::filesystem::path& coordinates, const BBox& box, RoutingPolicy policy) {
    validate_bbox(box);
    Graph graph;
    graph.bbox = box;
    graph.policy = policy;
    Rows coordinate_rows(coordinates);
    const auto coordinate_header = coordinate_rows.require(1);
    const auto node_count = coordinate_rows.integer<std::size_t>(coordinate_header[0]);
    if (node_count == 0 || node_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        coordinate_rows.fail("node count must be positive and fit component indices");
    graph.nodes.reserve(node_count);
    for (std::size_t i = 0; i < node_count; ++i) {
        const auto row = coordinate_rows.require(3);
        Node node;
        node.id = coordinate_rows.integer<std::int64_t>(row[0]);
        node.x = coordinate_rows.integer<std::int64_t>(row[1]);
        node.y = coordinate_rows.integer<std::int64_t>(row[2]);
        if (node.x < box.longitude_min || node.x > box.longitude_max ||
            node.y < box.latitude_min || node.y > box.latitude_max)
            coordinate_rows.fail("coordinate is outside the fixed bbox");
        node.coarse_cell = grid_axis(node.y, box.latitude_min, box.latitude_max, 32) * 32 +
                           grid_axis(node.x, box.longitude_min, box.longitude_max, 32);
        node.fine_cell = grid_axis(node.y, box.latitude_min, box.latitude_max, 64) * 64 +
                         grid_axis(node.x, box.longitude_min, box.longitude_max, 64);
        node.longitude_condition = condition(node.x, box.longitude_min, box.longitude_max);
        node.latitude_condition = condition(node.y, box.latitude_min, box.latitude_max);
        graph.nodes.push_back(node);
    }
    coordinate_rows.finish();
    std::sort(graph.nodes.begin(), graph.nodes.end(), [](const Node& a, const Node& b) {
        return a.id < b.id;
    });
    graph.indices_.reserve(node_count);
    for (std::size_t i = 0; i < node_count; ++i) {
        if (!graph.indices_.emplace(graph.nodes[i].id, i).second)
            coordinate_rows.fail("duplicate coordinate node ID");
    }
    graph.adjacency.resize(node_count);

    Rows snapshot_rows(snapshot);
    const auto snapshot_header = snapshot_rows.require(2);
    if (snapshot_rows.integer<std::size_t>(snapshot_header[0]) != node_count)
        snapshot_rows.fail("snapshot node count disagrees with complete coordinate registry");
    graph.input_arc_count = snapshot_rows.integer<std::uint64_t>(snapshot_header[1]);
    // Record positions, so replacing a duplicate/reverse row updates both
    // registered directions without adding duplicate adjacency entries.
    using Pair = std::pair<std::size_t, std::size_t>;
    std::unordered_map<Pair, Pair, PairHash> positions;
    std::map<Pair, lpf::Knots> ordered;
    for (std::uint64_t i = 0; i < graph.input_arc_count; ++i) {
        const auto row = snapshot_rows.require(98);
        const auto source_id = snapshot_rows.integer<std::int64_t>(row[0]);
        const auto destination_id = snapshot_rows.integer<std::int64_t>(row[1]);
        const auto source_found = graph.indices_.find(source_id);
        const auto destination_found = graph.indices_.find(destination_id);
        if (source_found == graph.indices_.end() || destination_found == graph.indices_.end())
            snapshot_rows.fail("snapshot endpoint missing from coordinate registry");
        const auto pair = std::minmax(source_found->second, destination_found->second);
        const Pair key{pair.first, pair.second};
        std::array<std::int64_t, 96> weights{};
        for (std::size_t slot = 0; slot < weights.size(); ++slot) {
            weights[slot] = snapshot_rows.integer<std::int64_t>(row[slot + 2]);
            if (weights[slot] <= 0) snapshot_rows.fail("edge weights must be positive integer seconds");
        }
        if (std::any_of(weights.begin() + 1, weights.end(), [&](std::int64_t weight) {
                return weight != weights.front();
            })) ++graph.input_time_varying_arc_count;
        if(policy == RoutingPolicy::CanonicalFifoLpf) {
            const Pair direction{source_found->second,destination_found->second};
            auto [it, inserted]=ordered.emplace(direction,weights);
            if(!inserted) {
                if(it->second!=weights) snapshot_rows.fail("conflicting duplicate ordered edge");
                ++graph.audit.duplicate_rows;
            }
            if(source_id>destination_id) ++graph.audit.reverse_rows_ignored;
            continue;
        }
        const auto found = positions.find(key);
        if (found == positions.end()) {
            const Pair offsets{graph.adjacency[key.first].size(), graph.adjacency[key.second].size()};
            graph.adjacency[key.first].push_back({key.second, weights});
            if (key.first != key.second) graph.adjacency[key.second].push_back({key.first, weights});
            positions.emplace(key, offsets);
        } else {
            graph.adjacency[key.first][found->second.first].weights = weights;
            if (key.first != key.second)
                graph.adjacency[key.second][found->second.second].weights = weights;
        }
    }
    snapshot_rows.finish();
    graph.audit.source_varying_rows=graph.input_time_varying_arc_count;
    if(policy == RoutingPolicy::CanonicalFifoLpf) {
        for(const auto& [pair, weights]:ordered) {
            if(pair.first>pair.second) {
                if(!ordered.count({pair.second,pair.first}))
                    throw std::runtime_error("missing canonical ordered edge");
                continue;
            }
            ++graph.audit.canonical_pair_count;
            auto varying=[](const lpf::Knots& w){return std::any_of(w.begin()+1,w.end(),[&](auto x){return x!=w[0];});};
            graph.audit.canonical_varying_pairs_before+=varying(weights);
            graph.audit.raw_lpf_violations+=lpf::violations(weights);
            auto repaired=lpf::least_fifo_majorant(weights);
            graph.audit.canonical_varying_pairs_after+=varying(repaired);
            const auto residual=lpf::violations(repaired);
            graph.audit.repaired_continuous_violations+=residual;
            graph.audit.repaired_integer_violations+=residual;
            if(residual) throw std::logic_error("LPF repair failed FIFO");
            graph.audit.changed_pairs+=(repaired!=weights);
            for(std::size_t i=0;i<96;++i) {
                const auto raise=static_cast<std::uint64_t>(repaired[i]-weights[i]);
                graph.audit.changed_knots+=(raise!=0);
                if(raise>std::numeric_limits<std::uint64_t>::max()-graph.audit.sum_raise)
                    throw std::overflow_error("LPF aggregate raise overflow");
                graph.audit.sum_raise+=raise;
                graph.audit.max_raise=std::max(graph.audit.max_raise,raise);
            }
            graph.adjacency[pair.first].push_back({pair.second,repaired});
            if(pair.first!=pair.second) graph.adjacency[pair.second].push_back({pair.first,repaired});
        }
    }
    for (auto& neighbors : graph.adjacency) {
        std::sort(neighbors.begin(), neighbors.end(), [&graph](const Edge& a, const Edge& b) {
            return graph.nodes[a.to].id < graph.nodes[b.to].id;
        });
        for (const auto& edge : neighbors) {
            for (std::size_t slot = 0; slot < edge.weights.size(); ++slot) {
                const auto next = edge.weights[(slot + 1) % edge.weights.size()];
                // A(b-1) > A(b) iff w_before > w_after + 1. Subtract only
                // positive ordered weights to avoid overflowing next + 1.
                if (edge.weights[slot] > next && edge.weights[slot] - next >
                    (policy == RoutingPolicy::LegacySlots ? 1 : 900))
                    ++graph.fifo_integer_violations;
            }
        }
    }

    graph.component.assign(node_count, -1);
    std::vector<std::size_t> queue;
    int next_component = 0;
    for (std::size_t root = 0; root < node_count; ++root) {
        if (graph.component[root] != -1) continue;
        queue.clear();
        queue.push_back(root);
        graph.component[root] = next_component;
        for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
            for (const auto& edge : graph.adjacency[queue[cursor]]) {
                if (graph.component[edge.to] != -1) continue;
                graph.component[edge.to] = next_component;
                queue.push_back(edge.to);
            }
        }
        ++next_component;
    }
    return graph;
}

} // namespace corpus
