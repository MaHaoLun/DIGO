#include "corpus/sampling.hpp"
#include "corpus/graph.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace corpus {
std::uint64_t SamplingConfig::points_per_group() const {
    if (interval_seconds <= 0 || point_step_seconds <= 0)
        throw std::invalid_argument("interval_seconds and point_step_seconds must be positive");
    if (interval_seconds % point_step_seconds != 0)
        throw std::invalid_argument("interval_seconds must be exactly divisible by point_step_seconds");
    if (start_second < 0 || interval_seconds > 86400 || start_second > 86400 - interval_seconds)
        throw std::invalid_argument("complete window must remain in the selected day: start+interval <= 86400");
    return static_cast<std::uint64_t>(interval_seconds / point_step_seconds);
}

namespace sampling_detail {

std::uint64_t Random::bounded(std::uint64_t exclusive_upper) {
    if (exclusive_upper == 0) throw std::invalid_argument("random bound must be positive");
    const std::uint64_t threshold = (std::uint64_t{0} - exclusive_upper) % exclusive_upper;
    for (;;) {
        const std::uint64_t value = engine_();
        if (value >= threshold) return value % exclusive_upper;
    }
}

void Random::shuffle(std::vector<std::size_t>& values) {
    for (std::size_t count = values.size(); count > 1; --count) {
        std::swap(values[count - 1], values[static_cast<std::size_t>(bounded(count))]);
    }
}

void validate(const Graph& graph, const SamplingConfig& config) {
    if (config.method != "hierarchical_grid" && config.method != "uniform_grid" &&
        config.method != "cluster") throw std::invalid_argument("unsupported sampler: " + config.method);
    if (config.cluster_count <= 0 || config.candidates_per_grid <= 0)
        throw std::invalid_argument("cluster_count and candidates_per_grid must be positive");
    if (config.fine_grid_dim != 2)
        throw std::invalid_argument("this registry requires fine_grid_dim=2 (32 coarse, 64 fine axes)");
    if (config.min_distance < 0 || config.max_distance < config.min_distance || config.max_distance > 62)
        throw std::invalid_argument("Manhattan bounds must satisfy 0 <= min <= max <= 62");
    const auto points = config.points_per_group();
    std::uint64_t total = 0;
    for (const auto rows : config.split_rows) {
        if (rows % points != 0) throw std::invalid_argument("every split row quota must be divisible by points_per_group");
        if (rows > std::numeric_limits<std::uint64_t>::max() - total)
            throw std::invalid_argument("split row total overflows uint64");
        total += rows;
    }
    const auto& b = graph.bbox;
    if (b.longitude_min < -180000000 || b.longitude_max > 180000000 ||
        b.latitude_min < -90000000 || b.latitude_max > 90000000 ||
        b.longitude_min >= b.longitude_max || b.latitude_min >= b.latitude_max)
        throw std::invalid_argument("invalid nondegenerate microdegree bbox");
    if (graph.nodes.empty()) throw std::invalid_argument("cannot sample an empty graph");
    if (graph.component.size() != graph.nodes.size())
        throw std::invalid_argument("complete graph component registry required before sampling");
    std::vector<std::int64_t> ids;
    ids.reserve(graph.nodes.size());
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& n = graph.nodes[i];
        if (n.x < b.longitude_min || n.x > b.longitude_max ||
            n.y < b.latitude_min || n.y > b.latitude_max || graph.component[i] < 0)
            throw std::invalid_argument("invalid coordinate or component in sampling registry");
        const auto grid_axis = [](std::int64_t value, std::int64_t low, std::int64_t high, int dim) {
            return static_cast<int>(std::min<std::int64_t>(dim - 1, (value - low) * dim / (high - low)));
        };
        const int cx = grid_axis(n.x, b.longitude_min, b.longitude_max, 32);
        const int cy = grid_axis(n.y, b.latitude_min, b.latitude_max, 32);
        const int fx = grid_axis(n.x, b.longitude_min, b.longitude_max, 64);
        const int fy = grid_axis(n.y, b.latitude_min, b.latitude_max, 64);
        if (n.coarse_cell != cy * 32 + cx || n.fine_cell != fy * 64 + fx)
            throw std::invalid_argument("sampling registry disagrees with fixed row-major bbox grid");
        ids.push_back(n.id);
    }
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
        throw std::invalid_argument("duplicate original node ID in sampling registry");
}

std::vector<std::vector<std::size_t>> coarse_buckets(const Graph& graph) {
    std::vector<std::vector<std::size_t>> buckets(32 * 32);
    for (std::size_t i = 0; i < graph.nodes.size(); ++i)
        buckets[static_cast<std::size_t>(graph.nodes[i].coarse_cell)].push_back(i);
    for (auto& bucket : buckets)
        std::sort(bucket.begin(), bucket.end(), [&](auto a, auto b) { return graph.nodes[a].id < graph.nodes[b].id; });
    return buckets;
}
} // namespace sampling_detail

QueryPlan make_query_plan(const Graph& graph, const SamplingConfig& config,
                         const std::string& city, const std::string& snapshot_id) {
    if (city.empty() || snapshot_id.empty())
        throw std::invalid_argument("query identity requires nonempty city and snapshot_id");
    const auto points = config.points_per_group();
    std::vector<std::size_t> targets;
    if (config.method == "hierarchical_grid") targets = hierarchical_representatives(graph, config);
    else if (config.method == "uniform_grid") targets = uniform_representatives(graph, config);
    else if (config.method == "cluster") targets = cluster_representatives(graph, config);
    else throw std::invalid_argument("unsupported sampler: " + config.method);

    QueryPlan result;
    result.representative_count = targets.size();
    std::vector<std::size_t> sources;
    if (config.method == "uniform_grid") {
        std::array<bool, 1024> seen{};
        for (auto index : targets) {
            const auto cell = static_cast<std::size_t>(graph.nodes[index].coarse_cell);
            if (!seen[cell]) { sources.push_back(index); seen[cell] = true; }
        }
    } else sources = targets;
    const auto by_id = [&](auto a, auto b) { return graph.nodes[a].id < graph.nodes[b].id; };
    std::sort(sources.begin(), sources.end(), by_id);
    std::sort(targets.begin(), targets.end(), by_id);
    if (std::adjacent_find(targets.begin(), targets.end()) != targets.end())
        throw std::logic_error("sampler returned duplicate representatives");

    struct Group { std::size_t source, destination; std::uint64_t interval; };
    std::vector<Group> groups;
    for (const auto source : sources) {
        const int sc = graph.nodes[source].coarse_cell;
        for (const auto destination : targets) {
            if (source == destination) continue;
            const int dc = graph.nodes[destination].coarse_cell;
            if (sc == dc) continue; // All three strategies pair different coarse cells.
            const int distance = std::abs(sc % 32 - dc % 32) + std::abs(sc / 32 - dc / 32);
            if (distance < config.min_distance || distance > config.max_distance) continue;
            if (groups.size() > (std::numeric_limits<std::uint64_t>::max() - (points - 1)) / points)
                throw std::overflow_error("candidate rank cannot fit point query IDs");
            groups.push_back({source, destination, static_cast<std::uint64_t>(groups.size())});
        }
    }
    result.candidate_groups = groups.size();
    groups.erase(std::remove_if(groups.begin(), groups.end(), [&](const Group& group) {
        return graph.component[group.source] != graph.component[group.destination];
    }), groups.end());
    result.reachable_groups = groups.size();
    std::uint64_t requested_rows = 0;
    for (const auto rows : config.split_rows) requested_rows += rows; // Checked by sampler validation.
    if (requested_rows == 0) throw std::invalid_argument("query plan requires a positive total row quota");
    const std::uint64_t requested_groups = requested_rows / points;
    if (requested_groups > result.reachable_groups) {
        throw std::runtime_error("insufficient whole-group capacity: method=" + config.method +
            " representatives=" + std::to_string(result.representative_count) +
            " candidate_groups=" + std::to_string(result.candidate_groups) +
            " reachable_groups=" + std::to_string(result.reachable_groups) +
            " required_groups=" + std::to_string(requested_groups) +
            "; no repeats, random fallback or quota reduction permitted");
    }
    std::vector<std::size_t> order(groups.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    sampling_detail::Random random(config.seed);
    random.shuffle(order);
    if (requested_rows > result.queries.max_size()) throw std::length_error("query plan row count exceeds vector capacity");
    result.queries.reserve(static_cast<std::size_t>(requested_rows));
    std::size_t selected = 0;
    for (std::size_t split = 0; split < config.split_rows.size(); ++split) {
        for (std::uint64_t row = 0; row < config.split_rows[split]; row += points) {
            const Group& group = groups[order[selected++]];
            for (std::uint64_t offset = 0; offset < points; ++offset) {
                QueryRecord query;
                query.query_id = group.interval * points + offset;
                query.interval_id = group.interval;
                query.row_index = row + offset;
                query.city = city;
                query.snapshot_id = snapshot_id;
                query.source = graph.nodes[group.source].id;
                query.destination = graph.nodes[group.destination].id;
                query.departure = config.start_second + static_cast<std::int64_t>(offset) * config.point_step_seconds;
                query.split = static_cast<Split>(split);
                result.queries.push_back(std::move(query));
            }
        }
    }
    return result;
}
} // namespace corpus
