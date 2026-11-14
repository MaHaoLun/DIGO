#include "corpus/sampling.hpp"
#include "corpus/graph.hpp"

#include <stdexcept>

namespace corpus {
namespace {
using Integer = __int128_t;
using Natural = __uint128_t;
constexpr Natural maximum = ~Natural{0};

struct Centroid {
    Integer sum_x = 0, sum_y = 0;
    std::uint64_t count = 1;
};
struct Distance { Natural numerator, denominator; };

Natural square(Integer value) {
    const Natural magnitude = value < 0 ? Natural(-value) : Natural(value);
    if (magnitude != 0 && magnitude > maximum / magnitude)
        throw std::overflow_error("cluster exact squared distance exceeds 128-bit capacity");
    return magnitude * magnitude;
}

Distance distance_to(const Node& node, const BBox& box, const Centroid& center) {
    const Integer dx = Integer(node.x - box.longitude_min) * center.count - center.sum_x;
    const Integer dy = Integer(node.y - box.latitude_min) * center.count - center.sum_y;
    const Natural x2 = square(dx), y2 = square(dy);
    if (x2 > maximum - y2) throw std::overflow_error("cluster exact distance sum exceeds 128-bit capacity");
    return {x2 + y2, Natural(center.count) * center.count};
}

bool less(Distance a, Distance b) {
    // Compare positive rational numbers by Euclidean division. Cross products
    // could overflow even though both exact squared distances fit in 128 bits.
    bool reversed = false;
    for (;;) {
        const Natural aq = a.numerator / a.denominator, bq = b.numerator / b.denominator;
        if (aq != bq) return reversed ? aq > bq : aq < bq;
        const Natural ar = a.numerator % a.denominator, br = b.numerator % b.denominator;
        if (ar == 0 || br == 0) {
            if (ar == br) return false;
            return reversed ? ar != 0 : ar == 0;
        }
        a = {a.denominator, ar};
        b = {b.denominator, br};
        reversed = !reversed;
    }
}
} // namespace

std::vector<std::size_t> cluster_representatives(const Graph& graph, const SamplingConfig& config) {
    sampling_detail::validate(graph, config);
    const auto cells = sampling_detail::coarse_buckets(graph);
    sampling_detail::Random random(config.seed);
    std::vector<std::vector<std::size_t>> largest_clusters(cells.size());
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
        const auto& nodes = cells[cell];
        if (nodes.empty()) continue;
        const auto draw_center = [&]() {
            const auto& node = graph.nodes[nodes[static_cast<std::size_t>(random.bounded(nodes.size()))]];
            return Centroid{node.x - graph.bbox.longitude_min, node.y - graph.bbox.latitude_min, 1};
        };
        std::vector<Centroid> centers(static_cast<std::size_t>(config.cluster_count));
        for (auto& center : centers) center = draw_center(); // Sampling with replacement matches old New.
        std::vector<std::vector<std::size_t>> clusters(centers.size());
        for (int iteration = 0; iteration < 5; ++iteration) {
            for (auto& cluster : clusters) cluster.clear();
            for (const auto index : nodes) {
                std::size_t closest = 0;
                Distance best = distance_to(graph.nodes[index], graph.bbox, centers.front());
                for (std::size_t c = 1; c < centers.size(); ++c) {
                    const Distance score = distance_to(graph.nodes[index], graph.bbox, centers[c]);
                    if (less(score, best)) { closest = c; best = score; }
                }
                clusters[closest].push_back(index); // Equal scores retain the first cluster.
            }
            for (std::size_t c = 0; c < centers.size(); ++c) {
                if (clusters[c].empty()) { centers[c] = draw_center(); continue; }
                Centroid center{0, 0, static_cast<std::uint64_t>(clusters[c].size())};
                for (const auto index : clusters[c]) {
                    center.sum_x += graph.nodes[index].x - graph.bbox.longitude_min;
                    center.sum_y += graph.nodes[index].y - graph.bbox.latitude_min;
                }
                centers[c] = center;
            }
        }
        std::size_t largest = 0;
        for (std::size_t c = 1; c < clusters.size(); ++c)
            if (clusters[c].size() > clusters[largest].size()) largest = c;
        largest_clusters[cell] = std::move(clusters[largest]);
    }
    std::vector<std::size_t> representatives;
    for (const auto& cluster : largest_clusters) {
        if (!cluster.empty())
            representatives.push_back(cluster[static_cast<std::size_t>(random.bounded(cluster.size()))]);
    }
    return representatives;
}
} // namespace corpus
