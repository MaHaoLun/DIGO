#include "corpus/sampling.hpp"
#include "corpus/graph.hpp"

#include <array>
#include <limits>

namespace corpus {
std::vector<std::size_t> hierarchical_representatives(const Graph& graph, const SamplingConfig& config) {
    sampling_detail::validate(graph, config);
    constexpr auto absent = std::numeric_limits<std::size_t>::max();
    std::array<std::size_t, 4096> best;
    best.fill(absent);
    // Both axes use the same denominator 128. Keeping each physical bbox span
    // in the numerator preserves real microdegree distances on unequal axes.
    using Wide = __int128_t;
    const auto score = [&](std::size_t index) {
        const auto& node = graph.nodes[index];
        const auto& box = graph.bbox;
        const int fx = node.fine_cell % 64, fy = node.fine_cell / 64;
        const Wide dx = Wide(node.x - box.longitude_min) * 128 -
                        Wide(2 * fx + 1) * (box.longitude_max - box.longitude_min);
        const Wide dy = Wide(node.y - box.latitude_min) * 128 -
                        Wide(2 * fy + 1) * (box.latitude_max - box.latitude_min);
        return dx * dx + dy * dy;
    };
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        auto& chosen = best[static_cast<std::size_t>(graph.nodes[i].fine_cell)];
        if (chosen == absent || score(i) < score(chosen) ||
            (score(i) == score(chosen) && graph.nodes[i].id < graph.nodes[chosen].id)) chosen = i;
    }
    std::vector<std::size_t> representatives;
    for (const auto index : best) if (index != absent) representatives.push_back(index);
    return representatives;
}
} // namespace corpus
