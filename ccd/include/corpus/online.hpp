#pragma once
#include "corpus/graph.hpp"
#include "corpus/sampling.hpp"
#include "sdc/io.hpp"
#include <array>
#include <filesystem>
#include <map>
#include <set>
#include <utility>

namespace corpus {
using OdPair = std::pair<std::int64_t, std::int64_t>;
using G0Membership = std::map<OdPair, Split>;
G0Membership read_g0_membership(const std::filesystem::path&);
struct ShiftPair { double absolute_change = 0, relative_change = 0; };
struct ShiftDetection {
    std::map<OdPair, ShiftPair> shifted_pairs;
    std::array<double, 1024> cell_scores{};
    std::set<int> hotspots;
    std::uint64_t canonical_pairs = 0, effective_changed_anchor_pairs = 0;
    std::uint64_t effective_changed_anchors = 0, monitoring_changed_pairs = 0;
    sdc::io::Json report() const;
};
ShiftDetection detect_shifts(const Graph& old_graph, const Graph& new_graph);
struct PointEvidence {
    bool shifted_pair_hit = false, changed_edge_at_entry = false;
    bool internal_hotspot_hit = false, endpoint_only_hit = false;
    bool has_old_route = false, old_new_path_changed = false, old_new_duration_changed = false;
    bool eligible() const { return changed_edge_at_entry || internal_hotspot_hit || old_new_path_changed; }
    sdc::io::Json json() const;
};
PointEvidence analyze_point(const Graph& old_graph, const Graph& new_graph,
                           const ShiftDetection&, const RouteRecord& new_route,
                           const RouteRecord* old_route = nullptr);
struct OnlineConfig {
    SamplingConfig sampling;
    std::array<std::uint64_t, 2> split_rows{80000, 20000};
    std::size_t qualification_chunk_groups = 4096;
    OnlineConfig();
};
struct OnlinePlan {
    std::vector<QueryRecord> queries;
    // Parallel to queries; 0=ordinary, 1=targeted. Evidence is populated for
    // qualified targets only. Final emission recomputes every point's flags.
    std::vector<std::uint8_t> targeted;
    std::vector<PointEvidence> qualification_evidence;
    std::uint64_t representative_count = 0, candidate_groups = 0, reachable_groups = 0;
    std::array<std::uint64_t, 2> legal_groups{}, ordinary_groups{}, targeted_groups{};
    std::array<std::uint64_t, 2> screened_groups{}, eligible_screened_groups{};
    std::uint64_t qualification_old_sssp = 0, qualification_new_sssp = 0;
    std::uint64_t new_od_groups = 0, new_od_adaptation_groups = 0;
    sdc::io::Json capacity_report() const;
};
OnlinePlan make_online_plan(const Graph& old_graph, const Graph& new_graph,
                           const OnlineConfig&, const G0Membership&,
                           const std::string& city, const std::string& new_model_id,
                           const std::string& old_model_id, const ShiftDetection&);
} // namespace corpus
