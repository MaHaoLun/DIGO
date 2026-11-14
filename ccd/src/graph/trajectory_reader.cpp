#include "sdc/snapshot.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace sdc {
namespace {

constexpr std::uint64_t kMaxWeight = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());

template <typename T> T parse_integer(std::string_view token) {
    T result{};
    if (!token.empty() && token.front() == '+') {
        token.remove_prefix(1);
        if (!token.empty() && (token.front() == '+' || token.front() == '-'))
            throw std::runtime_error("invalid raw integer sign");
    }
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), result);
    if (token.empty() || parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size())
        throw std::runtime_error("invalid or overflowing raw integer");
    return result;
}

void parse_csv(const std::string& line, std::vector<std::string_view>& fields,
               std::vector<std::string>& quoted_fields) {
    fields.clear();
    quoted_fields.clear();
    // Verified national shards are unquoted; views avoid copying their long link lists.
    if (line.find('"') == std::string::npos) {
        std::size_t start = 0;
        for (;;) {
            const auto end = line.find(',', start);
            fields.emplace_back(line.data() + start, (end == std::string::npos ? line.size() : end) - start);
            if (end == std::string::npos) break;
            start = end + 1;
        }
        return;
    }
    std::size_t pos = 0;
    for (;;) {
        std::string field;
        if (pos < line.size() && line[pos] == '"') {
            ++pos;
            bool closed = false;
            while (pos < line.size()) {
                const char value = line[pos++];
                if (value != '"') { field.push_back(value); continue; }
                if (pos < line.size() && line[pos] == '"') { field.push_back('"'); ++pos; }
                else { closed = true; break; }
            }
            if (!closed || (pos < line.size() && line[pos] != ','))
                throw std::runtime_error("malformed quoted CSV field");
        } else {
            while (pos < line.size() && line[pos] != ',') {
                if (line[pos] == '"') throw std::runtime_error("quote inside unquoted CSV field");
                field.push_back(line[pos++]);
            }
        }
        quoted_fields.push_back(std::move(field));
        if (pos == line.size()) break;
        ++pos;
    }
    for (const auto& field : quoted_fields) fields.emplace_back(field);
}

void parse_list(std::string_view field, std::vector<std::int64_t>& values) {
    values.clear();
    std::size_t start = 0;
    for (;;) {
        const auto end = field.find('|', start);
        values.push_back(parse_integer<std::int64_t>(field.substr(start,
            end == std::string_view::npos ? field.size() - start : end - start)));
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
}

void increment(std::uint64_t& value) {
    if (value == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("observation count overflow");
    ++value;
}

void accumulate(TimestampMean& mean, Weight seconds) {
    if (mean.count == 0) { mean = {seconds, 1}; return; }
    // Truncate after EACH observation, reproducing the reference's ordered integer mean.
    if (mean.count == std::numeric_limits<std::uint64_t>::max() ||
        mean.count > (std::numeric_limits<std::uint64_t>::max() - seconds) / mean.seconds)
        throw std::overflow_error("legacy timestamp mean arithmetic overflow");
    const auto sum = static_cast<std::uint64_t>(mean.seconds) * mean.count + seconds;
    ++mean.count;
    mean.seconds = static_cast<Weight>(sum / mean.count);
}

} // namespace

RawAggregation accumulate_raw_trajectories(const Graph& graph,
    const std::vector<std::filesystem::path>& ordered_raw_paths, const DayWindow& day) {
    const auto expected = day_window(day.observation_date);
    if (day.start_epoch != expected.start_epoch || day.end_epoch != expected.end_epoch)
        throw std::runtime_error("day window must match the fixed UTC+08:00 observation date");
    if (graph.roads.size() > std::numeric_limits<std::uint64_t>::max() / kDaySeconds)
        throw std::overflow_error("road/timestamp key overflow");
    RawAggregation raw;
    for (const char* name : {"raw_files", "raw_bytes", "lines_seen", "blank_lines", "valid_trajectories",
         "malformed_records", "rejected_reversed_trajectories", "outside_day_records", "legacy_date_excluded_records",
         "road_observations", "outside_day_observations", "outside_city_observations", "selected_day_observations",
         "rejected_negative_duration", "rejected_duration_overflow", "rejected_invalid_road_id", "clamped_observations",
         "zero_duration_observations", "duplicate_timestamp_observations", "timestamp_samples", "roads_observed"})
        raw.stats.counters.emplace(name, 0);
    raw.stats.counters["multiple_roads_same_arc"] = graph.multiple_roads_same_arc;
    std::vector<std::string_view> fields;
    fields.reserve(32);
    std::vector<std::string> quoted_fields;
    std::vector<std::int64_t> roads, times;
    std::vector<char> read_buffer(1024 * 1024);
    std::string line;
    for (const auto& path : ordered_raw_paths) {
        std::ifstream input;
        input.rdbuf()->pubsetbuf(read_buffer.data(), static_cast<std::streamsize>(read_buffer.size()));
        input.open(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open raw shard: " + path.string());
        increment_counter(raw.stats, "raw_files");
        increment_counter(raw.stats, "raw_bytes", std::filesystem::file_size(path));
        std::uint64_t line_number = 0;
        while (std::getline(input, line)) {
            increment(line_number);
            increment_counter(raw.stats, "lines_seen");
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) { increment_counter(raw.stats, "blank_lines"); continue; }
            try {
                parse_csv(line, fields, quoted_fields);
                if (fields.size() < 28) throw std::runtime_error("raw CSV requires at least 28 fields");
                const auto start = parse_integer<std::int64_t>(fields[26]);
                const auto end = parse_integer<std::int64_t>(fields[27]);
                const auto count = parse_integer<std::size_t>(fields[11]);
                parse_list(fields[4], roads);
                parse_list(fields[8], times);
                if (count == 0 || count != roads.size() || count != times.size())
                    throw std::runtime_error("road, entry timestamp and declared link counts disagree");
                // Column 7 is the trajectory-wide city list, never a per-road label.
                // Raw GPS columns 22-25 are not consumed by this snapshot stage;
                // city membership depends only on the validated road registry.
                increment_counter(raw.stats, "valid_trajectories");
                increment_counter(raw.stats, "road_observations", count);
                if (end < start) { increment_counter(raw.stats, "rejected_reversed_trajectories"); continue; }
                // Preserve the extraction stage's declared timestamp validity envelope.
                if (start < 1448899200 || end > 1454256000) {
                    increment_counter(raw.stats, "legacy_date_excluded_records");
                    increment_counter(raw.stats, "outside_day_records");
                    increment_counter(raw.stats, "outside_day_observations", count);
                    continue;
                }
                bool any_day_entry = false;
                for (std::size_t i = 0; i < count; ++i) {
                    const auto entry = times[i];
                    if (entry < day.start_epoch || entry >= day.end_epoch) {
                        increment_counter(raw.stats, "outside_day_observations"); continue;
                    }
                    any_day_entry = true;
                    if (roads[i] == 0 || roads[i] == std::numeric_limits<std::int64_t>::min()) {
                        increment_counter(raw.stats, "rejected_invalid_road_id"); continue;
                    }
                    const auto road_id = static_cast<RoadId>(roads[i] < 0 ? -roads[i] : roads[i]);
                    const auto road_lookup = graph.road_index.find(road_id);
                    if (road_lookup == graph.road_index.end()) {
                        increment_counter(raw.stats, "outside_city_observations"); continue;
                    }
                    const auto next_entry = i + 1 == count ? end : times[i + 1];
                    if (next_entry < entry) { increment_counter(raw.stats, "rejected_negative_duration"); continue; }
                    // entry is positive and within the selected day, so this subtraction cannot overflow.
                    const auto duration = static_cast<std::uint64_t>(next_entry - entry);
                    if (duration > kMaxWeight) { increment_counter(raw.stats, "rejected_duration_overflow"); continue; }
                    if (duration == 0) increment_counter(raw.stats, "zero_duration_observations");
                    const auto& road = graph.roads[road_lookup->second];
                    Weight seconds = static_cast<Weight>(duration);
                    // Keep the reference's floating-point constants and integer assignment semantics.
                    const double base_lower_bound = 0.83 * road.legacy_base_seconds;
                    if (seconds < base_lower_bound) seconds = static_cast<Weight>(base_lower_bound);
                    const auto distance_lower_bound = static_cast<Weight>(std::ceil(static_cast<double>(road.distance_m) / 33.33));
                    if (seconds < distance_lower_bound) seconds = distance_lower_bound;
                    if (seconds < 1) seconds = 1;
                    if (seconds != duration) increment_counter(raw.stats, "clamped_observations");
                    const auto offset = static_cast<std::uint64_t>(entry - day.start_epoch);
                    const auto key = static_cast<std::uint64_t>(road_lookup->second) * kDaySeconds + offset;
                    auto& mean = raw.timestamp_means[key];
                    if (mean.count != 0) increment_counter(raw.stats, "duplicate_timestamp_observations");
                    accumulate(mean, seconds);
                    increment_counter(raw.stats, "selected_day_observations");
                    increment(raw.stats.slot_observations[offset / kSlotSeconds]);
                    if (raw.stats.min_selected_epoch < 0 || entry < raw.stats.min_selected_epoch) raw.stats.min_selected_epoch = entry;
                    if (entry > raw.stats.max_selected_epoch) raw.stats.max_selected_epoch = entry;
                }
                if (!any_day_entry) increment_counter(raw.stats, "outside_day_records");
            } catch (const std::exception& error) {
                increment_counter(raw.stats, "malformed_records");
                throw std::runtime_error(path.string() + ":" + std::to_string(line_number) + ": " + error.what());
            }
        }
        if (input.bad()) throw std::runtime_error("raw shard read failure: " + path.string());
        std::cerr << "raw_file " << raw.stats.counters.at("raw_files") << '/' << ordered_raw_paths.size()
                  << " completed; cumulative_selected_observations="
                  << raw.stats.counters.at("selected_day_observations") << '\n';
    }
    raw.stats.counters["timestamp_samples"] = raw.timestamp_means.size();
    return raw;
}

} // namespace sdc
