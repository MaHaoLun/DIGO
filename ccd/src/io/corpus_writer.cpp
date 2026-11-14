#include "corpus/writer.hpp"

#include "corpus/encoding.hpp"
#include "corpus/npz.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace corpus {
namespace {
using Clock = std::chrono::steady_clock;
using Json = sdc::io::Json;
// Covers one EncodedRow, ZIP/NPY bookkeeping, buffer objects, and bounded context.
constexpr std::uint64_t working_bytes = 16384;
// Conservative per-index reservation also covers the JSON returned by finish().
constexpr std::uint64_t index_entry_bytes = 2048;
[[noreturn]] void invalid(const std::string& message) {
    throw sdc::io::InputError("corpus writer: " + message);
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) invalid("byte count overflow");
    return a + b;
}
std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b) invalid("byte count overflow");
    return a * b;
}
Json decimal(double value) {
    std::ostringstream out; out << std::setprecision(17) << value;
    return Json::number(out.str());
}
template<class T> std::uint64_t capacity_bytes(const std::vector<T>& values) {
    return multiply(values.capacity(), sizeof(T));
}
template<class T> npz::ArrayView view(const char* name, const char* descriptor,
                                     const std::vector<T>& values, std::vector<std::uint64_t> shape) {
    return {name, descriptor, std::move(shape), values.data(), multiply(values.size(), sizeof(T))};
}
struct Shard {
    std::uint64_t rows = 0, bytes = 0, route_bytes = 0, array_bytes = 0;
    std::uint64_t first_query_id = 0, last_query_id = 0;
    std::uint64_t first_row_index = 0, last_row_index = 0;
};
struct Buffer {
    std::vector<float> images, odt, arr;
    std::vector<std::uint64_t> query_id, interval_id, row_index, offsets;
    std::vector<std::int64_t> source, destination, departure, travel_time, node_ids, arrival_times;
    std::vector<std::uint8_t> status;
    std::vector<Shard> shards;
    std::uint64_t total_rows = 0, max_buffer_bytes = 0;

    template<class F> void each(F&& function) {
        function(images, 2048U); function(odt, 5U); function(arr, 1U);
        function(query_id, 1U); function(interval_id, 1U); function(row_index, 1U);
        function(source, 1U); function(destination, 1U); function(departure, 1U); function(travel_time, 1U);
        function(status, 1U); function(offsets, 0U); function(node_ids, 0U); function(arrival_times, 0U);
    }
    std::uint64_t allocated() {
        std::uint64_t bytes = 0;
        each([&](const auto& values, unsigned) { bytes = add(bytes, capacity_bytes(values)); });
        return bytes;
    }
    std::uint64_t payload() {
        std::uint64_t bytes = 0;
        each([&](const auto& values, unsigned) {
            bytes = add(bytes, multiply(values.size(), sizeof(typename std::decay_t<decltype(values)>::value_type)));
        });
        return bytes;
    }
    std::uint64_t overhead() const { return add(working_bytes, multiply(shards.capacity(), index_entry_bytes)); }
    void release_arrays() {
        each([](auto& values, unsigned) { std::decay_t<decltype(values)> empty; values.swap(empty); });
    }
};
std::string shard_stem(std::size_t index) {
    std::ostringstream name; name << "shard-" << std::setfill('0') << std::setw(6) << index;
    return name.str();
}
} // namespace

struct CorpusWriter::Impl {
    std::filesystem::path output;
    const Graph& graph;
    std::uint64_t budget;
    std::array<Buffer, 5> buffers;
    std::vector<Split> active_splits;
    std::array<bool, 5> active{};
    EncodingMetrics timings;
    double crc_write_seconds = 0, append_seconds = 0;
    std::uint64_t total_bytes = 0, total_rows = 0;
    std::string city, snapshot_id;
    bool finished = false, failed = false;

    Impl(std::filesystem::path directory, const Graph& input_graph, std::uint64_t byte_budget,
         std::vector<Split> splits)
        : output(std::move(directory)), graph(input_graph), budget(byte_budget), active_splits(std::move(splits)) {
        // Keep ZIP32 and size_t conversions unambiguous. Budgets describe RAM,
        // not compressed bytes; this writer exclusively uses ZIP store.
        if (budget < working_bytes + index_entry_bytes + 8305 || budget > 1024ULL * 1024 * 1024)
            invalid("shard byte budget must hold one row plus bookkeeping and be at most 1 GiB");
        if (!std::filesystem::is_directory(output)) invalid("output directory must already exist");
        if (active_splits.empty()) invalid("at least one active split is required");
        for (const auto split : active_splits) {
            const auto i = static_cast<unsigned>(split);
            if (i >= buffers.size() || active[i]) invalid("invalid or duplicate active split");
            active[i] = true;
        }
        for (const auto split : active_splits) {
            const auto directory_path = output / split_name(split);
            if (!std::filesystem::create_directory(directory_path))
                invalid("split directory already exists: " + directory_path.filename().string());
        }
    }

    void ensure_index_capacity(Buffer& buffer) {
        if (buffer.shards.size() < buffer.shards.capacity()) return;
        // Flush calls occur before this reservation. Releasing arrays avoids a
        // vector reallocation transient exceeding the declared memory bound.
        if (!buffer.arr.empty()) invalid("internal index reservation with nonempty shard");
        buffer.release_arrays();
        const auto next = buffer.shards.capacity() ? multiply(buffer.shards.capacity(), 2) : 1;
        const auto estimate = add(working_bytes, multiply(next, index_entry_bytes));
        if (estimate > budget || add(estimate, 8305) > budget)
            invalid("shard index exhausted byte budget; increase budget or reduce fragmentation");
        buffer.shards.reserve(static_cast<std::size_t>(next));
        if (add(buffer.overhead(), 8305) > budget) invalid("allocator exceeded reserved index budget");
    }

    bool reserve_row(Buffer& buffer, std::size_t route_length, std::uint64_t incoming_bytes, bool geometric) {
        std::array<std::uint64_t, 14> targets{};
        std::size_t item = 0;
        auto current = add(add(buffer.allocated(), buffer.overhead()), incoming_bytes);
        auto peak = current;
        // Simulate every reserve's old-plus-new allocation before changing any
        // vector. A failed simulation cannot partially append or lose a row.
        buffer.each([&](const auto& values, unsigned width) {
            using T = typename std::decay_t<decltype(values)>::value_type;
            const auto count = width ? add(values.size(), width) :
                (item == 11 ? add(buffer.arr.size(), 2) : add(values.size(), route_length));
            auto target = static_cast<std::uint64_t>(values.capacity());
            if (count > target) {
                target = geometric ? std::max(count, multiply(target, 2)) : count;
                const auto newly_allocated = multiply(target, sizeof(T));
                peak = std::max(peak, add(current, newly_allocated));
                current = add(current - capacity_bytes(values), newly_allocated);
            }
            targets[item++] = target;
        });
        if (peak > budget) return false;
        item = 0;
        buffer.each([&](auto& values, unsigned) {
            values.reserve(static_cast<std::size_t>(targets[item++]));
        });
        const auto resident = add(add(buffer.allocated(), buffer.overhead()), incoming_bytes);
        if (resident > budget) invalid("allocator exceeded shard byte budget");
        buffer.max_buffer_bytes = std::max(buffer.max_buffer_bytes, peak);
        return true;
    }

    void flush(unsigned split) {
        auto& buffer = buffers.at(split);
        if (buffer.arr.empty()) return;
        const auto rows = static_cast<std::uint64_t>(buffer.arr.size());
        if (buffer.offsets.size() != rows + 1 || buffer.offsets.back() != buffer.node_ids.size() ||
            buffer.node_ids.size() != buffer.arrival_times.size()) invalid("internal ragged alignment failure");
        const auto stem = shard_stem(buffer.shards.size());
        const auto directory = output / split_name(static_cast<Split>(split));
        const auto begin = Clock::now();
        const auto image_bytes = npz::write_store(directory / (stem + ".images.npz"), {
            view("image", "<f4", buffer.images, {rows, 2, 32, 32}),
            view("odt", "<f4", buffer.odt, {rows, 5}),
            view("arr", "<f4", buffer.arr, {rows})
        });
        const auto length = static_cast<std::uint64_t>(buffer.node_ids.size());
        const auto route_bytes = npz::write_store(directory / (stem + ".routes.npz"), {
            view("query_id", "<u8", buffer.query_id, {rows}),
            view("interval_id", "<u8", buffer.interval_id, {rows}),
            view("row_index", "<u8", buffer.row_index, {rows}),
            view("source", "<i8", buffer.source, {rows}),
            view("destination", "<i8", buffer.destination, {rows}),
            view("departure", "<i8", buffer.departure, {rows}),
            view("travel_time", "<i8", buffer.travel_time, {rows}),
            view("status", "|u1", buffer.status, {rows}),
            view("offsets", "<u8", buffer.offsets, {rows + 1}),
            view("node_ids", "<i8", buffer.node_ids, {length}),
            view("arrival_times", "<i8", buffer.arrival_times, {length})
        });
        crc_write_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
        Shard shard; shard.rows = rows; shard.bytes = add(image_bytes, route_bytes);
        shard.route_bytes = route_bytes; shard.array_bytes = buffer.payload();
        shard.first_query_id = buffer.query_id.front(); shard.last_query_id = buffer.query_id.back();
        shard.first_row_index = buffer.row_index.front(); shard.last_row_index = buffer.row_index.back();
        buffer.shards.push_back(shard); total_bytes = add(total_bytes, shard.bytes);
        buffer.each([](auto& values, unsigned) { values.clear(); });
    }

    void append(const RouteRecord& route) {
        if (finished || failed) invalid("writer is already finished or failed");
        const auto start = Clock::now();
        try {
            const auto split = static_cast<unsigned>(route.query.split);
            if (split >= buffers.size() || !active[split]) invalid("invalid or inactive frozen split");
            if (route.query.city.empty() || route.query.snapshot_id.empty() ||
                route.query.city.size() + route.query.snapshot_id.size() > 1024)
                invalid("city and snapshot_id must be nonempty bounded identifiers");
            if (!total_rows) { city = route.query.city; snapshot_id = route.query.snapshot_id; }
            if (route.query.city != city || route.query.snapshot_id != snapshot_id)
                invalid("cannot mix city or snapshot identity");
            auto& buffer = buffers[split];
            ensure_index_capacity(buffer);
            // A single oversize route is an error, never split, padded, dropped,
            // or written through an unbounded temporary allocation.
            const auto incoming_bytes = add(capacity_bytes(route.node_ids), capacity_bytes(route.arrival_times));
            const auto one_row = add(8289, multiply(route.node_ids.size(), 16));
            if (add(add(one_row, buffer.overhead()), incoming_bytes) > budget)
                invalid("one route exceeds shard byte budget");
            const auto encoded = encode_route(graph, route, &timings);
            if (!reserve_row(buffer, route.node_ids.size(), incoming_bytes, true) &&
                !reserve_row(buffer, route.node_ids.size(), incoming_bytes, false)) {
                flush(split);
                buffer.release_arrays();
                ensure_index_capacity(buffer);
                if (!reserve_row(buffer, route.node_ids.size(), incoming_bytes, false))
                    invalid("one route exceeds remaining shard/index byte budget");
            }
            if (buffer.offsets.empty()) buffer.offsets.push_back(0);
            buffer.images.insert(buffer.images.end(), encoded.image.begin(), encoded.image.end());
            buffer.odt.insert(buffer.odt.end(), encoded.odt.begin(), encoded.odt.end());
            buffer.arr.push_back(encoded.arr);
            buffer.query_id.push_back(route.query.query_id); buffer.interval_id.push_back(route.query.interval_id);
            buffer.row_index.push_back(route.query.row_index); buffer.source.push_back(route.query.source);
            buffer.destination.push_back(route.query.destination); buffer.departure.push_back(route.query.departure);
            buffer.travel_time.push_back(route.travel_time); buffer.status.push_back(0);
            buffer.node_ids.insert(buffer.node_ids.end(), route.node_ids.begin(), route.node_ids.end());
            buffer.arrival_times.insert(buffer.arrival_times.end(), route.arrival_times.begin(), route.arrival_times.end());
            buffer.offsets.push_back(buffer.node_ids.size());
            ++buffer.total_rows; ++total_rows;
            append_seconds += std::chrono::duration<double>(Clock::now() - start).count();
        } catch (...) { failed = true; throw; }
    }

    Json finish() {
        if (finished || failed) invalid("cannot finish a finished or failed writer");
        try {
            const auto begin = Clock::now();
            for (const auto split : active_splits) flush(static_cast<unsigned>(split));
            append_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
            // Array storage is released before materializing the final index.
            for (auto& buffer : buffers) buffer.release_arrays();
            Json::Object splits;
            std::uint64_t maximum_total_buffer_bytes = 0;
            for (const auto split_id : active_splits) {
                const auto i = static_cast<unsigned>(split_id);
                const auto& buffer = buffers[i];
                maximum_total_buffer_bytes = add(maximum_total_buffer_bytes, buffer.max_buffer_bytes);
                Json::Array shards;
                shards.reserve(buffer.shards.size());
                for (std::size_t j = 0; j < buffer.shards.size(); ++j) {
                    const auto& shard = buffer.shards[j];
                    const std::string stem = std::string(split_name(static_cast<Split>(i))) + '/' + shard_stem(j);
                    shards.emplace_back(Json::Object{
                        {"image_path", stem + ".images.npz"}, {"route_path", stem + ".routes.npz"},
                        {"rows", shard.rows}, {"bytes", shard.bytes}, {"route_bytes", shard.route_bytes},
                        {"array_bytes", shard.array_bytes}, {"byte_budget", budget},
                        {"first_query_id", shard.first_query_id}, {"last_query_id", shard.last_query_id},
                        {"first_row_index", shard.first_row_index}, {"last_row_index", shard.last_row_index}
                    });
                }
                Json::Object split;
                split.emplace("rows", buffer.total_rows);
                split.emplace("shards", std::move(shards));
                split.emplace("peak_reserved_buffer_bytes", buffer.max_buffer_bytes);
                splits.emplace(split_name(static_cast<Split>(i)), std::move(split));
            }
            finished = true;
            Json::Object result;
            result.emplace("splits", std::move(splits));
            result.emplace("route_context", Json::Object{{"city", city}, {"snapshot_id", snapshot_id},
                    {"status_encoding", Json::Object{{"0", "ok"}}}});
            result.emplace("conversion_metrics", Json::Object{
                    {"mapping_seconds", decimal(timings.mapping_seconds)},
                    {"rasterize_seconds", decimal(timings.rasterize_seconds)},
                    {"encode_seconds", decimal(timings.encode_seconds)},
                    {"crc_write_seconds", decimal(crc_write_seconds)},
                    {"conversion_seconds", decimal(append_seconds)},
                    {"rows_per_second", decimal(append_seconds > 0 ? total_rows / append_seconds : 0)},
                    {"rows", total_rows}, {"bytes", total_bytes}, {"zip_method", "store"},
                    {"shard_byte_budget", budget},
                    {"maximum_combined_reserved_buffer_bytes", maximum_total_buffer_bytes},
                    {"maximum_buffer_budget_bytes", multiply(budget, active_splits.size())},
                    {"memory_accounting", "incoming RouteRecord capacities and output array capacities including reserve transients; 16 KiB workspace and bounded shard index per split"}
                });
            return result;
        } catch (...) { failed = true; throw; }
    }
};

CorpusWriter::CorpusWriter(const std::filesystem::path& output_dir, const Graph& graph, std::uint64_t budget,
                           std::vector<Split> active_splits)
    : impl_(std::make_unique<Impl>(output_dir, graph, budget, std::move(active_splits))) {}
CorpusWriter::~CorpusWriter() = default;
void CorpusWriter::append(const RouteRecord& route) { impl_->append(route); }
sdc::io::Json CorpusWriter::finish() { return impl_->finish(); }

} // namespace corpus
