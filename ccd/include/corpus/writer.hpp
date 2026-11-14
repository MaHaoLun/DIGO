#pragma once

#include "corpus/graph.hpp"
#include "corpus/records.hpp"
#include "sdc/io.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace corpus {

// Consumes frozen query identities without assigning or changing splits. Each
// split owns an independent byte-bounded contiguous-array buffer. finish() must
// be called explicitly; a destructor never publishes unfinished output.
class CorpusWriter {
public:
    CorpusWriter(const std::filesystem::path& output_dir, const Graph& graph,
                 std::uint64_t shard_byte_budget = 67108864,
                 std::vector<Split> active_splits = {Split::train, Split::validation, Split::base_test});
    ~CorpusWriter();
    CorpusWriter(const CorpusWriter&) = delete;
    CorpusWriter& operator=(const CorpusWriter&) = delete;
    void append(const RouteRecord& route);
    sdc::io::Json finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace corpus
