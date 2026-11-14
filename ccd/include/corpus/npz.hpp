#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace corpus::npz {

// Array memory must remain valid for the duration of write_store(). Values are
// encoded as little-endian NPY, including on a big-endian build host.
struct ArrayView {
    std::string name;
    std::string descriptor;
    std::vector<std::uint64_t> shape;
    const void* data = nullptr;
    std::uint64_t bytes = 0;
};

// Exclusively creates a ZIP store NPZ. ZIP's required CRC is an archive-integrity
// field, not an artifact identity/content hash. ZIP64 is deliberately unsupported
// because shards are bounded below the ZIP32 size limit.
std::uint64_t write_store(const std::filesystem::path& path,
                          const std::vector<ArrayView>& arrays);

} // namespace corpus::npz
