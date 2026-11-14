#include "corpus/npz.hpp"

#include "sdc/io.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#include <sstream>
#include <unistd.h>

namespace corpus::npz {
namespace {
[[noreturn]] void fail(const std::string& reason) {
    throw sdc::io::InputError("NPZ: " + reason);
}
class File {
public:
    explicit File(const std::filesystem::path& path) {
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (fd_ < 0) fail("cannot exclusively create " + path.filename().string() + ": " + std::strerror(errno));
    }
    ~File() { if (fd_ >= 0) ::close(fd_); }
    void write(const void* pointer, std::size_t bytes) {
        auto data = static_cast<const unsigned char*>(pointer);
        while (bytes) {
            const auto count = ::write(fd_, data, std::min<std::size_t>(bytes, 1U << 30));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) fail("write failed: " + std::string(std::strerror(errno)));
            data += count; bytes -= static_cast<std::size_t>(count); position_ += static_cast<std::uint64_t>(count);
        }
    }
    void integer(std::uint32_t value, unsigned width) {
        std::array<unsigned char, 4> bytes{};
        for (unsigned i = 0; i < width; ++i) bytes[i] = static_cast<unsigned char>(value >> (8 * i));
        write(bytes.data(), width);
    }
    std::uint64_t position() const { return position_; }
    void close() {
        const int descriptor = fd_; fd_ = -1;
        if (::close(descriptor) != 0) fail("close failed: " + std::string(std::strerror(errno)));
    }
private:
    int fd_ = -1;
    std::uint64_t position_ = 0;
};
std::uint32_t narrow(std::uint64_t value) {
    if (value >= std::numeric_limits<std::uint32_t>::max()) fail("shard exceeds ZIP32 limit");
    return static_cast<std::uint32_t>(value);
}
const std::array<std::uint32_t, 256>& crc_table() {
    static const auto values = [] {
        std::array<std::uint32_t, 256> table{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            auto value = i;
            for (unsigned j = 0; j < 8; ++j) value = (value >> 1) ^ ((value & 1) ? 0xedb88320U : 0U);
            table[i] = value;
        }
        return table;
    }();
    return values;
}
std::uint32_t crc_update(std::uint32_t crc, const void* pointer, std::size_t bytes) {
    const auto* data = static_cast<const unsigned char*>(pointer);
    const auto& table = crc_table();
    for (std::size_t i = 0; i < bytes; ++i) crc = table[(crc ^ data[i]) & 255] ^ (crc >> 8);
    return crc;
}
std::size_t item_width(const std::string& descriptor) {
    if (descriptor == "<f4") return 4;
    if (descriptor == "<i8" || descriptor == "<u8") return 8;
    if (descriptor == "|u1") return 1;
    fail("unsupported array dtype " + descriptor);
}
bool little_endian() {
    const std::uint16_t value = 1;
    return *reinterpret_cast<const unsigned char*>(&value) == 1;
}
template<class Consumer>
void consume_little(const ArrayView& array, Consumer consume) {
    const auto width = item_width(array.descriptor);
    if (little_endian() || width == 1) { consume(array.data, static_cast<std::size_t>(array.bytes)); return; }
    std::array<unsigned char, 256> scratch{};
    const auto* input = static_cast<const unsigned char*>(array.data);
    for (std::uint64_t offset = 0; offset < array.bytes;) {
        const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(scratch.size(), array.bytes - offset));
        for (std::size_t i = 0; i < count; i += width)
            for (std::size_t j = 0; j < width; ++j) scratch[i + j] = input[offset + i + width - 1 - j];
        consume(scratch.data(), count); offset += count;
    }
}
std::string npy_header(const ArrayView& array) {
    std::ostringstream shape;
    shape << '(';
    std::uint64_t elements = 1;
    for (std::size_t i = 0; i < array.shape.size(); ++i) {
        if (i) shape << ", ";
        shape << array.shape[i];
        if (array.shape[i] && elements > std::numeric_limits<std::uint64_t>::max() / array.shape[i]) fail("shape overflow");
        elements *= array.shape[i];
    }
    if (array.shape.size() == 1) shape << ',';
    shape << ')';
    const auto width = item_width(array.descriptor);
    if (elements > std::numeric_limits<std::uint64_t>::max() / width || elements * width != array.bytes)
        fail("array shape and byte length disagree");
    if (array.bytes && !array.data) fail("null array data");
    std::string dictionary = "{'descr': '" + array.descriptor + "', 'fortran_order': False, 'shape': " + shape.str() + ", }";
    const auto padding = (64 - (10 + dictionary.size() + 1) % 64) % 64;
    dictionary.append(padding, ' '); dictionary += '\n';
    if (dictionary.size() > 65535) fail("NPY v1 header too large");
    std::string header("\x93NUMPY\x01\x00", 8);
    header += static_cast<char>(dictionary.size() & 255);
    header += static_cast<char>((dictionary.size() >> 8) & 255);
    return header + dictionary;
}
struct Member {
    std::string name;
    std::uint32_t size = 0, crc = 0, offset = 0;
};
} // namespace

std::uint64_t write_store(const std::filesystem::path& path, const std::vector<ArrayView>& arrays) {
    if (arrays.empty() || arrays.size() >= 65535) fail("invalid member count");
    std::set<std::string> names;
    // Validate all headers before creating a file. No error is silently skipped.
    for (const auto& array : arrays) {
        if (array.name.empty() || array.name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") != std::string::npos ||
            !names.insert(array.name).second) fail("invalid or duplicate member name");
        narrow(npy_header(array).size() + array.bytes);
    }
    File file(path);
    std::vector<Member> members; members.reserve(arrays.size());
    for (const auto& array : arrays) {
        const auto header = npy_header(array);
        Member member; member.name = array.name + ".npy";
        member.size = narrow(header.size() + array.bytes); member.offset = narrow(file.position());
        auto crc = crc_update(0xffffffffU, header.data(), header.size());
        consume_little(array, [&](const void* data, std::size_t bytes) { crc = crc_update(crc, data, bytes); });
        member.crc = crc ^ 0xffffffffU;
        file.integer(0x04034b50U, 4); file.integer(20, 2); // Local header and required ZIP version.
        file.integer(0, 2); file.integer(0, 2); // Flags and ZIP store method.
        file.integer(0, 2); file.integer(33, 2); // Deterministic 1980-01-01 timestamp.
        file.integer(member.crc, 4); file.integer(member.size, 4); file.integer(member.size, 4);
        file.integer(static_cast<std::uint32_t>(member.name.size()), 2); file.integer(0, 2);
        file.write(member.name.data(), member.name.size()); file.write(header.data(), header.size());
        consume_little(array, [&](const void* data, std::size_t bytes) { file.write(data, bytes); });
        members.push_back(std::move(member));
    }
    const auto central_offset = narrow(file.position());
    for (const auto& member : members) {
        file.integer(0x02014b50U, 4); file.integer(20, 2); file.integer(20, 2);
        file.integer(0, 2); file.integer(0, 2); file.integer(0, 2); file.integer(33, 2);
        file.integer(member.crc, 4); file.integer(member.size, 4); file.integer(member.size, 4);
        file.integer(static_cast<std::uint32_t>(member.name.size()), 2);
        file.integer(0, 2); file.integer(0, 2); file.integer(0, 2); file.integer(0, 2); file.integer(0, 4);
        file.integer(member.offset, 4); file.write(member.name.data(), member.name.size());
    }
    const auto central_size = narrow(file.position() - central_offset);
    file.integer(0x06054b50U, 4); file.integer(0, 2); file.integer(0, 2);
    file.integer(static_cast<std::uint32_t>(members.size()), 2); file.integer(static_cast<std::uint32_t>(members.size()), 2);
    file.integer(central_size, 4); file.integer(central_offset, 4); file.integer(0, 2);
    const auto bytes = file.position(); narrow(bytes); file.close(); return bytes;
}

} // namespace corpus::npz
