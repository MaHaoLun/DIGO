#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace sdc::io {

class BoundaryViolation : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
class InputError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A small strict JSON value for local contracts. Numbers retain their exact text.
class Json {
public:
    struct Number { std::string text; };
    using Object = std::map<std::string, Json>;
    using Array = std::vector<Json>;
    using Value = std::variant<std::nullptr_t, bool, Number, std::string, Array, Object>;
    Json() : value_(nullptr) {}
    Json(bool v) : value_(v) {}
    Json(int v) : value_(Number{std::to_string(v)}) {}
    Json(std::int64_t v) : value_(Number{std::to_string(v)}) {}
    Json(std::uint64_t v) : value_(Number{std::to_string(v)}) {}
    Json(const char* v) : value_(std::string(v)) {}
    Json(std::string v) : value_(std::move(v)) {}
    Json(Array v) : value_(std::move(v)) {}
    Json(Object v) : value_(std::move(v)) {}
    static Json number(std::string text);
    static Json parse(const std::string& text);
    const Object& object() const;
    const Array& array() const;
    const std::string& string() const;
    bool boolean() const;
    std::uint64_t unsigned_integer() const;
    const Json& at(const std::string& key) const;
    bool contains(const std::string& key) const;
    std::string dump() const;
private:
    Value value_;
};

Json read_json(const std::filesystem::path& path);
void write_text_exclusive(const std::filesystem::path& path, const std::string& text);

struct Options {
    std::filesystem::path workspace_root;
    std::filesystem::path input_dir;
    std::string city;
    std::string date;
    std::string run_id;
};

class Boundaries {
public:
    explicit Boundaries(const Options& options);
    const std::filesystem::path& root() const { return root_; }
    const std::filesystem::path& package() const { return package_; }
    const std::filesystem::path& output() const { return output_; }
    std::filesystem::path input(const std::string& raw) const;
    std::string relative(const std::filesystem::path& path) const;
    void verify_regular(const std::filesystem::path& path) const;
    void create_output() const;
private:
    std::filesystem::path root_, staged_, package_, output_;
    std::vector<std::filesystem::path> denied_;
};

int build_snapshot(const Options& options);

} // namespace sdc::io
