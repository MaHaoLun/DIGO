#include "sdc/io.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sdc::io {
namespace fs = std::filesystem;
namespace {
bool within(const fs::path& path, const fs::path& root) {
    auto p = path.begin();
    for (auto r = root.begin(); r != root.end(); ++r, ++p) if (p == path.end() || *p != *r) return false;
    return true;
}
void lexical(const fs::path& path) {
    if (path.empty() || path.string().find('\0') != std::string::npos) throw BoundaryViolation("empty path or NUL byte");
    for (const auto& part : path) if (part == "..") throw BoundaryViolation("parent traversal is forbidden");
}
void no_link(const fs::path& path) {
    struct stat info{};
    if (::lstat(path.c_str(), &info) != 0) throw BoundaryViolation("cannot inspect path: " + path.filename().string());
    if (S_ISLNK(info.st_mode)) throw BoundaryViolation("symlinks are forbidden: " + path.filename().string());
    if (S_ISREG(info.st_mode) && info.st_nlink != 1) throw BoundaryViolation("hardlinks are forbidden: " + path.filename().string());
    if (!S_ISREG(info.st_mode) && !S_ISDIR(info.st_mode)) throw BoundaryViolation("special files are forbidden: " + path.filename().string());
}
void scan_workspace(const fs::path& root) {
    no_link(root);
    for (const auto& entry : fs::recursive_directory_iterator(root)) no_link(entry.path());
}
void require_roots(const Json& config, const std::string& field, const char* expected) {
    const auto& values = config.at(field).array();
    if (values.size() != 1 || values[0].string() != expected) throw BoundaryViolation("unsupported boundary " + field);
}
} // namespace

Boundaries::Boundaries(const Options& options) {
    const auto* pythonpath = std::getenv("PYTHONPATH");
    if (pythonpath && *pythonpath) throw BoundaryViolation("PYTHONPATH must be unset or empty");
    lexical(options.workspace_root); lexical(options.input_dir);
    if (options.run_id.empty() || options.run_id == "." || options.run_id == "..") throw BoundaryViolation("run-id must be one new path component");
    for (unsigned char c : options.run_id) if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')) throw BoundaryViolation("invalid run-id");
    if (fs::is_symlink(options.workspace_root)) throw BoundaryViolation("workspace root is a symlink");
    root_ = fs::canonical(options.workspace_root);
    if (!fs::is_directory(root_)) throw BoundaryViolation("workspace root is not a directory");
    scan_workspace(root_);
    const auto contract_path = root_ / "boundaries.json";
    no_link(contract_path);
    auto config = read_json(contract_path);
    lexical(fs::path(config.at("workspace_root").string()));
    if (fs::canonical(config.at("workspace_root").string()) != root_) throw BoundaryViolation("configured workspace root differs from --workspace-root");
    require_roots(config, "data_input_roots", "data/staged");
    require_roots(config, "runtime_output_roots", "runs");
    const auto& policy = config.at("policy");
    if (!policy.at("require_empty_pythonpath").boolean() || policy.at("allow_symlinks").boolean() || policy.at("allow_hardlinks").boolean() || !policy.at("require_new_runtime_output_directory").boolean()) throw BoundaryViolation("boundary policy weakened");
    if (policy.contains("historical_assets_are_inputs_by_default") && policy.at("historical_assets_are_inputs_by_default").boolean()) throw BoundaryViolation("historical assets cannot be default inputs");
    if (policy.contains("allow_linked_git_worktree") && policy.at("allow_linked_git_worktree").boolean()) throw BoundaryViolation("linked worktrees cannot be allowed");
    if (config.contains("project_import_roots")) require_roots(config, "project_import_roots", "src");
    if (fs::is_regular_file(root_ / ".git")) throw BoundaryViolation("linked git worktrees are forbidden");
    for (const auto& denied : config.at("denied_roots").array()) {
        auto path = fs::weakly_canonical(denied.at("path").string());
        denied_.push_back(path);
        if (within(root_, path)) throw BoundaryViolation("workspace lies inside a denied root");
    }
    staged_ = root_ / "data" / "staged";
    if (!fs::is_directory(staged_) || !fs::is_directory(root_ / "runs")) throw BoundaryViolation("staged and runs directories must already exist");
    package_ = options.input_dir.is_absolute() ? options.input_dir : root_ / options.input_dir;
    package_ = fs::weakly_canonical(package_);
    if (!within(package_, staged_) || package_ == staged_) throw BoundaryViolation("input-dir must be a staged package directory");
    if (!fs::is_directory(package_)) throw BoundaryViolation("input package does not exist");
    output_ = root_ / "runs" / options.run_id;
    if (fs::exists(output_) || fs::is_symlink(output_)) throw BoundaryViolation("output already exists; runtime outputs are exclusive");
}

fs::path Boundaries::input(const std::string& raw) const {
    fs::path path(raw); lexical(path);
    if (path.is_absolute()) throw BoundaryViolation("listed input paths must be relative");
    auto it = path.begin(); const bool data = it != path.end() && *it++ == "data";
    const bool workspace_relative = data && it != path.end() && *it == "staged";
    auto candidate = fs::weakly_canonical((workspace_relative ? root_ : package_) / path);
    if (!within(candidate, staged_)) throw BoundaryViolation("input escapes staged-data root");
    for (const auto& denied : denied_) if (within(candidate, denied)) throw BoundaryViolation("input resolves inside denied root");
    if (!fs::exists(candidate) && !fs::is_symlink(candidate)) throw InputError("required input does not exist: " + path.generic_string());
    verify_regular(candidate);
    return candidate;
}
void Boundaries::verify_regular(const fs::path& path) const {
    if (!within(path, root_)) throw BoundaryViolation("file escapes workspace");
    auto current = root_;
    for (const auto& part : path.lexically_relative(root_)) { current /= part; no_link(current); }
    if (!fs::is_regular_file(path)) throw BoundaryViolation("input is not a regular file: " + path.filename().string());
}
std::string Boundaries::relative(const fs::path& path) const { return path.lexically_relative(root_).generic_string(); }
void Boundaries::create_output() const {
    no_link(root_ / "runs");
    if (!fs::create_directory(output_)) throw BoundaryViolation("output already exists; runtime outputs are exclusive");
}

void write_text_exclusive(const fs::path& path, const std::string& text) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0644);
    if (fd < 0) throw std::runtime_error("exclusive output open failed: " + path.filename().string() + ": " + std::strerror(errno));
    std::size_t offset = 0;
    while (offset < text.size()) {
        auto count = ::write(fd, text.data() + offset, text.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) { auto reason = std::string(std::strerror(errno)); ::close(fd); throw std::runtime_error("output write failed: " + reason); }
        offset += static_cast<std::size_t>(count);
    }
    if (::fsync(fd) != 0) { auto reason = std::string(std::strerror(errno)); ::close(fd); throw std::runtime_error("output flush failed: " + reason); }
    if (::close(fd) != 0) throw std::runtime_error("output close failed");
}
} // namespace sdc::io
