#include "sdc/io.hpp"

#include <charconv>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace sdc::io {
namespace {
[[noreturn]] void invalid(const std::string& message) { throw InputError("JSON: " + message); }
std::string quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c) << std::dec;
            else out << static_cast<char>(c);
        }
    }
    out << '"';
    return out.str();
}
class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}
    Json parse() { Json value = item(0); space(); if (pos_ != text_.size()) invalid("trailing content"); return value; }
private:
    const std::string& text_;
    std::size_t pos_ = 0;
    void space() { while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' || text_[pos_] == '\r' || text_[pos_] == '\t')) ++pos_; }
    bool take(char c) { space(); if (pos_ < text_.size() && text_[pos_] == c) { ++pos_; return true; } return false; }
    void expect(char c) { if (!take(c)) invalid(std::string("expected '") + c + "'"); }
    std::uint32_t hex4() {
        if (text_.size() - pos_ < 4) invalid("short unicode escape");
        std::uint32_t n = 0;
        for (int i = 0; i < 4; ++i) {
            char c = text_[pos_++]; n <<= 4;
            if (c >= '0' && c <= '9') n += c - '0';
            else if (c >= 'a' && c <= 'f') n += c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') n += c - 'A' + 10;
            else invalid("invalid unicode escape");
        }
        return n;
    }
    static void utf8(std::string& out, std::uint32_t n) {
        if (n < 0x80) out += static_cast<char>(n);
        else if (n < 0x800) { out += static_cast<char>(0xc0 | (n >> 6)); out += static_cast<char>(0x80 | (n & 63)); }
        else if (n < 0x10000) { out += static_cast<char>(0xe0 | (n >> 12)); out += static_cast<char>(0x80 | ((n >> 6) & 63)); out += static_cast<char>(0x80 | (n & 63)); }
        else { out += static_cast<char>(0xf0 | (n >> 18)); out += static_cast<char>(0x80 | ((n >> 12) & 63)); out += static_cast<char>(0x80 | ((n >> 6) & 63)); out += static_cast<char>(0x80 | (n & 63)); }
    }
    std::string string() {
        expect('"'); std::string out;
        while (pos_ < text_.size()) {
            unsigned char c = text_[pos_++];
            if (c == '"') return out;
            if (c < 32) invalid("unescaped control character");
            if (c != '\\') { out += static_cast<char>(c); continue; }
            if (pos_ == text_.size()) invalid("unterminated escape");
            switch (text_[pos_++]) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                auto n = hex4();
                if (n >= 0xd800 && n <= 0xdbff) {
                    if (text_.size() - pos_ < 6 || text_[pos_++] != '\\' || text_[pos_++] != 'u') invalid("unpaired high surrogate");
                    auto low = hex4(); if (low < 0xdc00 || low > 0xdfff) invalid("invalid low surrogate");
                    n = 0x10000 + ((n - 0xd800) << 10) + low - 0xdc00;
                } else if (n >= 0xdc00 && n <= 0xdfff) invalid("unpaired low surrogate");
                utf8(out, n); break;
            }
            default: invalid("invalid escape");
            }
        }
        invalid("unterminated string");
    }
    Json item(unsigned depth) {
        if (depth > 64) invalid("nesting exceeds 64");
        space(); if (pos_ == text_.size()) invalid("unexpected end");
        if (text_[pos_] == '"') return Json(string());
        if (take('{')) {
            Json::Object out; if (take('}')) return out;
            do { auto key = string(); expect(':'); if (!out.emplace(std::move(key), item(depth + 1)).second) invalid("duplicate key"); } while (take(','));
            expect('}'); return out;
        }
        if (take('[')) {
            Json::Array out; if (take(']')) return out;
            do { out.push_back(item(depth + 1)); } while (take(','));
            expect(']'); return out;
        }
        for (const auto& literal : {std::string("true"), std::string("false"), std::string("null")}) {
            if (text_.compare(pos_, literal.size(), literal) == 0) { pos_ += literal.size(); if (literal == "null") return {}; return Json(literal == "true"); }
        }
        const auto begin = pos_;
        if (text_[pos_] == '-') ++pos_;
        if (pos_ == text_.size()) invalid("invalid number");
        if (text_[pos_] == '0') ++pos_;
        else { if (text_[pos_] < '1' || text_[pos_] > '9') invalid("invalid value"); while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_; }
        if (pos_ < text_.size() && text_[pos_] == '.') { ++pos_; auto digits = pos_; while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_; if (digits == pos_) invalid("empty decimal"); }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) { ++pos_; if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_; auto digits = pos_; while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_; if (digits == pos_) invalid("empty exponent"); }
        return Json::number(text_.substr(begin, pos_ - begin));
    }
};
} // namespace

Json Json::number(std::string text) { Json out; out.value_ = Number{std::move(text)}; return out; }
Json Json::parse(const std::string& text) { return Parser(text).parse(); }
const Json::Object& Json::object() const { if (!std::holds_alternative<Object>(value_)) invalid("expected object"); return std::get<Object>(value_); }
const Json::Array& Json::array() const { if (!std::holds_alternative<Array>(value_)) invalid("expected array"); return std::get<Array>(value_); }
const std::string& Json::string() const { if (!std::holds_alternative<std::string>(value_)) invalid("expected string"); return std::get<std::string>(value_); }
bool Json::boolean() const { if (!std::holds_alternative<bool>(value_)) invalid("expected boolean"); return std::get<bool>(value_); }
std::uint64_t Json::unsigned_integer() const {
    if (!std::holds_alternative<Number>(value_)) invalid("expected integer");
    const auto& s = std::get<Number>(value_).text; std::uint64_t n = 0;
    auto result = std::from_chars(s.data(), s.data() + s.size(), n);
    if (result.ec != std::errc{} || result.ptr != s.data() + s.size()) invalid("expected nonnegative 64-bit integer");
    return n;
}
const Json& Json::at(const std::string& key) const { auto it = object().find(key); if (it == object().end()) invalid("missing field: " + key); return it->second; }
bool Json::contains(const std::string& key) const { return object().count(key) != 0; }
std::string Json::dump() const {
    if (std::holds_alternative<std::nullptr_t>(value_)) return "null";
    if (std::holds_alternative<bool>(value_)) return std::get<bool>(value_) ? "true" : "false";
    if (std::holds_alternative<Number>(value_)) return std::get<Number>(value_).text;
    if (std::holds_alternative<std::string>(value_)) return quote(std::get<std::string>(value_));
    std::string out; bool first = true;
    if (std::holds_alternative<Array>(value_)) {
        out = "["; for (const auto& item : array()) { if (!first) out += ','; first = false; out += item.dump(); } return out + ']';
    }
    out = "{"; for (const auto& pair : object()) { if (!first) out += ','; first = false; out += quote(pair.first) + ':' + pair.second.dump(); } return out + '}';
}
Json read_json(const std::filesystem::path& path) {
    constexpr std::uintmax_t limit = 64 * 1024 * 1024;
    if (std::filesystem::file_size(path) > limit) throw InputError("JSON exceeds 64 MiB: " + path.filename().string());
    std::ifstream in(path, std::ios::binary); if (!in) throw InputError("cannot read JSON: " + path.filename().string());
    std::ostringstream text; text << in.rdbuf(); if (in.bad()) throw InputError("JSON read failed");
    return Json::parse(text.str());
}
} // namespace sdc::io
