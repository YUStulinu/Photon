#include "scene/json.h"

#include <cstdlib>

namespace photon::json {

namespace {

struct Parser {
    const std::string& s;
    size_t i = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        size_t line = 1;
        for (size_t k = 0; k < i && k < s.size(); k++)
            if (s[k] == '\n') line++;
        throw ParseError("JSON error at line " + std::to_string(line) + ": " + msg);
    }

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
    }

    bool consume(char c) {
        ws();
        if (i < s.size() && s[i] == c) {
            i++;
            return true;
        }
        return false;
    }

    void expect(char c) {
        if (!consume(c)) fail(std::string("expected '") + c + "'");
    }

    static void append_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }

    unsigned hex4() {
        if (i + 4 > s.size()) fail("truncated \\u escape");
        unsigned v = 0;
        for (int k = 0; k < 4; k++) {
            char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= unsigned(c - '0');
            else if (c >= 'a' && c <= 'f') v |= unsigned(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= unsigned(c - 'A' + 10);
            else fail("invalid \\u escape");
        }
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (i >= s.size()) fail("unterminated string");
            char c = s[i++];
            if (c == '"') break;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i >= s.size()) fail("unterminated escape");
            char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        i += 2;
                        unsigned lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail(std::string("invalid escape \\") + e);
            }
        }
        return out;
    }

    Value parse_value(int depth) {
        if (depth > 512) fail("nesting too deep");
        ws();
        if (i >= s.size()) fail("unexpected end of input");
        Value v;
        char c = s[i];
        if (c == '{') {
            i++;
            v.type = Value::Type::Object;
            if (consume('}')) return v;
            do {
                ws();
                std::string key = parse_string();
                expect(':');
                v.obj[key] = parse_value(depth + 1);
            } while (consume(','));
            expect('}');
        } else if (c == '[') {
            i++;
            v.type = Value::Type::Array;
            if (consume(']')) return v;
            do {
                v.arr.push_back(parse_value(depth + 1));
            } while (consume(','));
            expect(']');
        } else if (c == '"') {
            v.type = Value::Type::String;
            v.str = parse_string();
        } else if (s.compare(i, 4, "true") == 0) {
            i += 4;
            v.type = Value::Type::Bool;
            v.b = true;
        } else if (s.compare(i, 5, "false") == 0) {
            i += 5;
            v.type = Value::Type::Bool;
        } else if (s.compare(i, 4, "null") == 0) {
            i += 4;
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            const char* start = s.c_str() + i;
            char* end = nullptr;
            v.num = std::strtod(start, &end);
            if (end == start) fail("invalid number");
            i += size_t(end - start);
            v.type = Value::Type::Number;
        } else {
            fail(std::string("unexpected character '") + c + "'");
        }
        return v;
    }
};

}  // namespace

Value parse(const std::string& text) {
    Parser p{text};
    Value v = p.parse_value(0);
    p.ws();
    if (p.i != text.size()) p.fail("trailing characters after the JSON value");
    return v;
}

std::string escape(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            default: out += c;
        }
    }
    return out + "\"";
}

}  // namespace photon::json
