// A small JSON parser (RFC 8259) building a value tree, enough for glTF.
#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace photon::json {

class Value {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    bool is_null() const { return type == Type::Null; }
    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }

    bool has(const std::string& key) const { return type == Type::Object && obj.count(key) > 0; }

    // Member access; a missing member (or a non-object) yields a null value.
    const Value& operator[](const std::string& key) const {
        static const Value null_value;
        if (type != Type::Object) return null_value;
        auto it = obj.find(key);
        return it == obj.end() ? null_value : it->second;
    }
    const Value& operator[](size_t i) const {
        static const Value null_value;
        return type == Type::Array && i < arr.size() ? arr[i] : null_value;
    }
    size_t size() const { return type == Type::Array ? arr.size() : (type == Type::Object ? obj.size() : 0); }

    double number(double fallback = 0) const { return type == Type::Number ? num : fallback; }
    int integer(int fallback = -1) const { return type == Type::Number ? int(num) : fallback; }
    bool boolean(bool fallback = false) const { return type == Type::Bool ? b : fallback; }
    const std::string& string() const { return str; }
};

struct ParseError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

Value parse(const std::string& text);

// Serialization (used to write glTF files of built-in scenes).
std::string escape(const std::string& s);

}  // namespace photon::json
