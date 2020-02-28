
#pragma once

#include "CString.hpp"

namespace quickjs {

struct Value;

struct ValueRef {
	ValueRef(JSValue handle, JSContext* context);

	operator bool() const;
	operator std::string() const;

	bool is_exception() const;
	bool is_error() const;
	bool is_undefined() const;
	bool is_bool() const;
	bool is_string() const;

	Value get_property_str(const std::string& name) const;
	void set_property_str(const std::string& name, Value val);

	::JSValue handle;
	::JSContext* context = nullptr;
};

std::ostream& operator<<(std::ostream& stream, const ValueRef& value);

struct Value: ValueRef {
	Value() = delete;
	using ValueRef::ValueRef;
	~Value();

	Value(const Value& other);
	Value& operator=(const Value& other);

	JSValue release();
};

}
