
#pragma once

#include "CString.hpp"
#include <memory>

namespace js {

struct Value;

struct ValueRef {
	ValueRef(JSValue handle, JSContext* context);

	operator int32_t() const;
	operator bool() const;
	operator std::string() const;

	bool is_exception() const;
	bool is_error() const;
	bool is_undefined() const;
	bool is_bool() const;
	bool is_integer() const;
	bool is_string() const;
	bool is_array() const;
	bool is_object() const;
	bool is_function() const;
	bool is_constructor() const;

	bool is_instance_of(ValueRef obj) const;

	Value get_property(const std::string& name) const;
	void set_property(const std::string& name, Value val);

	Value get_property(size_t index) const;
	void set_property(size_t index, Value val);

	Value get_property(JSAtom property) const;
	void set_property(JSAtom property, Value val);

	void* get_opaque(JSClassID class_id) const;
	void set_opaque(void* opaque);

	void set_property_function_list(const JSCFunctionListEntry *tab, int len);

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
