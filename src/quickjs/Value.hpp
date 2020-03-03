
#pragma once

#include "CString.hpp"
#include <memory>
#include "nn/Context.hpp"

namespace quickjs {

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

	Value get_property_str(const std::string& name) const;
	void set_property_str(const std::string& name, Value val);

	Value get_property_uint32(size_t index) const;
	void set_property_uint32(size_t index, Value val);

	Value get_property(JSAtom property) const;
	void set_property(JSAtom property, Value val);

	::JSValue handle;
	::JSContext* context = nullptr;
};

std::ostream& operator<<(std::ostream& stream, const ValueRef& value);

struct Value: ValueRef {
	Value() = delete;
	using ValueRef::ValueRef;
	virtual ~Value();

	Value(const Value& other);
	Value& operator=(const Value& other);

	JSValue release();

	std::shared_ptr<nn::Rect> opaque;
};

struct ObjectValue: Value {
	ObjectValue(int class_id, JSValue handle, JSContext* context): Value(handle, context), class_id(class_id) {}
	using Value::Value;

	std::shared_ptr<nn::Rect> get_opaque() const;
	void set_opaque(std::shared_ptr<nn::Rect> opaque);

private:
	int class_id;
	std::shared_ptr<nn::Rect> opaque;
};

struct ArrayValue: Value {
	ArrayValue(size_t length, JSValue handle, JSContext* context);

	Value get_elem(size_t index) const;
	void set_elem(size_t index, Value val);

	size_t size() const;
};

}
