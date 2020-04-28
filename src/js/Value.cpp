
#include "Value.hpp"
#include "Context.hpp"
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace js {

ValueRef::ValueRef(JSValue handle, JSContext* context): handle(handle), context(context) {
	if (!context) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

ValueRef::operator int32_t() const {
	if (!is_integer()) {
		throw std::runtime_error("Can't convert jsvalue to int32_t because it is not an integer");
	}

	int32_t result;

	if(JS_ToInt32(context, &result, handle)) {
		throw std::runtime_error("Can't convert jsvalue to int32_t because...something");
	}

	return result;
}

ValueRef::operator bool() const {
	if (!is_bool()) {
		throw std::runtime_error("Can't convert jsvalue to bool because it is not a bool");
	}
	return JS_ToBool(context, handle);
}

ValueRef::operator std::string() const {
	const char* str = JS_ToCString(context, handle);
	if (!str) {
		throw std::runtime_error("Can't convert js value to string");
	}
	std::string result(str);
	JS_FreeCString(context, str);
	return result;
}

bool ValueRef::is_exception() const {
	return JS_IsException(handle);
}

bool ValueRef::is_error() const {
	return JS_IsError(context, handle);
}

bool ValueRef::is_undefined() const {
	return JS_IsUndefined(handle);
}

bool ValueRef::is_bool() const {
	return JS_IsBool(handle);
}

bool ValueRef::is_integer() const {
	return JS_IsInteger(handle);
}

bool ValueRef::is_string() const {
	return JS_IsString(handle);
}

bool ValueRef::is_array() const {
	return JS_IsArray(context, handle);
}

bool ValueRef::is_object() const {
	return JS_IsObject(handle);
}

bool ValueRef::is_function() const {
	return JS_IsFunction(context, handle);
}

bool ValueRef::is_constructor() const {
	return JS_IsConstructor(context, handle);
}

bool ValueRef::is_instance_of(ValueRef obj) const {
	return JS_IsInstanceOf(context, handle, obj.handle);
}

Value ValueRef::get_property_str(const std::string& name) const {
	return Value(JS_GetPropertyStr(context, handle, name.c_str()), context);
}

void ValueRef::set_property_str(const std::string& name, Value val) {
	if (JS_SetPropertyStr(context, handle, name.c_str(), val.release()) < 0) {
		throw std::runtime_error("Can't set property " + name);
	}
}

Value ValueRef::get_property_uint32(size_t index) const {
	return Value(JS_GetPropertyUint32(context, handle, index), context);
}

void ValueRef::set_property_uint32(size_t index, Value val) {
	if (JS_SetPropertyUint32(context, handle, index, val.release()) < 0) {
		throw std::runtime_error("Can't set property uint32");
	}
}

Value ValueRef::get_property_atom(JSAtom property) const {
	return Value(JS_GetProperty(context, handle, property), context);
}

void ValueRef::set_property_atom(JSAtom property, Value val) {
	if (JS_SetProperty(context, handle, property, val.release()) < 0) {
		throw std::runtime_error("Can't set property ");
	}
}

void ValueRef::set_property_function_list(const JSCFunctionListEntry *tab, int len) {
	JS_SetPropertyFunctionList(context, handle, tab, len);
}

void* ValueRef::get_opaque(JSClassID class_id) const {
	return JS_GetOpaque(handle, class_id);
}

void ValueRef::set_opaque(void* opaque) {
	JS_SetOpaque(handle, opaque);
}

std::ostream& operator<<(std::ostream& stream, const ValueRef& value) {
	std::string str = std::string(value);
	return stream << str;
}

Value::~Value() {
	if (context) {
		JS_FreeValue(context, handle);
	}
}

Value::Value(const Value& other): ValueRef(JS_DupValue(other.context, other.handle), other.context) {
}

Value& Value::operator=(const Value& other) {
	context = other.context;
	handle = JS_DupValue(other.context, other.handle);

	return *this;
}

JSValue Value::release() {
	context = nullptr;
	return handle;
}

}
