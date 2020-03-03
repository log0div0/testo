
#include "Value.hpp"
#include "nn/Context.hpp"
#include <stdexcept>

namespace quickjs {

ValueRef::ValueRef(JSValue handle, JSContext* context): handle(handle), context(context) {
	if (!context) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

ValueRef::operator bool() const {
	if (!is_bool()) {
		throw std::runtime_error("Can't convert jsvalue to bool because it is not a bool");
	}
	return JS_ToBool(context, handle);
}

ValueRef::operator std::string() const {
	const char* str = JS_ToCString(context, handle);
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

bool ValueRef::is_object() const {
	return JS_IsObject(handle);
}

bool ValueRef::is_undefined() const {
	return JS_IsUndefined(handle);
}

bool ValueRef::is_bool() const {
	return JS_IsBool(handle);
}

bool ValueRef::is_string() const {
	return JS_IsString(handle);
}

Value ValueRef::get_property_str(const std::string& name) const {
	return Value(JS_GetPropertyStr(context, handle, name.c_str()), context);
}

void ValueRef::set_property_str(const std::string& name, Value val) {
	if (JS_SetPropertyStr(context, handle, name.c_str(), val.release()) < 0) {
		throw std::runtime_error("Can't set property " + name);
	}
}

Value ValueRef::get_property(JSAtom property) const {
	return Value(JS_GetProperty(context, handle, property), context);
}

void ValueRef::set_property(JSAtom property, Value val) {
	if (JS_SetProperty(context, handle, property, val.release()) < 0) {
		throw std::runtime_error("Can't set property ");
	}
}

void* ValueRef::get_opaque(int class_id) const {
	return JS_GetOpaque(handle, class_id);
}

void ValueRef::set_opaque(void* ptr) {
	JS_SetOpaque(handle, ptr);
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
