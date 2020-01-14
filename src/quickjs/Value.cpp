
#include "Value.hpp"
#include <stdexcept>

namespace quickjs {

Value::Value(JSValue handle, JSContext* context): handle(handle), context(context) {
	if (!context) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Value::~Value() {
	//context should always be valid
	JS_FreeValue(context, handle);
}

Value::Value(const Value& other): context(other.context) {
	handle = JS_DupValue(context, other.handle);
}

Value& Value::operator=(const Value& other) {
	context = other.context;
	handle = JS_DupValue(context, other.handle);
	return *this;
}

Value::operator bool() {
	if (!is_bool()) {
		throw std::runtime_error("Can't convert jsvalue to bool because it is not a bool");
	}
	return JS_ToBool(context, handle);
}

Value::operator CString() {
	return CString(JS_ToCString(context, handle), context);
}

bool Value::is_exception() {
	return JS_IsException(handle);
}

bool Value::is_error() {
	return JS_IsError(context, handle);
}

bool Value::is_undefined() {
	return JS_IsUndefined(handle);
}

bool Value::is_bool() {
	return JS_IsBool(handle);
}

bool Value::is_string() {
	return JS_IsString(handle);
}

}
