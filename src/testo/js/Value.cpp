
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

}
