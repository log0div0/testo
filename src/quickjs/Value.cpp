
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

bool Value::is_exception() {
	return JS_IsException(handle);
}

}
