
#include "Context.hpp"
#include "Runtime.hpp"
#include <stdexcept>
#include <iostream>

namespace js {

ContextRef::ContextRef(::JSContext* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

void ContextRef::compile(const std::string& script) {
	int flags = JS_EVAL_TYPE_GLOBAL | JS_EVAL_FLAG_COMPILE_ONLY;

	Value result(JS_Eval(handle, script.c_str(), script.length(), "<input>", flags), handle);
	if (result.is_exception()) {
		Value exception_val = get_exception();
		std::string message = exception_val;
		throw std::runtime_error(message);
	}
}

Value ContextRef::get_exception() {
	return Value(JS_GetException(handle), handle);
}

Context::Context(): ContextRef(JS_NewContext(Runtime::instance().handle)) {
}

Context::~Context() {
	if (handle) {
		JS_FreeContext(handle);
	}
}

Context::Context(Context&& other): ContextRef(other.handle) {
	other.handle = nullptr;
}

Context& Context::operator=(Context&& other) {
	std::swap(handle, other.handle);
	return *this;
}

}
