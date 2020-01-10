
#include "Context.hpp"
#include <stdexcept>

namespace quickjs {

Context::Context(::JSContext* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Context::~Context() {
	if (handle) {
		JS_FreeContext(handle);
	}
}

Context::Context(Context&& other): handle(other.handle) {
	other.handle = nullptr;
}

Context& Context::operator=(Context&& other) {
	std::swap(handle, other.handle);
}

Value Context::get_global_object() {
	Value result(JS_GetGlobalObject(handle), handle);
	if (result.is_exception()) {
		throw std::runtime_error("Error while getting global object");
	}
	return result;
}

Value Context::eval(const std::string& script) {
	Value result(JS_Eval(handle, script.c_str(), script.length(), "<input>", JS_EVAL_TYPE_GLOBAL), handle);
	if (result.is_exception()) {
		throw std::runtime_error("Error while executing javascript");
	}
	return result;
}

}
