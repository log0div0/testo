#include "Runtime.hpp"
#include <stdexcept>

namespace quickjs {

Runtime::Runtime(::JSRuntime* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Runtime::~Runtime() {
	if (handle) {
		JS_FreeRuntime(handle);
	}
}

Runtime::Runtime(Runtime&& other): handle(other.handle) {
	other.handle = nullptr;
}

Runtime& Runtime::operator=(Runtime&& other) {
	std::swap(handle, other.handle);
	return *this;
}

Runtime create_runtime() {
	return Runtime(JS_NewRuntime());
}

Context Runtime::create_context() {
	return Context(JS_NewContext(handle));
}

}