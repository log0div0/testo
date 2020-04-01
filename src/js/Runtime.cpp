
#include "Runtime.hpp"
#include <stdexcept>

namespace js {

RuntimeRef::RuntimeRef(::JSRuntime* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Context RuntimeRef::create_context(stb::Image* image) {
	return Context(JS_NewContext(handle), image);
}

Runtime::Runtime(::JSRuntime* handle): RuntimeRef(handle) {
	if(JS_NewClass(handle, nn_rect_class_id, &nn_rect_class)) {
		throw std::runtime_error("Can't register nn::rect js class");
	}
}

Runtime::~Runtime() {
	if (handle) {
		JS_FreeRuntime(handle);
	}
}

Runtime::Runtime(Runtime&& other): RuntimeRef(other.handle) {
	other.handle = nullptr;
}

Runtime& Runtime::operator=(Runtime&& other) {
	std::swap(handle, other.handle);
	return *this;
}

Runtime create_runtime() {
	return Runtime(JS_NewRuntime());
}

}