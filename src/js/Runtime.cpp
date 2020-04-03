
#include "Runtime.hpp"
#include <stdexcept>

namespace js {

RuntimeRef::RuntimeRef(::JSRuntime* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Runtime& Runtime::instance() {
	static Runtime instance;
	return instance;
}

Runtime::Runtime(): RuntimeRef(JS_NewRuntime()) {
}

Runtime::~Runtime() {
	JS_FreeRuntime(handle);
}

}
