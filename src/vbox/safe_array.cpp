
#include "safe_array.hpp"

namespace vbox {

SafeArray::SafeArray() {
#ifndef _MSC_VER
	handle = api->pfnSafeArrayOutParamAlloc();
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
#endif
}

SafeArray::SafeArray(VARTYPE vt, size_t size) {
	handle = api->pfnSafeArrayCreateVector(vt, 0, size);
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

SafeArray::~SafeArray() {
	if (handle) {
		api->pfnSafeArrayDestroy(handle);
	}
}

}
