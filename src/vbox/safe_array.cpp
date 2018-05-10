
#include "safe_array.hpp"
#include "throw_if_failed.hpp"

namespace vbox {

SafeArray::SafeArray() {
#ifndef _MSC_VER
	handle = api->pfnSafeArrayOutParamAlloc();
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
#endif
}

SafeArray::SafeArray(VARTYPE vt, ULONG size) {
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

SafeArray::SafeArray(SafeArray&& other): handle(other.handle) {
	other.handle = nullptr;
}

SafeArray& SafeArray::operator=(SafeArray&& other) {
	std::swap(handle, other.handle);
	return *this;
}

void SafeArray::copy_in(void* data, ULONG size) {
	try {
		if (!data || !size) {
			return;
		}
		throw_if_failed(api->pfnSafeArrayCopyInParamHelper(handle, data, size));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

ArrayOut SafeArray::copy_out(VARTYPE vartype) {
	try {
		void* data = nullptr;
		ULONG size = 0;
		throw_if_failed(api->pfnSafeArrayCopyOutParamHelper(&data, &size, vartype, handle));
		return {data, size};
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

ArrayOut SafeArray::copy_out_iface() {
	try {
		IUnknown** data = nullptr;
		ULONG size = 0;
		throw_if_failed(api->pfnSafeArrayCopyOutIfaceParamHelper(&data, &size, handle));
		return {data, size};
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
