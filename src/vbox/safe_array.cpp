
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

void SafeArray::copy_in(void* data, size_t size) {
	if (!data || !size) {
		return;
	}
	HRESULT rc = api->pfnSafeArrayCopyInParamHelper(handle, data, size);
	if (FAILED(rc)) {
		throw Error(rc);
	}
}

ArrayOut SafeArray::copy_out(VARTYPE vartype) {
	try {
		void* data = nullptr;
		ULONG size = 0;
		HRESULT rc = api->pfnSafeArrayCopyOutParamHelper(&data, &size, vartype, handle);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return {data, size};
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

ArrayOut SafeArray::copy_out() {
	try {
		IUnknown** data = nullptr;
		ULONG size = 0;
		HRESULT rc = api->pfnSafeArrayCopyOutIfaceParamHelper(&data, &size, handle);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return {data, size};
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
