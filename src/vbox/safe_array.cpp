
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

ArrayOut SafeArray::copy_out(VARTYPE vartype) const {
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

ArrayOut SafeArray::copy_out() const {
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

SafeArray SafeArray::bitset(int bitset) {
	std::vector<int> bits;
	for (size_t i = 0; i < sizeof(int) * 8; ++i) {
		int bit = 1 << i;
		if (bitset & bit) {
			bits.push_back(bit);
		}
	}
	SafeArray safe_array(VT_I4, (ULONG)bits.size());
	safe_array.copy_in(bits.data(), (ULONG)(bits.size() * sizeof(int)));
	return safe_array;
}

int SafeArray::bitset() const {
	ArrayOut array_out = copy_out(VT_I4);
    int bitset = 0;
    for (ULONG i = 0; i < array_out.values_count / sizeof(int); ++i) {
      bitset |= ((int*)array_out.values)[i];
    }
    return bitset;
}

}
