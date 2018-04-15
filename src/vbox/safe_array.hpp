
#pragma once

#include "error.hpp"
#include "array_out.hpp"
#include <vector>

#define SAFE_ARRAY_AS_OUT_IFACE_PARAM(safe_array) ComSafeArrayAsOutIfaceParam(safe_array.handle, decltype(safe_array.iface_param()))

namespace vbox {

template <typename Param>
struct SafeArray {
	typename Param::Iface* iface_param() const;

	SafeArray() {
#ifndef _MSC_VER
		handle = api->pfnSafeArrayOutParamAlloc();
		if (!handle) {
			throw std::runtime_error(__PRETTY_FUNCTION__);
		}
#endif
	}

	~SafeArray() {
		if (handle) {
			api->pfnSafeArrayDestroy(handle);
		}
	}

	SafeArray(const SafeArray&) = delete;
	SafeArray& operator=(const SafeArray&) = delete;

	SafeArray(SafeArray&& other);
	SafeArray& operator=(SafeArray&& other);

	std::vector<Param> copy_out_iface_param() const {
		try {
			ArrayOut<typename Param::Iface*> array_out;
			HRESULT rc = api->pfnSafeArrayCopyOutIfaceParamHelper((IUnknown***)&array_out.values, &array_out.values_count, handle);
			if (FAILED(rc)) {
				throw Error(rc);
			}
			std::vector<Param> result;
			for (ULONG i = 0; i < array_out.size(); ++i) {
				result.emplace_back(array_out[i]);
			}
			return result;
		}
		catch (const std::exception&) {
			std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
		}
	}

	SAFEARRAY* handle = nullptr;
};

}
