
#pragma once

#include <VBoxCAPIGlue/VBoxCAPIGlue.h>
#include "error.hpp"
#include "array_out.hpp"
#include <vector>

#define SAFE_ARRAY_AS_OUT_IFACE_PARAM(safe_array) ComSafeArrayAsOutIfaceParam((safe_array).handle, decltype(safe_array.iface_param()))

namespace vbox {

template <typename Param>
struct SafeArray {
	typename Param::Iface* iface_param() const;

	SafeArray() {
		handle = g_pVBoxFuncs->pfnSafeArrayOutParamAlloc();
		if (!handle) {
			throw std::runtime_error(__PRETTY_FUNCTION__);
		}
	}

	~SafeArray() {
		if (handle) {
			g_pVBoxFuncs->pfnSafeArrayDestroy(handle);
		}
	}

	SafeArray(const SafeArray&) = delete;
	SafeArray& operator=(const SafeArray&) = delete;

	SafeArray(SafeArray&& other);
	SafeArray& operator=(SafeArray&& other);

	std::vector<Param> copy_out_iface_param() const {
		try {
			ArrayOut<typename Param::Iface*> array_out;
			HRESULT rc = g_pVBoxFuncs->pfnSafeArrayCopyOutIfaceParamHelper((IUnknown***)&array_out.values, &array_out.values_count, handle);
			if (FAILED(rc)) {
				throw Error(rc);
			}
			std::vector<Param> result;
			for (ULONG i = 0; i < array_out.size(); ++i) {
				result.emplace_back(array_out[i]);
			}
			return result;
		}
		catch (const std::exception& error) {
			std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
		}
	}

	SAFEARRAY* handle = nullptr;
};

}
