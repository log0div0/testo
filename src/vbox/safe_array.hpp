
#pragma once

#include "error.hpp"
#include "array_out.hpp"
#include <vector>

namespace vbox {

struct SafeArray {
	SafeArray();
	~SafeArray();

	SafeArray(const SafeArray&) = delete;
	SafeArray& operator=(const SafeArray&) = delete;

	SafeArray(SafeArray&& other);
	SafeArray& operator=(SafeArray&& other);

	template <typename X, typename Y>
	std::vector<Y> copy_out_iface_param() const {
		try {
			ArrayOut<X> array_out;
			HRESULT rc = api->pfnSafeArrayCopyOutIfaceParamHelper((IUnknown***)&array_out.values, &array_out.values_count, handle);
			if (FAILED(rc)) {
				throw Error(rc);
			}
			std::vector<Y> result;
			for (ULONG i = 0; i < array_out.values_count; ++i) {
				result.push_back(Y(array_out.values[i]));
			}
			return result;
		}
		catch (const std::exception&) {
			std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
		}
	}

	template <typename X, typename Y>
	std::vector<Y> copy_out_param(VARTYPE vt) const {
		try {
			ArrayOut<X> array_out;
			HRESULT rc = api->pfnSafeArrayCopyOutParamHelper((void**)&array_out.values, &array_out.values_count, vt, handle);
			if (FAILED(rc)) {
				throw Error(rc);
			}
			std::vector<Y> result;
			for (ULONG i = 0; i < array_out.values_count / sizeof(X); ++i) {
				result.push_back(Y(array_out.values[i]));
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
