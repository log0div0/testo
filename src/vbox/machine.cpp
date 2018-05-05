
#include "machine.hpp"
#include <stdexcept>
#include "error.hpp"
#include "safe_array.hpp"

namespace vbox {

Machine::Machine(IMachine* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Machine::~Machine() {
	if (handle) {
		IMachine_Release(handle);
	}
}

Machine::Machine(Machine&& other): handle(other.handle) {
	other.handle = nullptr;
}

Machine& Machine::operator=(Machine&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string Machine::name() const {
	try {
		BSTR name = nullptr;
		HRESULT rc = IMachine_get_Name(handle, &name);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return StringOut(name);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Machine::save_settings() {
	try {
		HRESULT rc = IMachine_SaveSettings(handle);
		if (FAILED(rc)) {
			throw Error(rc);
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<StorageController> Machine::storage_controllers() const {
	try {
		SafeArray safe_array;
		HRESULT rc = IMachine_get_StorageControllers(handle, ComSafeArrayAsOutIfaceParam(safe_array.handle, IStorageController*));
		if (FAILED(rc)) {
			throw Error(rc);
		}
		ArrayOut array_out;
		rc = api->pfnSafeArrayCopyOutIfaceParamHelper((IUnknown***)&array_out.values, &array_out.values_count, safe_array.handle);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		std::vector<StorageController> result;
		for (ULONG i = 0; i < array_out.values_count; ++i) {
			result.push_back(StorageController(((IStorageController**)array_out.values)[i]));
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
