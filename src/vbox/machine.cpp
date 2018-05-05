
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
		ArrayOut array_out = safe_array.copy_out();
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

std::vector<Medium> Machine::unregister(CleanupMode cleanup_mode) {
	try {
		SafeArray safe_array;
		HRESULT rc = IMachine_Unregister(handle, cleanup_mode, ComSafeArrayAsOutIfaceParam(safe_array.handle, IMedium*));
		if (FAILED(rc)) {
			throw Error(rc);
		}
		ArrayOut array_out = safe_array.copy_out();
		std::vector<Medium> result;
		for (ULONG i = 0; i < array_out.values_count; ++i) {
			result.push_back(Medium(((IMedium**)array_out.values)[i]));
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Progress Machine::delete_config(std::vector<Medium> mediums) {
	try {
		SafeArray safe_array(VT_UNKNOWN, mediums.size());
		safe_array.copy_in(mediums.data(), mediums.size());

		IProgress* result = nullptr;
		HRESULT rc = IMachine_DeleteConfig(handle,
			ComSafeArrayAsInParam(safe_array.handle, IMedium*),
			&result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::ostream& operator<<(std::ostream& stream, const Machine& machine) {
	stream << machine.name() << std::endl;
	stream << "Storage Controllers:" << std::endl;
	for (auto& storage_controller: machine.storage_controllers()) {
		stream << storage_controller << std::endl;
	}
	return stream;
}

}
