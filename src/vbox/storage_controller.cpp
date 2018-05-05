
#include "storage_controller.hpp"
#include <stdexcept>
#include "error.hpp"

namespace vbox {

StorageController::StorageController(IStorageController* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

StorageController::~StorageController() {
	if (handle) {
		IStorageController_Release(handle);
	}
}

StorageController::StorageController(StorageController&& other): handle(other.handle) {
	other.handle = nullptr;
}

StorageController& StorageController::operator=(StorageController&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string StorageController::name() const {
	try {
		BSTR name = nullptr;
		HRESULT rc = IStorageController_get_Name(handle, &name);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return StringOut(name);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

StorageBus StorageController::bus() const {
	try {
		StorageBus result = StorageBus_Null;
		HRESULT rc = IStorageController_get_Bus(handle, &result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

StorageControllerType StorageController::controller_type() const {
	try {
		StorageControllerType result = StorageControllerType_Null;
		HRESULT rc = IStorageController_get_ControllerType(handle, &result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

size_t StorageController::port_count() const {
	try {
		ULONG result = 0;
		HRESULT rc = IStorageController_get_PortCount(handle, &result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool StorageController::host_io_cache() const {
	try {
		BOOL result = 0;
		HRESULT rc = IStorageController_get_UseHostIOCache(handle, &result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool StorageController::bootable() const {
	try {
		BOOL result = 0;
		HRESULT rc = IStorageController_get_Bootable(handle, &result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::ostream& operator<<(std::ostream& stream, const StorageController& storage_controller) {
	return stream << storage_controller.name()
		<< " bus=" << storage_controller.bus()
		<< " controller_type=" << storage_controller.controller_type()
		<< " port_count=" << storage_controller.port_count()
		<< " host_io_cache=" << storage_controller.host_io_cache()
		<< " bootable=" << storage_controller.bootable();
}

}
