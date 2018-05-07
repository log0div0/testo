
#include "storage_controller.hpp"
#include <stdexcept>
#include "throw_if_failed.hpp"

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
		throw_if_failed(IStorageController_get_Name(handle, &name));
		return StringOut(name);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

StorageBus StorageController::bus() const {
	try {
		StorageBus result = StorageBus_Null;
		throw_if_failed(IStorageController_get_Bus(handle, IF_UNIX((uint32_t*))&result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

StorageControllerType StorageController::controller_type() const {
	try {
		StorageControllerType result = StorageControllerType_Null;
		throw_if_failed(IStorageController_get_ControllerType(handle, IF_UNIX((uint32_t*))&result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

ULONG StorageController::port_count() const {
	try {
		ULONG result = 0;
		throw_if_failed(IStorageController_get_PortCount(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void StorageController::port_count(ULONG n) {
	try {
		throw_if_failed(IStorageController_put_PortCount(handle, n));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool StorageController::host_io_cache() const {
	try {
		BOOL result = 0;
		throw_if_failed(IStorageController_get_UseHostIOCache(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool StorageController::bootable() const {
	try {
		BOOL result = 0;
		throw_if_failed(IStorageController_get_Bootable(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::ostream& operator<<(std::ostream& stream, const StorageController& storage_controller) {
	stream << std::boolalpha;
	stream << storage_controller.name()
		<< " bus=" << storage_controller.bus()
		<< " controller_type=" << storage_controller.controller_type()
		<< " port_count=" << storage_controller.port_count()
		<< " host_io_cache=" << storage_controller.host_io_cache()
		<< " bootable=" << storage_controller.bootable();
	stream << std::noboolalpha;
	return stream;
}

}
