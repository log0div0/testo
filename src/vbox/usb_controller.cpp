
#include "usb_controller.hpp"
#include <stdexcept>
#include "throw_if_failed.hpp"

namespace vbox {

USBController::USBController(IUSBController* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

USBController::~USBController() {
	if (handle) {
		IUSBController_Release(handle);
	}
}

USBController::USBController(USBController&& other): handle(other.handle) {
	other.handle = nullptr;
}

USBController& USBController::operator=(USBController&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string USBController::name() const {
	try {
		BSTR name = nullptr;
		throw_if_failed(IUSBController_get_Name(handle, &name));
		return StringOut(name);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

USBControllerType USBController::controller_type() const {
	try {
		USBControllerType_T result = USBControllerType_Null;
		throw_if_failed(IUSBController_get_Type(handle, &result));
		return (USBControllerType)result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void USBController::controller_type(USBControllerType type) {
	try {
		throw_if_failed(IUSBController_put_Type(handle, type));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}


std::ostream& operator<<(std::ostream& stream, const USBController& storage_controller) {
	stream << std::boolalpha;
	stream << storage_controller.name()
		<< " controller_type=" << storage_controller.controller_type();
	stream << std::noboolalpha;
	return stream;
}

}
