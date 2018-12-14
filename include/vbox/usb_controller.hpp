
#pragma once

#include "api.hpp"
#include "string.hpp"
#include <ostream>

namespace vbox {

struct USBController {
	USBController(IUSBController* handle);
	~USBController();

	USBController(const USBController&) = delete;
	USBController& operator=(const USBController&) = delete;

	USBController(USBController&& other);
	USBController& operator=(USBController&& other);

	std::string name() const;
	USBControllerType controller_type() const;
	void controller_type(USBControllerType type);

	IUSBController* handle = nullptr;
};

std::ostream& operator<<(std::ostream& stream, const USBController& storage_controller);

}
