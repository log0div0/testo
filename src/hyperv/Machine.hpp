
#pragma once

#include "wmi.hpp"

namespace hyperv {

struct Machine {
	Machine(wmi::WbemClassObject computerSystem_, wmi::WbemServices services_);

	std::string name() const;
	bool is_running() const;
	std::vector<uint8_t> screenshot() const;
	size_t screenWidth() const;
	size_t screenHeight() const;

	wmi::WbemClassObject computerSystem, videoHead, virtualSystemSettingData, virtualSystemManagementService;
	wmi::WbemServices services;
};

}
