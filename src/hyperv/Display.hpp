
#pragma once

#include "wmi.hpp"

namespace hyperv {

struct Display {
	Display(wmi::WbemClassObject videoHead_, wmi::WbemClassObject virtualSystemSettingData_, wmi::WbemServices services_);

	std::vector<uint8_t> screenshot() const;
	size_t width() const;
	size_t height() const;

	wmi::WbemClassObject videoHead;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
