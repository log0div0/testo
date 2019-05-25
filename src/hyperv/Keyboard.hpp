
#pragma once

#include "wmi.hpp"

namespace hyperv {

struct Keyboard {
	Keyboard(wmi::WbemClassObject keyboard_, wmi::WbemClassObject virtualSystemSettingData_, wmi::WbemServices services_);

	void typeScancodes(const std::vector<uint8_t>& codes);

	wmi::WbemClassObject keyboard;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
