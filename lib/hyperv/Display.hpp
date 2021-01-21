
#pragma once

#include <wmi/WbemClassObject.hpp>
#include <wmi/WbemServices.hpp>

namespace hyperv {

struct Display {
	enum class State: uint16_t {
		Enabled = 2,
		Disabled = 3,
	};

	Display(wmi::WbemClassObject videoHead_, wmi::WbemClassObject virtualSystemSettingData_, wmi::WbemServices services_);

	std::vector<uint8_t> screenshot() const;
	size_t width() const;
	size_t height() const;
	State state() const;

	wmi::WbemClassObject videoHead;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
