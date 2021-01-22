
#pragma once

#include <wmi/WbemClassObject.hpp>
#include <wmi/WbemServices.hpp>

namespace hyperv {

struct SyntheticMouse {
	SyntheticMouse(wmi::WbemClassObject mouse_, wmi::WbemClassObject virtualSystemSettingData_, wmi::WbemServices services_);

	void set_absolute_position(int32_t x, int32_t y);
	void set_button_state(int32_t buttonIndex, bool isDown);

	wmi::WbemClassObject mouse;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
