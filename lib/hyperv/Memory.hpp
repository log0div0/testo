
#pragma once

#include <wmi/WbemClassObject.hpp>
#include <wmi/WbemServices.hpp>

namespace hyperv {

struct Memory {
	Memory(wmi::WbemClassObject memorySettingData_,
		wmi::WbemClassObject virtualSystemSettingData_,
		wmi::WbemServices services_);

	void setVirtualQuantity(int32_t value);

	wmi::WbemClassObject memorySettingData;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
