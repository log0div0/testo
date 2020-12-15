
#pragma once

#include <wmi/WbemClassObject.hpp>
#include <wmi/WbemServices.hpp>

namespace hyperv {

struct Processor {
	Processor(wmi::WbemClassObject processorSettingData_,
		wmi::WbemClassObject virtualSystemSettingData_,
		wmi::WbemServices services_);

	void setVirtualQuantity(int32_t value);

	wmi::WbemClassObject processorSettingData;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
