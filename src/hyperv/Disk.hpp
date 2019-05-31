
#pragma once

#include "wmi.hpp"

namespace hyperv {

struct Disk {
	Disk(wmi::WbemClassObject storageAllocationSettingData_,
		wmi::WbemClassObject virtualSystemSettingData_,
		wmi::WbemServices services_);

	std::string hostResource() const;

	wmi::WbemClassObject storageAllocationSettingData;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
