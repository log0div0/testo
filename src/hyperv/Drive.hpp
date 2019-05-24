
#pragma once

#include "wmi.hpp"

namespace hyperv {

struct Disk {

};

struct Drive {
	Drive(wmi::WbemClassObject resourceAllocationSettingData_,
		wmi::WbemClassObject virtualSystemSettingData_,
		wmi::WbemServices services_);

	Disk mountISO(const std::string& path);
	Disk mount(const std::string& path, const std::string& subtype);

	wmi::WbemClassObject resourceAllocationSettingData;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
