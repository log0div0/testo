
#pragma once

#include "Disk.hpp"

namespace hyperv {

struct Drive {
	Drive(wmi::WbemClassObject resourceAllocationSettingData_,
		wmi::WbemClassObject virtualSystemSettingData_,
		wmi::WbemServices services_);

	Disk mountISO(const std::string& path);
	Disk mountHDD(const std::string& path);
	Disk mount(const std::string& path, const std::string& subtype);

	std::vector<Disk> disks() const;

	wmi::WbemClassObject resourceAllocationSettingData;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
