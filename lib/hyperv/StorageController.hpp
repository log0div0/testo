
#pragma once

#include "Drive.hpp"

namespace hyperv {

struct StorageController {
	StorageController(wmi::WbemClassObject resourceAllocationSettingData_,
		wmi::WbemClassObject virtualSystemSettingData_,
		wmi::WbemServices services_);

	Drive addDiskDrive(size_t slot);
	Drive addDVDDrive(size_t slot);
	Drive addDrive(size_t slot, const std::string& subtype);

	std::vector<Drive> drives() const;

	wmi::WbemClassObject resourceAllocationSettingData;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
