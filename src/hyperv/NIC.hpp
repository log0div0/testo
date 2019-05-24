
#pragma once

#include "wmi.hpp"

namespace hyperv {

struct NIC {
	NIC(wmi::WbemClassObject syntheticEthernetPortSettingData_,
		wmi::WbemClassObject virtualSystemSettingData_,
		wmi::WbemServices services_);

	void setMAC(std::string mac);

	wmi::WbemClassObject syntheticEthernetPortSettingData;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
