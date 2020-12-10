
#pragma once

#include "Bridge.hpp"

namespace hyperv {

struct Link {

};

struct NIC {
	NIC(wmi::WbemClassObject ethernetPortSettingData_,
		wmi::WbemClassObject virtualSystemSettingData_,
		wmi::WbemServices services_);

	void setMAC(std::string mac);
	Link connect(const Bridge& bridge);

	wmi::WbemClassObject ethernetPortSettingData;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
