
#pragma once

#include "Bridge.hpp"

namespace hyperv {

struct Link {

};

struct NIC {
	NIC(wmi::WbemClassObject ethernetPortSettingData_,
		wmi::WbemClassObject virtualSystemSettingData_,
		wmi::WbemServices services_);

	std::string name() const;
	void setMAC(std::string mac);
	Link connect(const Bridge& bridge);
	bool is_connected() const;
	void disconnect();
	void destroy();

	wmi::WbemClassObject ethernetPortSettingData;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
