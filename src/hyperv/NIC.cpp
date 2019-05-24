
#include "NIC.hpp"
#include <algorithm>

namespace hyperv {

NIC::NIC(wmi::WbemClassObject syntheticEthernetPortSettingData_,
	wmi::WbemClassObject virtualSystemSettingData_,
	wmi::WbemServices services_):
	syntheticEthernetPortSettingData(std::move(syntheticEthernetPortSettingData_)),
	virtualSystemSettingData(std::move(virtualSystemSettingData_)),
	services(std::move(services_))
{

}

void NIC::setMAC(std::string mac) {
	try {
		mac.erase(std::remove(mac.begin(), mac.end(), ':'), mac.end());
		syntheticEthernetPortSettingData.put("Address", mac);
		syntheticEthernetPortSettingData.put("StaticMacAddress", true);
		syntheticEthernetPortSettingData = services.modifyResource(syntheticEthernetPortSettingData);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
