
#include "NIC.hpp"
#include <algorithm>

namespace hyperv {

NIC::NIC(wmi::WbemClassObject ethernetPortSettingData_,
	wmi::WbemClassObject virtualSystemSettingData_,
	wmi::WbemServices services_):
	ethernetPortSettingData(std::move(ethernetPortSettingData_)),
	virtualSystemSettingData(std::move(virtualSystemSettingData_)),
	services(std::move(services_))
{

}

void NIC::setMAC(std::string mac) {
	try {
		mac.erase(std::remove(mac.begin(), mac.end(), ':'), mac.end());
		ethernetPortSettingData.put("Address", mac);
		ethernetPortSettingData.put("StaticMacAddress", true);
		ethernetPortSettingData = services.modifyResource(ethernetPortSettingData);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Link NIC::connect(const Bridge& bridge) {
	auto linkTemplate = services.getResourceTemplate("Msvm_EthernetPortAllocationSettingData", "Microsoft:Hyper-V:Ethernet Connection");
	linkTemplate.put("HostResource", std::vector<std::string>{bridge.virtualEthernetSwitch.path()});
	linkTemplate.put("Parent", ethernetPortSettingData.path());
	services.addResource(virtualSystemSettingData, linkTemplate);
	return {};
}

}
