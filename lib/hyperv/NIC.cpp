
#include "NIC.hpp"
#include "ResourceTemplate.hpp"
#include <wmi/Call.hpp>
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
		auto result = services.call("Msvm_VirtualSystemManagementService", "ModifyResourceSettings")
			.with("ResourceSettings", std::vector<wmi::WbemClassObject>{ethernetPortSettingData})
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Link NIC::connect(const Bridge& bridge) {
	auto linkTemplate = ResourceTemplate(services, "Msvm_EthernetPortAllocationSettingData", "Microsoft:Hyper-V:Ethernet Connection");
	linkTemplate.put("HostResource", std::vector<std::string>{bridge.virtualEthernetSwitch.path()});
	linkTemplate.put("Parent", ethernetPortSettingData.path());
	linkTemplate.addTo(virtualSystemSettingData);
	return {};
}

}
