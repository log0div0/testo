
#include "NIC.hpp"
#include "ResourceTemplate.hpp"
#include <wmi/Call.hpp>
#include <algorithm>
#include <iostream>

namespace hyperv {

NIC::NIC(wmi::WbemClassObject ethernetPortSettingData_,
	wmi::WbemClassObject virtualSystemSettingData_,
	wmi::WbemServices services_):
	ethernetPortSettingData(std::move(ethernetPortSettingData_)),
	virtualSystemSettingData(std::move(virtualSystemSettingData_)),
	services(std::move(services_))
{

}

std::string NIC::name() const {
	try {
		return ethernetPortSettingData.get("ElementName");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
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
	try {
		auto linkTemplate = ResourceTemplate(services, "Msvm_EthernetPortAllocationSettingData", "Microsoft:Hyper-V:Ethernet Connection");
		linkTemplate.put("HostResource", std::vector<std::string>{bridge.virtualEthernetSwitch.path()});
		linkTemplate.put("Parent", ethernetPortSettingData.path());
		linkTemplate.addTo(virtualSystemSettingData);
		return {};
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string escape(const std::string& in) {
	std::string out;
	for (char ch: in) {
		switch (ch) {
			case '\\':
				out.push_back('\\');
				out.push_back('\\');
				break;
			case '"':
				out.push_back('\\');
				out.push_back('"');
				break;
			default:
				out.push_back(ch);
				break;
		}
	}
	return out;
}

bool NIC::is_connected() const {
	try {
		std::string query =
			"SELECT * FROM Msvm_EthernetPortAllocationSettingData "
			"WHERE Parent=\"" + escape(ethernetPortSettingData.path()) + "\"";
		auto objects = services.execQuery(query).getAll();
		return objects.size();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void NIC::disconnect() {
	try {
		std::string query =
			"SELECT * FROM Msvm_EthernetPortAllocationSettingData "
			"WHERE Parent=\"" + escape(ethernetPortSettingData.path()) + "\"";
		auto object = services.execQuery(query).getOne();
		auto result = services.call("Msvm_VirtualSystemManagementService", "RemoveResourceSettings")
			.with("ResourceSettings", std::vector<std::string>{object.path()})
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void NIC::destroy() {
	try {
		auto result = services.call("Msvm_VirtualSystemManagementService", "RemoveResourceSettings")
			.with("ResourceSettings", std::vector<std::string>{ethernetPortSettingData.path()})
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
