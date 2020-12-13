
#include "Connect.hpp"
#include <wmi/Call.hpp>

namespace hyperv {

Connect::Connect() {
	try {
		services = locator.connectServer("root\\virtualization\\v2");
		services.setProxyBlanket();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<Machine> Connect::machines() const {
	try {
		std::vector<Machine> result;
		auto objects = services.execQuery("SELECT * FROM Msvm_ComputerSystem WHERE Caption=\"Virtual Machine\"").getAll();
		for (auto& object: objects) {
			result.push_back(Machine(std::move(object), services));
		}
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Machine Connect::machine(const std::string& name) const {
	try {
		auto object = services.execQuery("SELECT * FROM Msvm_ComputerSystem WHERE Caption=\"Virtual Machine\" AND ElementName=\"" + name + "\"").getOne();
		return Machine(std::move(object), services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Machine Connect::defineMachine(const std::string& name) {
	try {
		auto virtualSystemSettingData = services.getObject("Msvm_VirtualSystemSettingData").spawnInstance()
			.put("ElementName", name)
			// .put("VirtualSystemSubType", "Microsoft:Hyper-V:SubType:2")
			.put("SecureBootEnabled", false);
		auto result = services.call("Msvm_VirtualSystemManagementService", "DefineSystem")
			.with("SystemSettings", virtualSystemSettingData)
			.exec();
		return Machine(services.getObject(result.get("ResultingSystem")), services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<Bridge> Connect::bridges() const {
	try {
		std::vector<Bridge> result;
		auto objects = services.execQuery("SELECT * FROM Msvm_VirtualEthernetSwitch").getAll();
		for (auto& object: objects) {
			result.push_back(Bridge(std::move(object), services));
		}
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Bridge Connect::bridge(const std::string& name) const {
	try {
		auto object = services.execQuery("SELECT * FROM Msvm_VirtualEthernetSwitch WHERE ElementName=\"" + name + "\"").getOne();
		return Bridge(std::move(object), services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Bridge Connect::defineBridge(const std::string& name) {
	try {
		auto virtualSystemSettingData = services.getObject("Msvm_VirtualEthernetSwitchSettingData").spawnInstance()
			.put("ElementName", name);
		auto result = services.call("Msvm_VirtualEthernetSwitchManagementService", "DefineSystem")
			.with("SystemSettings", virtualSystemSettingData)
			.exec();
		return Bridge(services.getObject(result.get("ResultingSystem")), services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string Connect::defaultVirtualHardDiskPath() const {
	try {
		auto virtualSystemManagementServiceSettingData = services.execQuery("SELECT * FROM Msvm_VirtualSystemManagementServiceSettingData").getOne();
		return virtualSystemManagementServiceSettingData.get("DefaultVirtualHardDiskPath");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Connect::createHDD(const std::string& path, size_t size) {
	try {
		auto virtualHardDiskSettingData = services.getObject("Msvm_VirtualHardDiskSettingData").spawnInstance()
			.put("Path", path)
			.put("Type", (int32_t)3)
			.put("Format", (int32_t)2)
			.put("MaxInternalSize", std::to_string(size));
		auto result = services.call("Msvm_ImageManagementService", "CreateVirtualHardDisk")
			.with("VirtualDiskSettingData", virtualHardDiskSettingData)
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
