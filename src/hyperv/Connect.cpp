
#include "Connect.hpp"

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
			.put("ElementName", name);
		services.call("Msvm_VirtualSystemManagementService", "DefineSystem")
			.with("SystemSettings", virtualSystemSettingData)
			.exec();
		return machine(name);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
