
#include "Machine.hpp"
#include <iostream>

namespace hyperv {

Machine::Machine(wmi::WbemClassObject computerSystem_,
	wmi::WbemServices services_):
	computerSystem(std::move(computerSystem_)),
	services(std::move(services_))
{
	try {
		virtualSystemSettingData = services.getObject("Msvm_VirtualSystemSettingData.InstanceID=\"Microsoft:" + computerSystem.get("Name").get<std::string>() + "\"");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string Machine::name() const {
	try {
		return computerSystem.get("ElementName");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Machine::State Machine::state() const {
	try {
		return (State)computerSystem.get("EnabledState").get<int32_t>();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Display Machine::display() const {
	try {
		auto videoHead = services.execQuery(
			"SELECT * FROM Msvm_VideoHead WHERE SystemName=\"" +
			computerSystem.get("Name").get<std::string>() +
			"\" AND EnabledState=2").getOne();
		return Display(std::move(videoHead), virtualSystemSettingData, services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Machine::destroy() {
	try {
		services.call("Msvm_VirtualSystemManagementService", "DestroySystem")
			.with("AffectedSystem", computerSystem.path())
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Machine::setNotes(const std::vector<std::string>& notes) {
	try {
		services.call("Msvm_VirtualSystemManagementService", "ModifySystemSettings")
			.with("SystemSettings", virtualSystemSettingData.put("Notes", notes))
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<std::string> Machine::notes() const {
	return virtualSystemSettingData.get("Notes");
}

void Machine::requestStateChange(State requestedState) {
	try {
		services.call("Msvm_ComputerSystem", "RequestStateChange")
			.with("RequestedState", (int32_t)requestedState)
			.exec(computerSystem);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Machine::start() {
	requestStateChange(State::Enabled);
}

void Machine::stop() {
	requestStateChange(State::Disabled);
}

void Machine::pause() {
	requestStateChange(State::Paused);
}

std::vector<StorageController> Machine::ideControllers() const {
	return controllers("Microsoft:Hyper-V:Emulated IDE Controller");
}

std::vector<StorageController> Machine::controllers(const std::string& subtype) const {
	try {
		std::vector<StorageController> result;
		auto objects = services.execQuery(
				"SELECT * FROM Msvm_ResourceAllocationSettingData "
				"WHERE InstanceID LIKE \"" + virtualSystemSettingData.get("InstanceID").get<std::string>() + "%\" "
				"AND ResourceSubType=\"" + subtype + "\""
			).getAll();
		for (auto& object: objects) {
			result.push_back(StorageController(std::move(object), virtualSystemSettingData, services));
		}
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Keyboard Machine::keyboard() const {
	try {
		auto keyboard = services.execQuery(
			"SELECT * FROM Msvm_Keyboard "
			"WHERE SystemName=\"" + computerSystem.get("Name").get<std::string>() + "\""
		).getOne();
		return Keyboard(std::move(keyboard), virtualSystemSettingData, services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

NIC Machine::addNIC(const std::string& name, bool legacy) {
	auto nicTemplate = legacy ?
		services.getResourceTemplate("Msvm_EmulatedEthernetPortSettingData", "Microsoft:Hyper-V:Emulated Ethernet Port") :
		services.getResourceTemplate("Msvm_SyntheticEthernetPortSettingData", "Microsoft:Hyper-V:Synthetic Ethernet Port");
	nicTemplate.put("ElementName", name);
	auto nic = services.addResource(virtualSystemSettingData, nicTemplate);
	return NIC(nic, virtualSystemSettingData, services);
}

}
