
#include "Machine.hpp"
#include "ResourceTemplate.hpp"
#include <wmi/Call.hpp>
#include <iostream>

namespace hyperv {

Machine::Machine(wmi::WbemClassObject computerSystem_,
	wmi::WbemServices services_):
	computerSystem(std::move(computerSystem_)),
	services(std::move(services_))
{
	try {
		virtualSystemSettingData = activeSettings();
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

std::string Machine::guid() const {
	try {
		return computerSystem.get("Name");
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

void Machine::enable() {
	requestStateChange(State::Enabled);
}

void Machine::disable() {
	requestStateChange(State::Disabled);
}

std::vector<StorageController> Machine::scsiControllers() const {
	return controllers("Microsoft:Hyper-V:Synthetic SCSI Controller");
}

std::vector<StorageController> Machine::ideControllers() const {
	return controllers("Microsoft:Hyper-V:Emulated IDE Controller");
}

StorageController Machine::addSCSIController() const {
	return addController("Microsoft:Hyper-V:Synthetic SCSI Controller");
}

StorageController Machine::addIDEController() const {
	return addController("Microsoft:Hyper-V:Emulated IDE Controller");
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

StorageController Machine::addController(const std::string& subtype) const {
	try {
		auto controllerTemplate = ResourceTemplate(services, "Msvm_ResourceAllocationSettingData", subtype);
		auto controller = controllerTemplate.addTo(virtualSystemSettingData);
		return StorageController(controller, virtualSystemSettingData, services);
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

SyntheticMouse Machine::synthetic_mouse() const {
	try {
		auto synthetic_mouse = services.execQuery(
			"SELECT * FROM Msvm_SyntheticMouse "
			"WHERE SystemName=\"" + computerSystem.get("Name").get<std::string>() + "\""
		).getOne();
		return SyntheticMouse(std::move(synthetic_mouse), virtualSystemSettingData, services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Processor Machine::processor() const {
	try {
		auto processor = services.execQuery(
			"SELECT * FROM Msvm_ProcessorSettingData "
			"WHERE InstanceID LIKE \"" + virtualSystemSettingData.get("InstanceID").get<std::string>() + "%\""
		).getOne();
		return Processor(std::move(processor), virtualSystemSettingData, services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Memory Machine::memory() const {
	try {
		auto memory = services.execQuery(
			"SELECT * FROM Msvm_MemorySettingData "
			"WHERE InstanceID LIKE \"" + virtualSystemSettingData.get("InstanceID").get<std::string>() + "%\""
		).getOne();
		return Memory(std::move(memory), virtualSystemSettingData, services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

NIC Machine::addNIC(const std::string& name, bool legacy) {
	try {
		auto nicTemplate = legacy ?
			ResourceTemplate(services, "Msvm_EmulatedEthernetPortSettingData", "Microsoft:Hyper-V:Emulated Ethernet Port") :
			ResourceTemplate(services, "Msvm_SyntheticEthernetPortSettingData", "Microsoft:Hyper-V:Synthetic Ethernet Port");
		nicTemplate.put("ElementName", name);
		auto nic = nicTemplate.addTo(virtualSystemSettingData);
		return NIC(nic, virtualSystemSettingData, services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Snapshot Machine::createSnapshot() {
	try {
		services.call("Msvm_VirtualSystemSnapshotService", "CreateSnapshot")
			.with("AffectedSystem", computerSystem.path())
			.with("SnapshotType", 2)
			.exec();
		virtualSystemSettingData = activeSettings();
		auto snapshot = services.getObject(virtualSystemSettingData.get("Parent"));
		return Snapshot(snapshot, services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Snapshot Machine::snapshot(const std::string& name) {
	try {
		auto snapshot = services.execQuery(
			"SELECT * FROM Msvm_VirtualSystemSettingData "
			"WHERE VirtualSystemIdentifier=\"" + computerSystem.get("Name").get<std::string>() + "\" AND "
			"ElementName=\"" + name + "\" AND "
			"Description=\"Checkpoint settings for the virtual machine.\""
		).getOne();
		return Snapshot(snapshot, services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<Snapshot> Machine::snapshots() {
	try {
		auto snapshots = services.execQuery(
			"SELECT * FROM Msvm_VirtualSystemSettingData "
			"WHERE VirtualSystemIdentifier=\"" + computerSystem.get("Name").get<std::string>() + "\" AND "
			"Description=\"Checkpoint settings for the virtual machine.\""
		).getAll();
		std::vector<Snapshot> result;
		for (auto& snapshot: snapshots) {
			result.push_back(Snapshot(snapshot, services));
		}
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

wmi::WbemClassObject Machine::activeSettings() {
	try {
		// return services.execQuery(
		// 	"SELECT * FROM Msvm_VirtualSystemSettingData "
		// 	"WHERE Description=\"Active settings for the virtual machine.\" AND "
		// 	"ElementName=\"" + this->name() + "\""
		// ).getOne();
		return services.getObject("Msvm_VirtualSystemSettingData.InstanceID=\"Microsoft:" + computerSystem.get("Name").get<std::string>() + "\"");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
