
#include "Machine.hpp"

namespace hyperv {

Machine::Machine(wmi::WbemClassObject computerSystem_,
	wmi::WbemServices services_):
	computerSystem(std::move(computerSystem_)),
	services(std::move(services_))
{
	try {
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

bool Machine::is_running() const {
	try {
		return computerSystem.get("EnabledState").get<int32_t>() == 2;
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
		return Display(std::move(videoHead), services);
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
			.with("SystemSettings", settings().put("Notes", notes))
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<std::string> Machine::notes() const {
	return settings().get("Notes");
}

void Machine::requestStateChange(uint16_t requestedState) {
	try {
		services.call("Msvm_ComputerSystem", "RequestStateChange")
			.with("RequestedState", (int32_t)requestedState)
			.exec(computerSystem);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Machine::start() {
	requestStateChange(2);
}

void Machine::stop() {
	requestStateChange(3);
}

void Machine::pause() {
	requestStateChange(32776);
}

wmi::WbemClassObject Machine::settings() const {
	return services.getObject("Msvm_VirtualSystemSettingData.InstanceID=\"Microsoft:" + computerSystem.get("Name").get<std::string>() + "\"");
}

}
