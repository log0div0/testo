
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
		std::string name = computerSystem.get("Name");
		auto videoHead = services.execQuery("SELECT * FROM Msvm_VideoHead WHERE SystemName=\"" + name + "\"").getOne();
		auto virtualSystemSettingData = services.execQuery("SELECT * FROM Msvm_VirtualSystemSettingData WHERE InstanceID=\"Microsoft:" + name + "\"").getOne();
		return Display(std::move(videoHead), std::move(virtualSystemSettingData), services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
