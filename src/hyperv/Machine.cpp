
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
		auto videoHead = services.execQuery("ASSOCIATORS OF {" + computerSystem.relpath() + "} WHERE ResultClass=Msvm_VideoHead").getOne();
		auto virtualSystemSettingData = services.execQuery("ASSOCIATORS OF {" + computerSystem.relpath() + "} WHERE ResultClass=Msvm_VirtualSystemSettingData").getOne();
		return Display(std::move(videoHead), std::move(virtualSystemSettingData), services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
