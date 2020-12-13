
#include "Keyboard.hpp"
#include <wmi/Call.hpp>

namespace hyperv {

Keyboard::Keyboard(wmi::WbemClassObject keyboard_, wmi::WbemClassObject virtualSystemSettingData_, wmi::WbemServices services_):
	keyboard(std::move(keyboard_)), virtualSystemSettingData(std::move(virtualSystemSettingData_)), services(std::move(services_))
{
}

void Keyboard::typeScancodes(const std::vector<uint8_t>& codes) {
	try {
		services.call("Msvm_Keyboard", "TypeScancodes")
			.with("Scancodes", codes)
			.exec(keyboard);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
