
#include "SyntheticMouse.hpp"
#include <wmi/Call.hpp>

namespace hyperv {

SyntheticMouse::SyntheticMouse(wmi::WbemClassObject mouse_, wmi::WbemClassObject virtualSystemSettingData_, wmi::WbemServices services_):
	mouse(std::move(mouse_)), virtualSystemSettingData(std::move(virtualSystemSettingData_)), services(std::move(services_))
{
}

void SyntheticMouse::set_absolute_position(int32_t x, int32_t y) {
	try {
		services.call("Msvm_SyntheticMouse", "SetAbsolutePosition")
			.with("horizontalPosition", x)
			.with("verticalPosition", y)
			.exec(mouse);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
