
#include "Memory.hpp"
#include <wmi/Call.hpp>

namespace hyperv {

Memory::Memory(wmi::WbemClassObject memorySettingData_,
	wmi::WbemClassObject virtualSystemSettingData_,
	wmi::WbemServices services_):
	memorySettingData(std::move(memorySettingData_)),
	virtualSystemSettingData(std::move(virtualSystemSettingData_)),
	services(std::move(services_))
{

}

void Memory::setVirtualQuantity(int32_t value) {
	try {
		memorySettingData.put("VirtualQuantity", value);
		auto result = services.call("Msvm_VirtualSystemManagementService", "ModifyResourceSettings")
			.with("ResourceSettings", std::vector<wmi::WbemClassObject>{memorySettingData})
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
