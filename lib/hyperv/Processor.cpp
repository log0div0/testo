
#include "Processor.hpp"
#include <wmi/Call.hpp>

namespace hyperv {

Processor::Processor(wmi::WbemClassObject processorSettingData_,
	wmi::WbemClassObject virtualSystemSettingData_,
	wmi::WbemServices services_):
	processorSettingData(std::move(processorSettingData_)),
	virtualSystemSettingData(std::move(virtualSystemSettingData_)),
	services(std::move(services_))
{

}

void Processor::setVirtualQuantity(int32_t value) {
	try {
		processorSettingData.put("VirtualQuantity", value);
		auto result = services.call("Msvm_VirtualSystemManagementService", "ModifyResourceSettings")
			.with("ResourceSettings", std::vector<wmi::WbemClassObject>{processorSettingData})
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}