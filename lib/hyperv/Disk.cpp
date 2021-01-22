
#include "Disk.hpp"
#include <wmi/Call.hpp>

namespace hyperv {

Disk::Disk(wmi::WbemClassObject storageAllocationSettingData_,
	wmi::WbemClassObject virtualSystemSettingData_,
	wmi::WbemServices services_):
	storageAllocationSettingData(std::move(storageAllocationSettingData_)),
	virtualSystemSettingData(std::move(virtualSystemSettingData_)),
	services(std::move(services_))
{

}

void Disk::umount() {
	try {
		auto result = services.call("Msvm_VirtualSystemManagementService", "RemoveResourceSettings")
			.with("ResourceSettings", std::vector<std::string>{storageAllocationSettingData.path()})
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
