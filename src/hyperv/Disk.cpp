
#include "Disk.hpp"

namespace hyperv {

Disk::Disk(wmi::WbemClassObject storageAllocationSettingData_,
	wmi::WbemClassObject virtualSystemSettingData_,
	wmi::WbemServices services_):
	storageAllocationSettingData(std::move(storageAllocationSettingData_)),
	virtualSystemSettingData(std::move(virtualSystemSettingData_)),
	services(std::move(services_))
{

}

std::string Disk::hostResource() const {
	try {
		std::vector<std::string> resource = storageAllocationSettingData.get("HostResource");
		return resource.at(0);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
