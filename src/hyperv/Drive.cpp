
#include "Drive.hpp"

namespace hyperv {

Drive::Drive(wmi::WbemClassObject resourceAllocationSettingData_,
	wmi::WbemClassObject virtualSystemSettingData_,
	wmi::WbemServices services_):
	resourceAllocationSettingData(std::move(resourceAllocationSettingData_)),
	virtualSystemSettingData(std::move(virtualSystemSettingData_)),
	services(std::move(services_))
{

}

Disk Drive::mountISO(const std::string& path) {
	return mount(path, "Microsoft:Hyper-V:Virtual CD/DVD Disk");
}

Disk Drive::mount(const std::string& path, const std::string& subtype) {
	auto isoTemplate = services.getResourceTemplate("Msvm_StorageAllocationSettingData", subtype);
	isoTemplate.put("HostResource", std::vector<std::string>{path});
	isoTemplate.put("Parent", resourceAllocationSettingData.path());
	services.addResource(virtualSystemSettingData, isoTemplate);
	return {};
}

}
