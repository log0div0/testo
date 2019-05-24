
#include "StorageController.hpp"

namespace hyperv {

StorageController::StorageController(wmi::WbemClassObject resourceAllocationSettingData_,
	wmi::WbemClassObject virtualSystemSettingData_,
	wmi::WbemServices services_):
	resourceAllocationSettingData(std::move(resourceAllocationSettingData_)),
	virtualSystemSettingData(std::move(virtualSystemSettingData_)),
	services(std::move(services_))
{

}

Drive StorageController::addDiskDrive(size_t slot) {
	return addDrive(slot, "Microsoft:Hyper-V:Synthetic Disk Drive");
}

Drive StorageController::addDVDDrive(size_t slot) {
	return addDrive(slot, "Microsoft:Hyper-V:Synthetic DVD Drive");
}

Drive StorageController::addDrive(size_t slot, const std::string& subtype) {
	auto driveTemplate = services.getResourceTemplate("Msvm_ResourceAllocationSettingData", subtype);
	driveTemplate.put("Parent", resourceAllocationSettingData.path());
	driveTemplate.put("AddressOnParent", std::to_string(slot));
	auto drive = services.addResource(virtualSystemSettingData, driveTemplate);
	return Drive(drive, virtualSystemSettingData, services);
}

}
