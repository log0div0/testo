
#include "StorageController.hpp"
#include "ResourceTemplate.hpp"
#include <regex>

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
	auto driveTemplate = ResourceTemplate(services, "Msvm_ResourceAllocationSettingData", subtype);
	driveTemplate.put("Parent", resourceAllocationSettingData.path());
	driveTemplate.put("AddressOnParent", std::to_string(slot));
	auto drive = driveTemplate.addTo(virtualSystemSettingData);
	return Drive(drive, virtualSystemSettingData, services);
}

std::vector<Drive> StorageController::drives() const {
	try {
		auto escaped_path = std::regex_replace(resourceAllocationSettingData.path(), std::regex("\\\\"), "\\\\");
		auto escaped_path2 = std::regex_replace(escaped_path, std::regex("\""), "\\\"");
		std::vector<Drive> result;
		auto objects = services.execQuery(
				"SELECT * FROM Msvm_ResourceAllocationSettingData "
				"WHERE Parent=\"" + escaped_path2 + "\""
			).getAll();
		for (auto& object: objects) {
			result.push_back(Drive(std::move(object), virtualSystemSettingData, services));
		}
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
