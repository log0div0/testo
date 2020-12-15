
#include "Drive.hpp"
#include "ResourceTemplate.hpp"
#include <regex>

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

Disk Drive::mountHDD(const std::string& path) {
	return mount(path, "Microsoft:Hyper-V:Virtual Hard Disk");
}

Disk Drive::mount(const std::string& path_, const std::string& subtype) {
	try {
		std::string path = std::regex_replace(path_, std::regex("/"), "\\\\");
		ResourceTemplate isoTemplate(services, "Msvm_StorageAllocationSettingData", subtype);
		isoTemplate.put("HostResource", std::vector<std::string>{path});
		isoTemplate.put("Parent", resourceAllocationSettingData.path());
		auto disk = isoTemplate.addTo(virtualSystemSettingData);
		return Disk(std::move(disk), virtualSystemSettingData, services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<Disk> Drive::disks() const {
	try {
		auto escaped_path = std::regex_replace(resourceAllocationSettingData.path(), std::regex("\\\\"), "\\\\");
		auto escaped_path2 = std::regex_replace(escaped_path, std::regex("\""), "\\\"");
		std::vector<Disk> result;
		auto objects = services.execQuery(
				"SELECT * FROM Msvm_StorageAllocationSettingData "
				"WHERE Parent=\"" + escaped_path2 + "\""
			).getAll();
		for (auto& object: objects) {
			result.push_back(Disk(std::move(object), virtualSystemSettingData, services));
		}
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
