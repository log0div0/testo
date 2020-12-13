
#include "Disk.hpp"
#include <wmi/Call.hpp>
#include <regex>

namespace msft {

Disk::Disk(wmi::WbemClassObject disk_, wmi::WbemServices services_):
	disk(std::move(disk_)), services(std::move(services_))
{

}

std::string Disk::friendlyName() const {
	return disk.get("FriendlyName");
}

void Disk::initialize(uint16_t partitionStyle) {
	try {
		services.call("Msft_Disk", "Initialize")
			.with("PartitionStyle", (int32_t)partitionStyle)
			.exec(disk);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Disk::clear() {
	try {
		services.call("Msft_Disk", "Clear")
			.exec(disk);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Disk::createPartition() {
	try {
		auto result = services.call("Msft_Disk", "CreatePartition")
			.with("UseMaximumSize", true)
			.exec(disk);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<Partition> Disk::partitions() const {
	try {
		std::string path = disk.get("Path");
		auto escaped_path = std::regex_replace(path, std::regex("\\\\"), "\\\\");
		std::vector<Partition> result;
		auto objects = services.execQuery("SELECT * FROM Msft_Partition WHERE DiskId=\"" + escaped_path + "\"").getAll();
		for (auto& object: objects) {
			result.push_back(Partition(std::move(object), services));
		}
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
