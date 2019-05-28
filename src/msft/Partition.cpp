
#include "Partition.hpp"
#include <regex>

namespace msft {

Partition::Partition(wmi::WbemClassObject partition_, wmi::WbemServices services_):
	partition(std::move(partition_)), services(std::move(services_))
{

}

void Partition::deleteObject() {
	try {
		services.call("Msft_Partition", "DeleteObject")
			.exec(partition);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<std::string> Partition::getAccessPaths() const {
	try {
		auto result = services.call("Msft_Partition", "GetAccessPaths")
			.exec(partition);
		return result.get("AccessPaths");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<std::string> Partition::accessPaths() const {
	try {
		return partition.get("AccessPaths");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Volume Partition::volume() const {
	try {
		for (auto& path: accessPaths()) {
			auto escaped_path = std::regex_replace(path, std::regex("\\\\"), "\\\\");
			auto objects = services.execQuery("SELECT * FROM Msft_Volume WHERE Path=\"" + escaped_path + "\"").getAll();
			if (objects.size()) {
				return Volume(std::move(objects.at(0)), services);
			}
		}
		throw std::runtime_error("Failed to find volume");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Partition::addAccessPath(const std::string& path) {
	try {
		auto result = services.call("Msft_Partition", "AddAccessPath")
			.with("AccessPath", path)
			.exec(partition);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Partition::removeAccessPath(const std::string& path) {
	try {
		auto result = services.call("Msft_Partition", "RemoveAccessPath")
			.with("AccessPath", path)
			.exec(partition);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
