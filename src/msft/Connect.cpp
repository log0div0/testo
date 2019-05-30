
#include "Connect.hpp"
#include <iostream>
#include <regex>

namespace msft {

Connect::Connect() {
	try {
		services = locator.connectServer("root\\microsoft\\windows\\storage");
		services.setProxyBlanket();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<Disk> Connect::disks() const {
	try {
		std::vector<Disk> result;
		auto objects = services.execQuery("SELECT * FROM Msft_Disk").getAll();
		for (auto& object: objects) {
			result.push_back(Disk(std::move(object), services));
		}
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Disk Connect::virtualDisk(const std::string& location) const {
	try {
		auto escaped_location = std::regex_replace(location, std::regex("\\\\"), "\\\\");
		escaped_location = std::regex_replace(escaped_location, std::regex("/"), "\\\\");
		auto object = services.execQuery("SELECT * FROM Msft_Disk WHERE Location=\"" + escaped_location + "\"").getOne();
		return Disk(std::move(object), services);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
