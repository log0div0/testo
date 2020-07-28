
#include "FlashDrive.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

#ifdef __linux__
#include <unistd.h>
#endif

FlashDrive::FlashDrive(const nlohmann::json& config_): config(config_) {
	if (!config.count("name")) {
		throw std::runtime_error("Constructing FlashDriveController error: field NAME is not specified");
	}

	if (!config.count("size")) {
		throw std::runtime_error("Constructing FlashDriveController error: field SIZE is not specified");
	}

	//TODO: check for fs types
	if (!config.count("fs")) {
		throw std::runtime_error("Constructing FlashDriveController error: field FS is not specified");
	}

	auto fs = config.at("fs").get<std::string>();
	/*if (fs != "ntfs" &&
		fs != "vfat")
	{
		throw std::runtime_error(std::string("Constructing FlashDriveController error: unsupported filesystem: ") + fs);
	}*/
}

std::string FlashDrive::id() const {
	return config.at("prefix").get<std::string>() + config.at("name").get<std::string>();
}

std::string FlashDrive::name() const {
	return config.at("name").get<std::string>();
}

std::string FlashDrive::prefix() const {
	return config.at("prefix").get<std::string>();
}
