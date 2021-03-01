
#include "FlashDrive.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

#ifdef __linux__
#include <unistd.h>
#endif

FlashDrive::FlashDrive(const nlohmann::json& config_): config(config_) {

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
