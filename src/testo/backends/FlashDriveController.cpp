
#include "FlashDriveController.hpp"

FlashDriveController::FlashDriveController(const nlohmann::json& config_): config(config_) {

}

std::string FlashDriveController::name() const {
	return config.at("name").get<std::string>();
}

bool FlashDriveController::has_folder() const {
	return config.count("folder");
}

nlohmann::json FlashDriveController::get_config() const {
	return config;
}

bool FlashDriveController::cache_enabled() const {
	return config.value("cache_enabled", 1);
}
