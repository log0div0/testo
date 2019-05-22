
#include "VmController.hpp"

VmController::VmController(const nlohmann::json& config_): config(config_) {

}

nlohmann::json VmController::get_config() const {
	return config;
}

std::string VmController::name() const {
	return config.at("name");
}
