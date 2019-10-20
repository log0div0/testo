
#include "VM.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

VM::VM(const nlohmann::json& config_): config(config_) {

}

nlohmann::json VM::get_config() const {
	return config;
}

std::string VM::id() const {
	return config.at("prefix").get<std::string>() + config.at("name").get<std::string>();
}

std::string VM::name() const {
	return config.at("name");
}
