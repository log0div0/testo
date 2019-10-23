
#include "Network.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

Network::Network(const nlohmann::json& config_): config(config_) {
}

nlohmann::json Network::get_config() const {
	return config;
}

std::string Network::id() const {
	return config.at("prefix").get<std::string>() + config.at("name").get<std::string>();
}

std::string Network::name() const {
	return config.at("name");
}