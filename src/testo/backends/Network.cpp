
#include "Environment.hpp"
#include "Network.hpp"
#include <fmt/format.h>

Network::Network(const nlohmann::json& config_): config(config_) {

}

std::string Network::id() const {
	return config.at("prefix").get<std::string>() + config.at("name").get<std::string>();
}

std::string Network::name() const {
	return config.at("name");
}

std::string Network::prefix() const {
	return config.at("prefix").get<std::string>();
}