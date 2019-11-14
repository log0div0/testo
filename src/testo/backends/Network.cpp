
#include "Network.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

Network::Network(const nlohmann::json& config_): config(config_) {
	if (!config.count("mode")) {
		throw std::runtime_error("Constructing NetworkController error: field MODE is not specified");
	}

	auto mode = config.at("mode").get<std::string>();

	if ((mode != "nat") &&
		(mode != "internal"))
	{
		throw std::runtime_error(std::string("Constructing NetworkController error: Unsupported MODE: ") + mode);
	}

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

std::string Network::prefix() const {
	return config.at("prefix").get<std::string>();
}