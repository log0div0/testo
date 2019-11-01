
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

std::set<std::string> VM::nics() const {
	std::set<std::string> result;

	if (config.count("nic")) {
		for (auto& nic: config.at("nic")) {
			result.insert(nic.at("name").get<std::string>());
		}
	}
	return result;
}

std::set<std::string> VM::networks() const {
	std::set<std::string> result;

	if (config.count("nic")) {
		auto nics = config.at("nic");
		for (auto& nic: nics) {
			std::string source_network = nic.at("attached_to").get<std::string>();
			result.insert(source_network);
		}
	}

	return result;
}

