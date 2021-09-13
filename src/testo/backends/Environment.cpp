
#include "../NNServiceClient.hpp"
#include "Environment.hpp"

std::string EnvironmentConfig::nn_service_ip() const {
	auto semicolon_pos = nn_service_endpoint.find(":");
	return nn_service_endpoint.substr(0, semicolon_pos);
}

std::string EnvironmentConfig::nn_service_port() const {
	auto semicolon_pos = nn_service_endpoint.find(":");
	return nn_service_endpoint.substr(semicolon_pos + 1, nn_service_endpoint.length() - 1);
}

void EnvironmentConfig::validate() const {
	auto semicolon_pos = nn_service_endpoint.find(":");
	if (semicolon_pos == std::string::npos) {
		throw std::runtime_error("ip_port string is malformed: no semicolon");
	}

	auto port = nn_service_port();

	try {
		auto uport = std::stoul(port);
		if (uport > 65535) {
			throw std::runtime_error("");
		}
	} catch (const std::exception& error) {
		throw std::runtime_error(std::string("nn_service port doesn't seem to be valid: ") + port);
	}
}

void EnvironmentConfig::dump(nlohmann::json& j) const {
	j["nn_service_ip"] = nn_service_ip();
	j["nn_service_port"] = nn_service_port();
}

void Environment::setup(const EnvironmentConfig& config) {
	if (!fs::exists(vm_metadata_dir())) {
		if (!fs::create_directories(vm_metadata_dir())) {
			throw std::runtime_error("Can't create directory: " + vm_metadata_dir().generic_string());
		}
	}

	if (!fs::exists(network_metadata_dir())) {
		if (!fs::create_directories(network_metadata_dir())) {
			throw std::runtime_error("Can't create directory: " + network_metadata_dir().generic_string());
		}
	}

	if (!fs::exists(flash_drives_metadata_dir())) {
		if (!fs::create_directories(flash_drives_metadata_dir())) {
			throw std::runtime_error("Can't create directory: " + flash_drives_metadata_dir().generic_string());
		}
	}

	nn_client = std::make_unique<NNServiceClient>(config.nn_service_ip(), config.nn_service_port());
}

Environment::Environment() {

}

Environment::~Environment() {

}