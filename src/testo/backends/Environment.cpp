
#include "../NNServiceClient.hpp"
#include "Environment.hpp"

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