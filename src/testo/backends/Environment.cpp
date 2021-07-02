
#include "Environment.hpp"

void Environment::setup() {
	if (!fs::exists(vm_metadata_dir())) {
		if (!fs::create_directories(vm_metadata_dir())) {
			throw std::runtime_error(std::string("Can't create directory: ") + vm_metadata_dir().generic_string());
		}
	}

	if (!fs::exists(network_metadata_dir())) {
		if (!fs::create_directories(network_metadata_dir())) {
			throw std::runtime_error(std::string("Can't create directory: ") + network_metadata_dir().generic_string());
		}
	}

	if (!fs::exists(flash_drives_metadata_dir())) {
		if (!fs::create_directories(flash_drives_metadata_dir())) {
			throw std::runtime_error(std::string("Can't create directory: ") + flash_drives_metadata_dir().generic_string());
		}
	}

	nn_client = NNServiceClient("127.0.0.1", "8888");
}
