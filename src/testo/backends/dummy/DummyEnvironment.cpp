
#include "DummyEnvironment.hpp"

fs::path DummyEnvironment::testo_dir() const {
	throw std::runtime_error("Implement me");
}

fs::path DummyEnvironment::flash_drives_mount_dir() const {
	throw std::runtime_error("Implement me");
}

fs::path DummyEnvironment::flash_drives_img_dir() const {
	throw std::runtime_error("Implement me");
}

fs::path DummyEnvironment::vm_metadata_dir() const {
	return "./vm_metadata";
}

fs::path DummyEnvironment::flash_drives_metadata_dir() const {
	return "./flash_drives_metadata";
}

void DummyEnvironment::setup() {
	if (!fs::exists(vm_metadata_dir())) {
		if (!fs::create_directories(vm_metadata_dir())) {
			throw std::runtime_error(std::string("Can't create directory: ") + vm_metadata_dir().generic_string());
		}
	}

	if (!fs::exists(flash_drives_metadata_dir())) {
		if (!fs::create_directories(flash_drives_metadata_dir())) {
			throw std::runtime_error(std::string("Can't create directory: ") + flash_drives_metadata_dir().generic_string());
		}
	}
}

void DummyEnvironment::cleanup() {

}