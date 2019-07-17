
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
	return ".";
}

fs::path DummyEnvironment::flash_drives_metadata_dir() const {
	return ".";
}

void DummyEnvironment::setup() {

}

void DummyEnvironment::cleanup() {

}
