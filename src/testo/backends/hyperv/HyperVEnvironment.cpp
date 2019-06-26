
#include "HyperVEnvironment.hpp"

fs::path HyperVEnvironment::testo_dir() const {
	throw std::runtime_error("Implement me");
}

fs::path HyperVEnvironment::flash_drives_mount_dir() const {
	throw std::runtime_error("Implement me");
}

fs::path HyperVEnvironment::flash_drives_img_dir() const {
	throw std::runtime_error("Implement me");
}

fs::path HyperVEnvironment::metadata_dir() const {
	return ".";
}

void HyperVEnvironment::setup() {
	_putenv_s("HYPERV", "1");
}

void HyperVEnvironment::cleanup() {

}
