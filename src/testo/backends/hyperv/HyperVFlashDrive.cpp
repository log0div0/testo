
#include "HyperVFlashDrive.hpp"

HyperVFlashDrive::HyperVFlashDrive(const nlohmann::json& config_): FlashDrive(config_) {
	std::cout << "HyperVFlashDrive " << config.dump(4) << std::endl;
}

HyperVFlashDrive::~HyperVFlashDrive() {

}

void HyperVFlashDrive::create() {
	std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
}
bool HyperVFlashDrive::is_mounted() const {
	std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
	return false;
}
void HyperVFlashDrive::mount() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVFlashDrive::umount() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
fs::path HyperVFlashDrive::img_path() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
fs::path HyperVFlashDrive::mount_dir() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
