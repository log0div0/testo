
#include "HyperVFlashDriveController.hpp"

HyperVFlashDriveController::HyperVFlashDriveController(const nlohmann::json& config_): FlashDriveController(config_) {
	std::cout << "HyperVFlashDriveController " << config.dump(4) << std::endl;
}

HyperVFlashDriveController::~HyperVFlashDriveController() {

}

void HyperVFlashDriveController::create() {
	std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
}
bool HyperVFlashDriveController::is_mounted() const {
	std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
	return false;
}
void HyperVFlashDriveController::mount() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVFlashDriveController::umount() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
fs::path HyperVFlashDriveController::img_path() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
fs::path HyperVFlashDriveController::mount_dir() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
