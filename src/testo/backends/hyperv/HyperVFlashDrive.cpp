
#include "HyperVFlashDrive.hpp"

HyperVFlashDrive::HyperVFlashDrive(const nlohmann::json& config_): FlashDrive(config_) {
	std::cout << "HyperVFlashDrive " << config.dump(4) << std::endl;
}

HyperVFlashDrive::~HyperVFlashDrive() {

}

bool HyperVFlashDrive::is_defined() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void HyperVFlashDrive::create() {
	std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
}
void HyperVFlashDrive::undefine() {
	std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
}
bool HyperVFlashDrive::is_mounted() const {
	std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
	return false;
}
void HyperVFlashDrive::mount() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVFlashDrive::umount() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
fs::path HyperVFlashDrive::img_path() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVFlashDrive::has_snapshot(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVFlashDrive::make_snapshot(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVFlashDrive::delete_snapshot(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVFlashDrive::rollback(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}