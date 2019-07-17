
#include "DummyFlashDrive.hpp"

DummyFlashDrive::DummyFlashDrive(const nlohmann::json& config_): FlashDrive(config_) {
	std::cout << "DummyFlashDrive " << config.dump(4) << std::endl;
}

DummyFlashDrive::~DummyFlashDrive() {

}

bool DummyFlashDrive::is_defined() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void DummyFlashDrive::create() {
	std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
}
bool DummyFlashDrive::is_mounted() const {
	std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
	return false;
}
void DummyFlashDrive::mount() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyFlashDrive::umount() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
fs::path DummyFlashDrive::img_path() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool DummyFlashDrive::has_snapshot(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyFlashDrive::make_snapshot(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyFlashDrive::delete_snapshot(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyFlashDrive::rollback(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}