
#include "QemuVmController.hpp"
#include "QemuFlashDriveController.hpp"

#include "Utils.hpp"

QemuVmController::QemuVmController(const nlohmann::json& config): config(config) {

}


int QemuVmController::set_metadata(const nlohmann::json& metadata) {
	return 0;
}

int QemuVmController::set_metadata(const std::string& key, const std::string& value) {
	return 0;
}

std::vector<std::string> QemuVmController::keys() {
	return {};
}

bool QemuVmController::has_key(const std::string& key) {
	return true;
}

std::string QemuVmController::get_metadata(const std::string& key) {
	return "";
}

int QemuVmController::install() {
	return 0;
}

int QemuVmController::make_snapshot(const std::string& snapshot) {
	return 0;
}

std::set<std::string> QemuVmController::nics() const {
	return {};
}

int QemuVmController::set_snapshot_cksum(const std::string& snapshot, const std::string& cksum) {
	return 0;
}

std::string QemuVmController::get_snapshot_cksum(const std::string& snapshot) {
	return "";
}

int QemuVmController::rollback(const std::string& snapshot) {
	return 0;
}

int QemuVmController::press(const std::vector<std::string>& buttons) {
	return 0;
}

int QemuVmController::set_nic(const std::string& nic, bool is_enabled) {
	return 0;
}

int QemuVmController::set_link(const std::string& nic, bool is_connected) {
	return 0;
}

bool QemuVmController::is_plugged(std::shared_ptr<FlashDriveController> fd) {
	return true;
}

int QemuVmController::plug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	return 0;
}

int QemuVmController::unplug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	return 0;
}

void QemuVmController::unplug_all_flash_drives() {

}

int QemuVmController::plug_dvd(fs::path path) {
	return 0;
}

int QemuVmController::unplug_dvd() {
	return 0;
}

int QemuVmController::start() {
	return 0;
}

int QemuVmController::stop() {
	return 0;
}

int QemuVmController::type(const std::string& text) {
	return 0;
}

int QemuVmController::wait(const std::string& text, const std::string& time) {
	return 0;
}

int QemuVmController::run(const fs::path& exe, std::vector<std::string> args) {
	return 0;
}

bool QemuVmController::has_snapshot(const std::string& snapshot) {
	return 0;
}

bool QemuVmController::is_defined() const {
	return true;
}

bool QemuVmController::is_running() {
	return true;
}

bool QemuVmController::is_additions_installed() {
	return true;
}


int QemuVmController::copy_to_guest(const fs::path& src, const fs::path& dst) {
	return 0;
}

int QemuVmController::remove_from_guest(const fs::path& obj) {
	return 0;
}
