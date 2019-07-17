
#include "DummyVM.hpp"
#include <iostream>

DummyVM::DummyVM(const nlohmann::json& config_): VM(config_) {
	std::cout << "DummyVM " << config.dump(4) << std::endl;

}

DummyVM::~DummyVM() {

}

void DummyVM::install() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void DummyVM::make_snapshot(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void DummyVM::rollback(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void DummyVM::press(const std::vector<std::string>& buttons) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

bool DummyVM::is_nic_plugged(const std::string& nic) const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyVM::set_nic(const std::string& nic, bool is_enabled) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool DummyVM::is_link_plugged(const std::string& nic) const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyVM::set_link(const std::string& nic, bool is_connected) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyVM::plug_flash_drive(std::shared_ptr<FlashDrive> fd) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyVM::unplug_flash_drive(std::shared_ptr<FlashDrive> fd) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool DummyVM::is_dvd_plugged() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyVM::plug_dvd(fs::path path) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyVM::unplug_dvd() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void DummyVM::start() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void DummyVM::stop() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void DummyVM::suspend() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyVM::resume() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void DummyVM::power_button() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

stb::Image DummyVM::screenshot() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

int DummyVM::run(const fs::path& exe, std::vector<std::string> args, uint32_t timeout_seconds) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool DummyVM::is_flash_plugged(std::shared_ptr<FlashDrive> fd) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool DummyVM::has_snapshot(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyVM::delete_snapshot(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

bool DummyVM::is_defined() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

VmState DummyVM::state() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

bool DummyVM::is_additions_installed() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyVM::copy_to_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_seconds) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyVM::copy_from_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_seconds) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void DummyVM::remove_from_guest(const fs::path& obj) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
std::set<std::string> DummyVM::nics() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
