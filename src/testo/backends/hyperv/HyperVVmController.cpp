
#include "HyperVVmController.hpp"
#include <iostream>

HyperVVmController::HyperVVmController(const nlohmann::json& config_): VmController(config_) {
	std::cout << "HyperVVmController " << config.dump(4) << std::endl;
}

HyperVVmController::~HyperVVmController() {

}

void HyperVVmController::install() {
	try {
		for (auto& machine: connect.machines()) {
			if (machine.name() == name()) {
				if (machine.is_running()) {
					machine.stop();
				}
				machine.destroy();
			}
		}

		auto machine = connect.defineMachine(name());

		nlohmann::json metadata = config.at("metadata");
		metadata["vm_config"] = config.dump(4);
		metadata["vm_nic_count"] = std::to_string(config.count("nic") ? config.at("nic").size() : 0);
		metadata["vm_name"] = config.at("name");
		metadata["dvd_signature"] = "";
		machine.setNotes({metadata.dump(4)});

		machine.addDVDDrive();
		machine.addDiskDrive();

		std::cout << "TODO: " << __FUNCSIG__ << std::endl;
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVmController::make_snapshot(const std::string& snapshot, const std::string& cksum) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::set_metadata(const nlohmann::json& metadata) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::set_metadata(const std::string& key, const std::string& value) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

std::string HyperVVmController::get_metadata(const std::string& key) {
	try {
		auto machine = connect.machine(name());
		auto json = nlohmann::json::parse(machine.notes().at(0));
		return json.at(key);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string HyperVVmController::get_snapshot_cksum(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::rollback(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::press(const std::vector<std::string>& buttons) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVmController::is_nic_plugged(const std::string& nic) const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::set_nic(const std::string& nic, bool is_enabled) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVmController::is_link_plugged(const std::string& nic) const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::set_link(const std::string& nic, bool is_connected) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::plug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::unplug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVmController::is_dvd_plugged() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::plug_dvd(fs::path path) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::unplug_dvd() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void HyperVVmController::start() {
	try {
		connect.machine(name()).start();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVmController::stop() {
	try {
		connect.machine(name()).stop();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVmController::suspend() {
	try {
		connect.machine(name()).pause();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}
void HyperVVmController::resume() {
	try {
		connect.machine(name()).start();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVmController::shutdown(uint32_t timeout_seconds) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::type(const std::string& text) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVmController::wait(const std::string& text, const nlohmann::json& params, const std::string& time) {
	std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
	return false;
}
bool HyperVVmController::check(const std::string& text, const nlohmann::json& params) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
int HyperVVmController::run(const fs::path& exe, std::vector<std::string> args, uint32_t timeout_seconds) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVmController::is_flash_plugged(std::shared_ptr<FlashDriveController> fd) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVmController::has_snapshot(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
std::vector<std::string> HyperVVmController::keys() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVmController::has_key(const std::string& key) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

bool HyperVVmController::is_defined() const {
	try {
		for (auto& machine: connect.machines()) {
			if (machine.name() == name()) {
				return true;
			}
		}
		return false;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

bool HyperVVmController::is_running() {
	try {
		return connect.machine(name()).is_running();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

bool HyperVVmController::is_additions_installed() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::copy_to_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_seconds) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::copy_from_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_seconds) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::remove_from_guest(const fs::path& obj) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
std::set<std::string> HyperVVmController::nics() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
