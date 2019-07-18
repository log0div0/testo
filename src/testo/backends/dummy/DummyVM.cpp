
#include "DummyVM.hpp"
#include <iostream>

DummyVM::DummyVM(const nlohmann::json& config_): VM(config_) {
}

DummyVM::~DummyVM() {

}

void DummyVM::install() {
	//remove the file if it exists

	if (fs::exists(metadata_file())) {
		fs::remove(metadata_file());
	}

	nlohmann::json config;
	config["state"] = "stopped";
	config["snapshots"] = nlohmann::json::array();

	write_metadata_file(metadata_file(), config);
}

void DummyVM::make_snapshot(const std::string& snapshot) {
	auto config = read_metadata_file(metadata_file());
	nlohmann::json new_snapshot = {
		{"name", snapshot},
		{"state", config.at("state").get<std::string>()}
	};
	config["snapshots"].push_back(new_snapshot);
	write_metadata_file(metadata_file(), config);
}

void DummyVM::rollback(const std::string& snapshot) {
	auto config = read_metadata_file(metadata_file());
	for (auto snap: config["snapshots"]) {
		if (snap["name"] == snapshot) {
			config["state"] = snap.at("state").get<std::string>();
			write_metadata_file(metadata_file(), config);
			return;
		}
	}
	throw std::runtime_error(std::string("Trying to restore non-existent snapshot: ") + snapshot);
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
	auto config = read_metadata_file(metadata_file());
	config["state"] = "running";
	write_metadata_file(metadata_file(), config);
}

void DummyVM::stop() {
	auto config = read_metadata_file(metadata_file());
	config["state"] = "stopped";
	write_metadata_file(metadata_file(), config);
}

void DummyVM::suspend() {
	auto config = read_metadata_file(metadata_file());

	auto current_state = config.at("state").get<std::string>();
	if (current_state != "running") {
		throw std::runtime_error(std::string("Can't suspend vm not in running state, current state is ") + current_state);
	}

	config["state"] = "suspended";
	write_metadata_file(metadata_file(), config);
}
void DummyVM::resume() {
	auto config = read_metadata_file(metadata_file());
	auto current_state = config.at("state").get<std::string>();
	if (current_state != "suspended") {
		throw std::runtime_error(std::string("Can't suspend vm not in running state, current state is ") + current_state);
	}
	write_metadata_file(metadata_file(), config);
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
	auto config = read_metadata_file(metadata_file());
	for (auto snap: config["snapshots"]) {
		if (snap["name"] == snapshot) {
			return true;
		}
	}
	return false;
}
void DummyVM::delete_snapshot(const std::string& snapshot) {
	auto config = read_metadata_file(metadata_file());

	auto snapshots = config.at("snapshots");
	for (auto it = snapshots.begin(); it != snapshots.end(); ++it) {
		if (it.value()["name"] == snapshot) {
			snapshots.erase(it);
			config["snapshots"] = snapshots;
			write_metadata_file(metadata_file(), config);
			return;
		}
	}
	throw std::runtime_error(std::string("Trying to delete non-existent snapshot: ") + snapshot);
}

bool DummyVM::is_defined() const {
	return fs::exists(metadata_file());
}

VmState DummyVM::state() const {
	auto config = read_metadata_file(metadata_file());

	std::string state = config.at("state").get<std::string>();

	if (state == "stopped") {
		return VmState::Stopped;
	} else if (state == "running") {
		return VmState::Running;
	} else if (state == "suspended") {
		return VmState::Suspended;
	} else if (state == "other") {
		return VmState::Other;
	} else {
		throw std::runtime_error(std::string("Unknown vm state: ") + state);
	}
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

void DummyVM::write_metadata_file(const fs::path& file, const nlohmann::json& metadata) {
	std::ofstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't write metadata file " + file.generic_string());
	}

	metadata_file_stream << metadata;
	metadata_file_stream.close();
}

nlohmann::json DummyVM::read_metadata_file(const fs::path& file) const {
	std::ifstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't read metadata file " + file.generic_string());
	}

	nlohmann::json result = nlohmann::json::parse(metadata_file_stream);
	metadata_file_stream.close();
	return result;
}


