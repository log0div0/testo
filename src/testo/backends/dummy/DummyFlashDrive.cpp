
#include "DummyFlashDrive.hpp"
#include <fstream>

DummyFlashDrive::DummyFlashDrive(const nlohmann::json& config_): FlashDrive(config_) {
}

DummyFlashDrive::~DummyFlashDrive() {

}

bool DummyFlashDrive::is_defined() {
	return fs::exists(metadata_file());
}

void DummyFlashDrive::create() {
	//remove the file if it exists
	if(is_defined()) {
		undefine();
	}

	nlohmann::json config;
	config["snapshots"] = nlohmann::json::array();

	write_metadata_file(metadata_file(), config);
}

void DummyFlashDrive::undefine() {
	//remove the file if it exists

	if (fs::exists(metadata_file())) {
		fs::remove(metadata_file());
	}

}

bool DummyFlashDrive::is_mounted() const {
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
	auto config = read_metadata_file(metadata_file());
	for (auto snap: config["snapshots"]) {
		if (snap["name"] == snapshot) {
			return true;
		}
	}
	return false;
}
void DummyFlashDrive::make_snapshot(const std::string& snapshot) {
	auto config = read_metadata_file(metadata_file());
	nlohmann::json new_snapshot = {
		{"name", snapshot},
	};
	config["snapshots"].push_back(new_snapshot);
	write_metadata_file(metadata_file(), config);
}
void DummyFlashDrive::delete_snapshot(const std::string& snapshot) {
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
void DummyFlashDrive::rollback(const std::string& snapshot) {
	auto config = read_metadata_file(metadata_file());
	for (auto snap: config["snapshots"]) {
		if (snap["name"] == snapshot) {
			write_metadata_file(metadata_file(), config);
			return;
		}
	}
	throw std::runtime_error(std::string("Trying to restore non-existent snapshot: ") + snapshot);
}
