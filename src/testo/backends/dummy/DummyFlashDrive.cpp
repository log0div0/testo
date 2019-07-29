
#include "DummyFlashDrive.hpp"
#include <fstream>

DummyFlashDrive::DummyFlashDrive(const nlohmann::json& config_): FlashDrive(config_) {
	std::cout << "DummyFlashDrive " << config.dump(4) << std::endl;
}

DummyFlashDrive::~DummyFlashDrive() {

}

bool DummyFlashDrive::is_defined() {
	return fs::exists(metadata_file());
}

void DummyFlashDrive::create() {
	//remove the file if it exists

	if (fs::exists(metadata_file())) {
		fs::remove(metadata_file());
	}

	nlohmann::json config;
	config["state"] = "stopped";
	config["snapshots"] = nlohmann::json::array();

	write_metadata_file(metadata_file(), config);
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

void DummyFlashDrive::write_metadata_file(const fs::path& file, const nlohmann::json& metadata) {
	std::ofstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't write metadata file " + file.generic_string());
	}

	metadata_file_stream << metadata;
	metadata_file_stream.close();
}

nlohmann::json DummyFlashDrive::read_metadata_file(const fs::path& file) const {
	std::ifstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't read metadata file " + file.generic_string());
	}

	nlohmann::json result = nlohmann::json::parse(metadata_file_stream);
	metadata_file_stream.close();
	return result;
}

