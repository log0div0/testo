
#include "Controller.hpp"

#include <fmt/format.h>

bool Controller::is_defined() const {
	return fs::exists(main_file());
}

bool Controller::has_snapshot(const std::string& snapshot) {
	fs::path metadata_file = get_metadata_dir();
	metadata_file /= id() + "_" + snapshot;
	return fs::exists(metadata_file);
}

std::string Controller::get_snapshot_cksum(const std::string& snapshot) {
	try {
		fs::path metadata_file = get_metadata_dir();
		metadata_file /= id() + "_" + snapshot;
		auto metadata = read_metadata_file(metadata_file);
		if (!metadata.count("cksum")) {
			throw std::runtime_error("Can't find cksum field in snapshot metadata " + snapshot);
		}

		return metadata.at("cksum").get<std::string>();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("getting snapshot cksum error"));
	}
}

bool Controller::has_key(const std::string& key) {
	try {
		auto metadata = read_metadata_file(main_file());
		return metadata.count(key);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking metadata with key {}", key)));
	}
}


std::string Controller::get_metadata(const std::string& key) const {
	try {
		return ::get_metadata(main_file(), key);

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Getting vm metadata with key {}", key)));
	}
}

void Controller::set_metadata(const std::string& key, const std::string& value) {
	try {
		auto metadata = read_metadata_file(main_file());
		metadata[key] = value;
		write_metadata_file(main_file(), metadata);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting metadata with key {}", key)));
	}
}
