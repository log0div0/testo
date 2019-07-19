
#include "Controller.hpp"

#include <fmt/format.h>
#include <fstream>

bool Controller::has_snapshot(const std::string& snapshot) {
	fs::path metadata_file = get_metadata_dir();
	metadata_file /= name() + "_" + snapshot;
	return fs::exists(metadata_file);
}

std::string Controller::get_snapshot_cksum(const std::string& snapshot) {
	try {
		fs::path metadata_file = get_metadata_dir();
		metadata_file /= name() + "_" + snapshot;
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


std::string Controller::get_metadata(const std::string& key) {
	try {
		auto metadata = read_metadata_file(main_file());
		if (!metadata.count(key)) {
			throw std::runtime_error("Requested key is not present in vm metadata");
		}
		return metadata.at(key).get<std::string>();

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Getting metadata with key {}", key)));
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

void Controller::write_metadata_file(const fs::path& file, const nlohmann::json& metadata) {
	std::ofstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't write metadata file " + file.generic_string());
	}

	metadata_file_stream << metadata;
	metadata_file_stream.close();
}

nlohmann::json Controller::read_metadata_file(const fs::path& file) const {
	std::ifstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't read metadata file " + file.generic_string());
	}

	nlohmann::json result = nlohmann::json::parse(metadata_file_stream);
	metadata_file_stream.close();
	return result;
}
