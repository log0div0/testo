
#include "VM.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

VM::VM(const nlohmann::json& config_): config(config_) {

}

nlohmann::json VM::get_config() const {
	return config;
}

std::string VM::name() const {
	return config.at("name");
}

std::string VM::get_metadata(const std::string& key) {
	try {
		fs::path metadata_file = env->metadata_dir() / name();
		auto metadata = read_metadata_file(metadata_file);
		if (!metadata.count(key)) {
			throw std::runtime_error("Requested key is not present in vm metadata");
		}
		return metadata.at(key).get<std::string>();

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Getting metadata with key {}", key)));
	}
}

void VM::set_metadata(const std::string& key, const std::string& value) {
	try {
		fs::path metadata_file = env->metadata_dir() / name();
		auto metadata = read_metadata_file(metadata_file);
		metadata[key] = value;
		write_metadata_file(metadata_file, metadata);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting metadata with key {}", key)));
	}
}

std::string VM::get_snapshot_cksum(const std::string& snapshot) {
	try {
		fs::path metadata_file = env->metadata_dir() / (name() + "_" + snapshot);
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

bool VM::has_key(const std::string& key) {
	try {
		fs::path metadata_file = env->metadata_dir() / name();
		auto metadata = read_metadata_file(metadata_file);
		return metadata.count(key);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking metadata with key {}", key)));
	}
}

void VM::write_metadata_file(const fs::path& file, const nlohmann::json& metadata) {
	std::ofstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't write metadata file " + file.generic_string());
	}

	metadata_file_stream << metadata;
	metadata_file_stream.close();
}

nlohmann::json VM::read_metadata_file(const fs::path& file) const {
	std::ifstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't read metadata file " + file.generic_string());
	}

	nlohmann::json result;
	metadata_file_stream >> result;
	metadata_file_stream.close();
	return result;
}
