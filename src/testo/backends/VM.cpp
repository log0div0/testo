
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

std::string VM::get_snapshot_cksum(const std::string& snapshot) {
	try {
		/*fs::path metadata_file = env->metadata_dir() / (name() + "_" + snapshot);
		auto metadata = read_metadata_file(metadata_file);
		if (!metadata.count("cksum")) {
			throw std::runtime_error("Can't find cksum field in snapshot metadata " + snapshot);
		}

		return metadata.at("cksum").get<std::string>();*/
		return "";
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("getting snapshot cksum error"));
	}
}

bool VM::has_key(const std::string& key) {
	try {
		/*fs::path metadata_file = env->metadata_dir() / name();
		auto metadata = read_metadata_file(metadata_file);
		return metadata.count(key);*/
		return true;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking metadata with key {}", key)));
	}
}
