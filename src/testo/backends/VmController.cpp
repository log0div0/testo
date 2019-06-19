
#include "VmController.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

void VmController::create_vm() {
	try {
		fs::path metadata_file = env->metadata_dir() / vm->name();

		if (fs::exists(metadata_file)) {
			if (!fs::remove(metadata_file)) {
				throw std::runtime_error("Error deleting metadata file " + metadata_file.generic_string());
			}
		}

		vm->install();

		auto config = vm->get_config();

		nlohmann::json metadata;

		if (config.count("metadata")) {
			auto metadata = config.at("metadata");
			for (auto it = metadata.begin(); it != metadata.end(); ++it) {
				metadata[it.key()] = it.value();
			}
		}

		metadata["vm_config"] = config.dump();
		metadata["vm_nic_count"] = std::to_string(config.count("nic") ? config.at("nic").size() : 0);
		metadata["vm_name"] = config.at("name");
		metadata["dvd_signature"] = file_signature(config.at("iso").get<std::string>());
		write_metadata_file(metadata_file, nlohmann::json::object());
	} catch (const std::exception& error) {
		std::throw_with_nested("creating vm");
	}
}

std::string VmController::get_metadata(const std::string& key) {
	try {
		fs::path metadata_file = env->metadata_dir() / vm->name();
		auto metadata = read_metadata_file(metadata_file);
		if (!metadata.count(key)) {
			throw std::runtime_error("Requested key is not present in vm metadata");
		}
		return metadata.at(key).get<std::string>();

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Getting metadata with key {}", key)));
	}
}

void VmController::set_metadata(const std::string& key, const std::string& value) {
	try {
		fs::path metadata_file = env->metadata_dir() / vm->name();
		auto metadata = read_metadata_file(metadata_file);
		metadata[key] = value;
		write_metadata_file(metadata_file, metadata);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting metadata with key {}", key)));
	}
}

void VmController::write_metadata_file(const fs::path& file, const nlohmann::json& metadata) {
	std::ofstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't write metadata file " + file.generic_string());
	}

	metadata_file_stream << metadata;
	metadata_file_stream.close();
}

nlohmann::json VmController::read_metadata_file(const fs::path& file) const {
	std::ifstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't read metadata file " + file.generic_string());
	}

	nlohmann::json result;
	metadata_file_stream >> result;
	metadata_file_stream.close();
	return result;
}
