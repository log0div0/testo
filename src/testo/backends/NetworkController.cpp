
#include "NetworkController.hpp"
#include "Environment.hpp"
#include "../Utils.hpp"
#include <fmt/format.h>

std::string NetworkController::id() const {
	return network->id();
}

std::string NetworkController::name() const {
	return network->name();
}

bool NetworkController::is_defined() const {
	return fs::exists(main_file()) && network->is_defined();
}

void NetworkController::create() {
	try {
		if (fs::exists(get_metadata_dir())) {
			if (!fs::remove_all(get_metadata_dir())) {
				throw std::runtime_error("Error deleting metadata dir " + get_metadata_dir().generic_string());
			}
		}

		network->create();

		auto config = network->get_config();

		nlohmann::json metadata;

		if (!fs::create_directory(get_metadata_dir())) {
			throw std::runtime_error("Error creating metadata dir " + get_metadata_dir().generic_string());
		}

		metadata["network_config"] = config.dump();
		metadata["current_state"] = "";
		write_metadata_file(main_file(), metadata);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("creating network"));
	}
}

std::string NetworkController::get_metadata(const std::string& key) const {
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

bool NetworkController::check_config_relevance() {
	auto old_config = nlohmann::json::parse(get_metadata("network_config"));
	auto new_config = network->get_config();

	new_config.erase("src_file");
	//old_config already doesn't have the src_file

	return (old_config == new_config);
}

fs::path NetworkController::get_metadata_dir() const {
	return env->network_metadata_dir() / id();
}

