
#include "NetworkController.hpp"
#include "Environment.hpp"
#include "../Utils.hpp"
#include <fmt/format.h>

NetworkController::NetworkController(const fs::path& main_file) {

}

std::string NetworkController::id() const {
	return network->id();
}

std::string NetworkController::name() const {
	return network->name();
}

std::string NetworkController::prefix() const {
	return network->prefix();
}


bool NetworkController::is_defined() const {
	return fs::exists(main_file()) && network->is_defined();
}

void NetworkController::undefine() {
	try {
		auto metadata_dir = get_metadata_dir();
		if (!network->is_defined()) {
			if (fs::exists(metadata_dir)) {
				//The check would be valid only if we have the main file

				if (!fs::remove_all(metadata_dir)) {
					throw std::runtime_error("Error deleting metadata dir " + metadata_dir.generic_string());
				}
			}
			return;
		}

		network->undefine();

		if (!fs::remove_all(metadata_dir)) {
			throw std::runtime_error("Error deleting metadata dir " + metadata_dir.generic_string());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("undefining network controller"));
	}
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

		config.erase("src_file");

		metadata["network_config"] = config.dump();
		metadata["current_state"] = "";
		write_metadata_file(main_file(), metadata);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("creating network"));
	}
}

std::string NetworkController::get_metadata(const std::string& key) const {
	try {
		return ::get_metadata(main_file(), key);

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Getting network metadata with key {}", key)));
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

