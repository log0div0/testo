
#include "NetworkController.hpp"
#include "Environment.hpp"
#include "../Utils.hpp"
#include <fmt/format.h>

std::string NetworkController::id() const {
	network->id();
}

std::string NetworkController::name() const {
	return network->name();
}

bool NetworkController::is_defined() const {
	return Controller::is_defined() && network->is_defined();
}

void NetworkController::create() {
	try {
		if (fs::exists(get_metadata_dir())) {
			if (!fs::remove_all(get_metadata_dir())) {
				throw std::runtime_error("Error deleting metadata dir " + get_metadata_dir().generic_string());
			}
		}

		network->create();

		/*auto config = network->get_config();

		nlohmann::json metadata;

		metadata["user_metadata"] = nlohmann::json::object();

		if (config.count("metadata")) {
			auto config_metadata = config.at("metadata");
			for (auto it = config_metadata.begin(); it != config_metadata.end(); ++it) {
				metadata["user_metadata"][it.key()] = it.value();
			}
		}

		if (!fs::create_directory(get_metadata_dir())) {
			throw std::runtime_error("Error creating metadata dir " + get_metadata_dir().generic_string());
		}

		fs::path iso_file = config.at("iso").get<std::string>();
		if (iso_file.is_relative()) {
			fs::path src_file(config.at("src_file").get<std::string>());
			iso_file = src_file.parent_path() / iso_file;
		}
		iso_file = fs::canonical(iso_file);

		if (!fs::exists(iso_file)) {
			throw std::runtime_error("Target iso file doesn't exist");
		}

		config.erase("src_file");
		config.erase("iso");
		config.erase("metadata");

		metadata["vm_config"] = config.dump();
		metadata["user_metadata"]["vm_nic_count"] = std::to_string(config.count("nic") ? config.at("nic").size() : 0);
		metadata["user_metadata"]["vm_name"] = config.at("name");
		metadata["current_state"] = "";
		metadata["dvd_signature"] = file_signature(iso_file);
		write_metadata_file(main_file(), metadata);*/
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("creating vm"));
	}
}

void NetworkController::create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed)
{

}

void NetworkController::restore_snapshot(const std::string& snapshot) {

}

void NetworkController::delete_snapshot_with_children(const std::string& snapshot)
{

}

bool NetworkController::has_user_key(const std::string& key) {

}


std::string NetworkController::get_user_metadata(const std::string& key) {

}

void NetworkController::set_user_metadata(const std::string& key, const std::string& value) {

}

bool NetworkController::check_config_relevance() {

}

fs::path NetworkController::get_metadata_dir() const {
}

