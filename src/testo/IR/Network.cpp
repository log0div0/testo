
#include "Network.hpp"
#include <fmt/format.h>
#include "../backends/Environment.hpp"

namespace IR {

std::shared_ptr<::Network> Network::nw() const {
	if (!_nw) {
		_nw = env->create_network(config);
	}
	return _nw;
}

std::string Network::type() const {
	return "network";
}

std::string Network::id() const {
	return nw()->id();
}

bool Network::is_defined() const {
	return fs::exists(main_file()) && nw()->is_defined();
}

void Network::create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void Network::restore_snapshot(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void Network::delete_snapshot_with_children(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void Network::undefine() {
	try {
		auto metadata_dir = get_metadata_dir();
		if (!nw()->is_defined()) {
			if (fs::exists(metadata_dir)) {
				//The check would be valid only if we have the main file

				if (!fs::remove_all(metadata_dir)) {
					throw std::runtime_error("Error deleting metadata dir " + metadata_dir.generic_string());
				}
			}
			return;
		}

		nw()->undefine();

		if (fs::exists(metadata_dir)) {
			if (!fs::remove_all(metadata_dir)) {
				throw std::runtime_error("Error deleting metadata dir " + metadata_dir.generic_string());
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("undefining network controller"));
	}
}

void Network::create() {
	try {
		undefine();

		if (fs::exists(get_metadata_dir())) {
			if (!fs::remove_all(get_metadata_dir())) {
				throw std::runtime_error("Error deleting metadata dir " + get_metadata_dir().generic_string());
			}
		}

		nw()->create();

		auto nw_config = config;

		nlohmann::json metadata;

		if (!fs::create_directory(get_metadata_dir())) {
			throw std::runtime_error("Error creating metadata dir " + get_metadata_dir().generic_string());
		}

		nw_config.erase("src_file");

		metadata["network_config"] = nw_config.dump();
		write_metadata_file(main_file(), metadata);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("creating network"));
	}
}

bool Network::check_config_relevance() {
	auto old_config = nlohmann::json::parse(get_metadata("network_config"));
	auto new_config = config;

	new_config.erase("src_file");
	//old_config already doesn't have the src_file

	return (old_config == new_config);
}

fs::path Network::get_metadata_dir() const {
	return env->network_metadata_dir() / id();
}

void Network::validate_config() {
	if (!config.count("mode")) {
		throw std::runtime_error("Constructing NetworkController error: field MODE is not specified");
	}

	auto mode = config.at("mode").get<std::string>();

	if ((mode != "nat") &&
		(mode != "internal"))
	{
		throw std::runtime_error(std::string("Constructing NetworkController error: Unsupported MODE: ") + mode);
	}
}

}
