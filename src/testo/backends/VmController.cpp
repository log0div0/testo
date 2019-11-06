
#include "VmController.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

std::string VmController::id() const {
	return vm->id();
}

std::string VmController::name() const {
	return vm->name();
}

bool VmController::is_defined() const {
	return Controller::is_defined() && vm->is_defined();
}

void VmController::create() {
	try {
		if (fs::exists(get_metadata_dir())) {
			if (!fs::remove_all(get_metadata_dir())) {
				throw std::runtime_error("Error deleting metadata dir " + get_metadata_dir().generic_string());
			}
		}

		vm->install();

		auto config = vm->get_config();

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
		write_metadata_file(main_file(), metadata);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("creating vm"));
	}
}

void VmController::create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed)
{
	try {
		if (has_snapshot(snapshot)) {
			delete_snapshot_with_children(snapshot);
		}

		//1) Let's try and create the actual snapshot. If we fail then no additional work
		if (hypervisor_snapshot_needed) {
			vm->make_snapshot(snapshot);
		}

		//Where to store new metadata file?
		fs::path metadata_file = get_metadata_dir();
		metadata_file /= vm->id() + "_" + snapshot;

		auto current_state = get_metadata("current_state");

		nlohmann::json metadata;
		metadata["cksum"] = cksum;
		metadata["children"] = nlohmann::json::array();
		metadata["parent"] = current_state;
		write_metadata_file(metadata_file, metadata);

		//link parent to a child
		if (current_state.length()) {
			fs::path parent_metadata_file = get_metadata_dir();
			parent_metadata_file /= vm->id() + "_" + current_state;
			auto parent_metadata = read_metadata_file(parent_metadata_file);
			parent_metadata.at("children").push_back(snapshot);
			write_metadata_file(parent_metadata_file, parent_metadata);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("creating snapshot"));
	}
}

void VmController::restore_snapshot(const std::string& snapshot) {
	vm->rollback(snapshot);
	set_metadata("current_state", snapshot);
}

void VmController::delete_snapshot_with_children(const std::string& snapshot)
{
	try {
		//This thins needs to be recursive
		//I guess... go through the children and call recursively on them
		fs::path metadata_file = get_metadata_dir();
		metadata_file /= vm->id() + "_" + snapshot;

		auto metadata = read_metadata_file(metadata_file);

		for (auto& child: metadata.at("children")) {
			delete_snapshot_with_children(child.get<std::string>());
		}

		//Now we're at the bottom of the hierarchy
		//Delete the hypervisor child if we have one

		if (vm->has_snapshot(snapshot)) {
			vm->delete_snapshot(snapshot);
		}

		//Ok, now we need to get our parent
		auto parent = metadata.at("parent").get<std::string>();

		//Unlink the parent
		if (parent.length()) {
			fs::path parent_metadata_file = get_metadata_dir();
			parent_metadata_file /= vm->id() + "_" + parent;

			auto parent_metadata = read_metadata_file(parent_metadata_file);
			auto& children = parent_metadata.at("children");

			for (auto it = children.begin(); it != children.end(); ++it) {
				if (it.value() == snapshot) {
					children.erase(it);
					break;
				}
			}
			write_metadata_file(parent_metadata_file, parent_metadata);
		}

		//Now we can delete the metadata file
		if (!fs::remove(metadata_file)) {
			throw std::runtime_error("Error deleting metadata file " + metadata_file.generic_string());
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("deleting snapshot"));
	}
}

bool VmController::has_user_key(const std::string& key) {
	try {
		auto metadata = read_metadata_file(main_file());
		return metadata["user_metadata"].count(key);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking metadata with key {}", key)));
	}
}


std::string VmController::get_user_metadata(const std::string& key) {
	try {
		auto metadata = read_metadata_file(main_file());
		if (!metadata["user_metadata"].count(key)) {
			throw std::runtime_error("Requested key is not present in vm metadata");
		}
		return metadata["user_metadata"].at(key).get<std::string>();

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Getting metadata with key {}", key)));
	}
}

void VmController::set_user_metadata(const std::string& key, const std::string& value) {
	try {
		auto metadata = read_metadata_file(main_file());
		metadata["user_metadata"][key] = value;
		write_metadata_file(main_file(), metadata);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting metadata with key {}", key)));
	}
}

void VmController::update_user_metadata() {
	auto metadata = read_metadata_file(main_file());
	//need to preserve vm_name and vm_nic_count

	auto vm_name = metadata["user_metadata"].at("vm_name").get<std::string>();
	auto vm_nic_count = metadata["user_metadata"].at("vm_nic_count").get<std::string>();

	//we just... update it.... completely
	auto new_config = vm->get_config();


	if (new_config.count("metadata")) {
		metadata["user_metadata"] = vm->get_config().at("metadata");
	} else {
		metadata["user_metadata"] = nlohmann::json::object();
	}

	metadata["user_metadata"]["vm_name"] = vm_name;
	metadata["user_metadata"]["vm_nic_count"] = vm_nic_count;
	write_metadata_file(main_file(), metadata);
}

bool VmController::check_config_relevance() {
	try {
		update_user_metadata();
	} catch (const std::exception& error) {
		return false;
	}

	auto old_config = nlohmann::json::parse(get_metadata("vm_config"));
	auto new_config = vm->get_config();
	//So....
	//1) get rid of metadata
	new_config.erase("metadata");
	old_config.erase("user_metadata");

	//2) Actually.... Let's just be practical here.
	//Check if both have or don't have nics

	auto old_nics = old_config.value("nic", nlohmann::json::array());
	auto new_nics = new_config.value("nic", nlohmann::json::array());

	if (old_nics.size() != new_nics.size()) {
		return false;
	}

	if (!std::is_permutation(old_nics.begin(), old_nics.end(), new_nics.begin())) {
		return false;
	}

	new_config.erase("nic");
	old_config.erase("nic");

	new_config.erase("iso");
	//old_config already doesn't have the iso

	new_config.erase("src_file");
	//old_config already doesn't have the src_file

	bool config_is_ok = (old_config == new_config);

	//Check also dvd contingency

	fs::path iso_file = vm->get_config().at("iso").get<std::string>();
	if (iso_file.is_relative()) {
		fs::path src_file(vm->get_config().at("src_file").get<std::string>());
		iso_file = src_file.parent_path() / iso_file;
	}
	iso_file = fs::canonical(iso_file);

	bool iso_is_ok = (file_signature(iso_file) == get_metadata("dvd_signature"));

	return (config_is_ok && iso_is_ok);
}

fs::path VmController::get_metadata_dir() const {
	return env->vm_metadata_dir() / id();
}

