
#include "VmController.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

void VmController::create() {
	try {
		fs::path metadata_dir = env->metadata_dir() / vm->name();

		if (fs::exists(metadata_dir)) {
			if (!fs::remove_all(metadata_dir)) {
				throw std::runtime_error("Error deleting metadata dir " + metadata_dir.generic_string());
			}
		}

		vm->install();

		auto config = vm->get_config();

		nlohmann::json metadata;

		if (config.count("metadata")) {
			auto config_metadata = config.at("metadata");
			for (auto it = config_metadata.begin(); it != config_metadata.end(); ++it) {
				metadata[it.key()] = it.value();
			}
		}

		if (!fs::create_directory(metadata_dir)) {
			throw std::runtime_error("Error creating metadata dir " + metadata_dir.generic_string());
		}

		fs::path metadata_file = metadata_dir / vm->name();

		metadata["vm_config"] = config.dump();
		metadata["vm_nic_count"] = std::to_string(config.count("nic") ? config.at("nic").size() : 0);
		metadata["vm_name"] = config.at("name");
		metadata["vm_current_state"] = "";
		metadata["dvd_signature"] = file_signature(config.at("iso").get<std::string>());
		write_metadata_file(metadata_file, metadata);

	} catch (const std::exception& error) {
		std::throw_with_nested("creating vm");
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
		fs::path metadata_file = env->metadata_dir() / vm->name();
		metadata_file /= vm->name() + "_" + snapshot;

		auto current_state = get_metadata("vm_current_state");

		nlohmann::json metadata;
		metadata["cksum"] = cksum;
		metadata["children"] = nlohmann::json::array();
		metadata["parent"] = current_state;
		write_metadata_file(metadata_file, metadata);

		//link parent to a child
		if (current_state.length()) {
			fs::path parent_metadata_file = env->metadata_dir() / vm->name();
			parent_metadata_file /= vm->name() + "_" + current_state;
			auto parent_metadata = read_metadata_file(parent_metadata_file);
			parent_metadata.at("children").push_back(snapshot);
			write_metadata_file(parent_metadata_file, parent_metadata);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested("creating snapshot");
	}
}

void VmController::restore_snapshot(const std::string& snapshot) {
	vm->rollback(snapshot);
	set_metadata("vm_current_state", snapshot);
}

void VmController::delete_snapshot_with_children(const std::string& snapshot)
{
	try {
		//This thins needs to be recursive
		//I guess... go through the children and call recursively on them
		fs::path metadata_file = env->metadata_dir() / vm->name();
		metadata_file /= vm->name() + "_" + snapshot;

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
			fs::path parent_metadata_file = env->metadata_dir() / vm->name();
			parent_metadata_file /= vm->name() + "_" + parent;

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
		std::throw_with_nested("deleting snapshot");
	}
}

bool VmController::has_snapshot(const std::string& snapshot) {
	fs::path metadata_file = env->metadata_dir() / vm->name();
	metadata_file /= vm->name() + "_" + snapshot;
	return fs::exists(metadata_file);
}

std::string VmController::get_snapshot_cksum(const std::string& snapshot) {
	try {
		fs::path metadata_file = env->metadata_dir() / vm->name();
		metadata_file /= vm->name() + "_" + snapshot;
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

bool VmController::has_key(const std::string& key) {
	try {
		fs::path metadata_file = env->metadata_dir() / vm->name();
		metadata_file /= vm->name();
		auto metadata = read_metadata_file(metadata_file);
		return metadata.count(key);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking metadata with key {}", key)));
	}
}


std::string VmController::get_metadata(const std::string& key) {
	try {
		fs::path metadata_file = env->metadata_dir() / vm->name();
		metadata_file /= vm->name();
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
		metadata_file /= vm->name();
		auto metadata = read_metadata_file(metadata_file);
		metadata[key] = value;
		write_metadata_file(metadata_file, metadata);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting metadata with key {}", key)));
	}
}
