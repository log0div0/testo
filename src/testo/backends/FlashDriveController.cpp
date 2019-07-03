
#include "FlashDriveController.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

std::string FlashDriveController::name() const {
	return fd->name();
}

bool FlashDriveController::is_defined() {
	return fd->is_defined();
}

void FlashDriveController::create() {
	try {
		fs::path metadata_dir = env->flash_drives_metadata_dir() / fd->name();

		if (fs::exists(metadata_dir)) {
			if (!fs::remove_all(metadata_dir)) {
				throw std::runtime_error("Error deleting metadata dir " + metadata_dir.generic_string());
			}
		}

		fd->create();

		auto config = fd->get_config();

		nlohmann::json metadata;

		if (!fs::create_directory(metadata_dir)) {
			throw std::runtime_error("Error creating metadata dir " + metadata_dir.generic_string());
		}

		fs::path metadata_file = metadata_dir / fd->name();

		metadata["fd_config"] = config.dump();
		metadata["fd_name"] = config.at("name");
		metadata["current_state"] = "";
		write_metadata_file(metadata_file, metadata);

	} catch (const std::exception& error) {
		std::throw_with_nested("creating fd");
	}
}

void FlashDriveController::create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed)
{
	try {
		if (has_snapshot(snapshot)) {
			delete_snapshot_with_children(snapshot);
		}

		//1) Let's try and create the actual snapshot. If we fail then no additional work
		if (hypervisor_snapshot_needed) {
			fd->make_snapshot(snapshot);
		}

		//Where to store new metadata file?
		fs::path metadata_file = env->flash_drives_metadata_dir() / fd->name();
		metadata_file /= fd->name() + "_" + snapshot;

		auto current_state = get_metadata("current_state");

		nlohmann::json metadata;
		metadata["cksum"] = cksum;
		metadata["children"] = nlohmann::json::array();
		metadata["parent"] = current_state;
		write_metadata_file(metadata_file, metadata);

		//link parent to a child
		if (current_state.length()) {
			fs::path parent_metadata_file = env->flash_drives_metadata_dir() / fd->name();
			parent_metadata_file /= fd->name() + "_" + current_state;
			auto parent_metadata = read_metadata_file(parent_metadata_file);
			parent_metadata.at("children").push_back(snapshot);
			write_metadata_file(parent_metadata_file, parent_metadata);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested("creating snapshot");
	}
}

void FlashDriveController::restore_snapshot(const std::string& snapshot) {
	fd->rollback(snapshot);
	set_metadata("current_state", snapshot);
}

void FlashDriveController::delete_snapshot_with_children(const std::string& snapshot)
{
	try {
		//This thins needs to be recursive
		//I guess... go through the children and call recursively on them
		fs::path metadata_file = env->flash_drives_metadata_dir() / fd->name();
		metadata_file /= fd->name() + "_" + snapshot;

		auto metadata = read_metadata_file(metadata_file);

		for (auto& child: metadata.at("children")) {
			delete_snapshot_with_children(child.get<std::string>());
		}

		//Now we're at the bottom of the hierarchy
		//Delete the hypervisor child if we have one

		if (fd->has_snapshot(snapshot)) {
			fd->delete_snapshot(snapshot);
		}

		//Ok, now we need to get our parent
		auto parent = metadata.at("parent").get<std::string>();

		//Unlink the parent
		if (parent.length()) {
			fs::path parent_metadata_file = env->flash_drives_metadata_dir() / fd->name();
			parent_metadata_file /= fd->name() + "_" + parent;

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

bool FlashDriveController::has_snapshot(const std::string& snapshot) {
	fs::path metadata_file = env->flash_drives_metadata_dir() / fd->name();
	metadata_file /= fd->name() + "_" + snapshot;
	return fs::exists(metadata_file);
}

std::string FlashDriveController::get_snapshot_cksum(const std::string& snapshot) {
	try {
		fs::path metadata_file = env->flash_drives_metadata_dir() / fd->name();
		metadata_file /= fd->name() + "_" + snapshot;
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

bool FlashDriveController::has_key(const std::string& key) {
	try {
		fs::path metadata_file = env->flash_drives_metadata_dir() / fd->name();
		metadata_file /= fd->name();
		auto metadata = read_metadata_file(metadata_file);
		return metadata.count(key);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking metadata with key {}", key)));
	}
}


std::string FlashDriveController::get_metadata(const std::string& key) {
	try {
		fs::path metadata_file = env->flash_drives_metadata_dir() / fd->name();
		metadata_file /= fd->name();
		auto metadata = read_metadata_file(metadata_file);
		if (!metadata.count(key)) {
			throw std::runtime_error("Requested key is not present in vm metadata");
		}
		return metadata.at(key).get<std::string>();

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Getting metadata with key {}", key)));
	}
}

void FlashDriveController::set_metadata(const std::string& key, const std::string& value) {
	try {
		fs::path metadata_file = env->flash_drives_metadata_dir() / fd->name();
		metadata_file /= fd->name();
		auto metadata = read_metadata_file(metadata_file);
		metadata[key] = value;
		write_metadata_file(metadata_file, metadata);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting metadata with key {}", key)));
	}
}

bool FlashDriveController::check_config_relevance() {
	auto old_config = nlohmann::json::parse(get_metadata("vm_config"));
	auto new_config = fd->get_config();

	return (old_config == new_config);
}
