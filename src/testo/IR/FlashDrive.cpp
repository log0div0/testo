
#include "FlashDrive.hpp"
#include <fmt/format.h>
#include "../backends/Environment.hpp"

namespace IR {

std::shared_ptr<::FlashDrive> FlashDrive::fd() const {
	if (!_fd) {
		_fd = env->create_flash_drive(config);
	}
	return _fd;
}

std::string FlashDrive::type() const {
	return "flash drive";
}

std::string FlashDrive::id() const {
	return fd()->id();
}

bool FlashDrive::is_defined() const {
	return Controller::is_defined() && fd()->is_defined();
}

void FlashDrive::create() {
	try {
		undefine();

		// Check if we have the init snapshot. If we do and config is relevant
		// then just rollback there and exit. If not - do the usual procedure

		fd()->create();

		if (has_folder()) {
			fd()->load_folder();
		}

		auto fd_config = config;

		nlohmann::json metadata;

		fs::path metadata_dir = get_metadata_dir();
		if (!fs::create_directory(metadata_dir)) {
			throw std::runtime_error("Error creating metadata dir " + metadata_dir.generic_string());
		}

		std::string cksum_input = "";
		if (has_folder()) {
			fs::path folder(fd_config.at("folder").get<std::string>());
			if (folder.is_relative()) {
				fs::path src_file(fd_config.at("src_file").get<std::string>());
				folder = src_file.parent_path() / folder;
			}
			folder = fs::canonical(folder);
			cksum_input += directory_signature(folder, env->content_cksum_maxsize());
		}

		std::hash<std::string> h;

		auto folder_cksum = std::to_string(h(cksum_input));

		fd_config.erase("src_file");
		metadata["fd_config"] = fd_config.dump();
		metadata["fd_name"] = fd_config.at("name");
		metadata["folder_cksum"] = folder_cksum;

		write_metadata_file(main_file(), metadata);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("creating fd"));
	}
}

void FlashDrive::undefine() {
	try {
		auto metadata_dir = get_metadata_dir();
		if (!fd()->is_defined()) {
			if (fs::exists(metadata_dir)) {
				//The check would be valid only if we have the main file

				if (!fs::remove_all(metadata_dir)) {
					throw std::runtime_error("Error deleting metadata dir " + metadata_dir.generic_string());
				}
			}
			return;
		}

		if (Controller::has_snapshot("_init")) {
			delete_snapshot_with_children("_init");
		}

		fd()->undefine();

		if (fs::exists(metadata_dir)) {
			if (!fs::remove_all(metadata_dir)) {
				throw std::runtime_error("Error deleting metadata dir " + metadata_dir.generic_string());
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("undefining flash drive controller"));
	}
}

void FlashDrive::create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed)
{
	try {
		if (has_snapshot(snapshot)) {
			delete_snapshot_with_children(snapshot);
		}

		//1) Let's try and create the actual snapshot. If we fail then no additional work
		if (hypervisor_snapshot_needed) {
			fd()->make_snapshot(snapshot);
		}

		//Where to store new metadata file?
		fs::path metadata_file = get_metadata_dir();
		metadata_file /= fd()->id() + "_" + snapshot;

		nlohmann::json metadata;
		metadata["cksum"] = cksum;
		metadata["children"] = nlohmann::json::array();
		metadata["parent"] = current_state;
		write_metadata_file(metadata_file, metadata);

		//link parent to a child
		if (current_state.length()) {
			fs::path parent_metadata_file = get_metadata_dir();
			parent_metadata_file /= fd()->id() + "_" + current_state;
			auto parent_metadata = read_metadata_file(parent_metadata_file);
			parent_metadata.at("children").push_back(snapshot);
			write_metadata_file(parent_metadata_file, parent_metadata);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("creating snapshot"));
	}
}

void FlashDrive::restore_snapshot(const std::string& snapshot) {
	fd()->rollback(snapshot);
	current_state = snapshot;
}

void FlashDrive::delete_snapshot_with_children(const std::string& snapshot)
{
	try {
		//This thins needs to be recursive
		//I guess... go through the children and call recursively on them
		fs::path metadata_file = get_metadata_dir();
		metadata_file /= fd()->id() + "_" + snapshot;

		auto metadata = read_metadata_file(metadata_file);

		for (auto& child: metadata.at("children")) {
			delete_snapshot_with_children(child.get<std::string>());
		}

		//Now we're at the bottom of the hierarchy
		//Delete the hypervisor child if we have one

		if (fd()->has_snapshot(snapshot)) {
			fd()->delete_snapshot(snapshot);
		}

		//Ok, now we need to get our parent
		auto parent = metadata.at("parent").get<std::string>();

		//Unlink the parent
		if (parent.length()) {
			fs::path parent_metadata_file = env->flash_drives_metadata_dir() / fd()->id();
			parent_metadata_file /= fd()->id() + "_" + parent;

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

bool FlashDrive::check_config_relevance() {
	auto old_config = nlohmann::json::parse(get_metadata("fd_config"));
	auto new_config = config;

	new_config.erase("src_file");

	if (old_config != new_config) {
		return false;
	}

	new_config = config;

	std::string cksum_input = "";
	if (has_folder()) {
		fs::path folder(new_config.at("folder").get<std::string>());
		if (folder.is_relative()) {
			fs::path src_file(new_config.at("src_file").get<std::string>());
			folder = src_file.parent_path() / folder;
		}
		folder = fs::canonical(folder);
		cksum_input += directory_signature(folder, env->content_cksum_maxsize());
	}

	std::hash<std::string> h;
	bool cksums_are_ok = (get_metadata("folder_cksum") == std::to_string(h(cksum_input)));

	return cksums_are_ok;
}

fs::path FlashDrive::get_metadata_dir() const{
	return env->flash_drives_metadata_dir() / id();
}

bool FlashDrive::has_folder() const {
	return config.count("folder");
}

void FlashDrive::validate_folder() const {
	try {
		fs::path folder(config.at("folder").get<std::string>());
		if (folder.is_relative()) {
			fs::path src_file(config.at("src_file").get<std::string>());
			folder = src_file.parent_path() / folder;
		}

		if (!fs::exists(folder)) {
			throw std::runtime_error("Target folder doesn't exist");
		}

		folder = fs::canonical(folder);

		if (!fs::is_directory(folder)) {
			throw std::runtime_error(fmt::format("specified folder {} for flash drive {} is not a folder",
				folder.generic_string(), name()));
		}
	} catch (const std::runtime_error& error) {
		std::throw_with_nested(std::runtime_error("Validating host folder failed"));
	}

}

}
