
#include "FlashDrive.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

#ifdef __linux__
#include <unistd.h>
#endif

FlashDrive::FlashDrive(const nlohmann::json& config_): config(config_) {
	if (!config.count("name")) {
		throw std::runtime_error("Constructing VboxFlashDriveController error: field NAME is not specified");
	}

	if (!config.count("size")) {
		throw std::runtime_error("Constructing VboxFlashDriveController error: field SIZE is not specified");
	}

	//TODO: check for fs types
	if (!config.count("fs")) {
		throw std::runtime_error("Constructing VboxFlashDriveController error: field FS is not specified");
	}

	if (config.count("folder")) {
		fs::path folder(config.at("folder").get<std::string>());
		if (folder.is_relative()) {
			fs::path src_file(config.at("src_file").get<std::string>());
			folder = src_file.parent_path() / folder;
		}
		folder = fs::canonical(folder);
		if (!fs::exists(folder)) {
			throw std::runtime_error(fmt::format("specified folder {} for flash drive {} does not exist",
				folder.generic_string(), name()));
		}

		if (!fs::is_directory(folder)) {
			throw std::runtime_error(fmt::format("specified folder {} for flash drive {} is not a folder",
				folder.generic_string(), name()));
		}
	}
}

std::string FlashDrive::name() const {
	return config.at("name").get<std::string>();
}

bool FlashDrive::has_folder() const {
	return config.count("folder");
}

nlohmann::json FlashDrive::get_config() const {
	return config;
}

void FlashDrive::load_folder() const {
	try {
		fs::path folder(config.at("folder").get<std::string>());
		if (folder.is_relative()) {
			fs::path src_file(config.at("src_file").get<std::string>());
			folder = src_file.parent_path() / folder;
		}
		folder = fs::canonical(folder);

		if (!fs::exists(folder)) {
			throw std::runtime_error("Target folder doesn't exist");
		}

		mount();
		fs::copy(folder, env->flash_drives_mount_dir(), fs::copy_options::overwrite_existing | fs::copy_options::recursive);
#ifdef __linux__
		sync();
#endif
		umount();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}
