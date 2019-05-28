
#include "FlashDriveController.hpp"
#include <fmt/format.h>
#include <fstream>

FlashDriveController::FlashDriveController(const nlohmann::json& config_): config(config_) {
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

std::string FlashDriveController::name() const {
	return config.at("name").get<std::string>();
}

bool FlashDriveController::has_folder() const {
	return config.count("folder");
}

nlohmann::json FlashDriveController::get_config() const {
	return config;
}

bool FlashDriveController::cache_enabled() const {
	return config.value("cache_enabled", 1);
}

std::string FlashDriveController::cksum() const {
	return read_cksum();
}

std::string FlashDriveController::read_cksum() const {
	if (!fs::exists(cksum_path())) {
		return "";
	};

	if (!fs::is_regular_file(cksum_path())) {
		return "";
	};

	std::ifstream input_stream(cksum_path());

	if (!input_stream) {
		return "";
	}

	std::string result = std::string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
	return result;
}

void FlashDriveController::write_cksum(const std::string& cksum) const {
	std::ofstream output_stream(cksum_path(), std::ofstream::out);
	if (!output_stream) {
		throw std::runtime_error(std::string("Can't create file for writing cksum: ") + cksum_path().generic_string());
	}
	output_stream << cksum;
}

std::string FlashDriveController::calc_cksum() const {
	std::string cksum_input = name() + std::to_string(config.at("size").get<uint32_t>()) + config.at("fs").get<std::string>();
	if (has_folder()) {
		cksum_input += directory_signature(config.at("folder").get<std::string>());
	}

	std::hash<std::string> h;
	return std::to_string(h(cksum_input));
}

void FlashDriveController::delete_cksum() const {
	if (fs::exists(cksum_path())) {
		fs::remove(cksum_path());
	}
}

fs::path FlashDriveController::cksum_path() const {
	return img_path().generic_string() + ".cksum";
}

void FlashDriveController::load_folder() const {
	try {
		fs::path target_folder(config.at("folder").get<std::string>());

		if (target_folder.is_relative()) {
			target_folder = fs::canonical(target_folder);
		}

		if (!fs::exists(target_folder)) {
			throw std::runtime_error("Target folder doesn't exist");
		}

		mount();
		fs::copy(target_folder, mount_dir(), fs::copy_options::overwrite_existing | fs::copy_options::recursive);
		umount();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}
