
#include "VM.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

VM::VM(const nlohmann::json& config_): config(config_) {
	if (config.count("iso")) {
		fs::path iso_file = config.at("iso").get<std::string>();
		if (iso_file.is_relative()) {
			fs::path src_file(config.at("src_file").get<std::string>());
			iso_file = src_file.parent_path() / iso_file;
		}

		if (!fs::exists(iso_file)) {
			throw std::runtime_error(fmt::format("Can't construct VmController for vm {}: target iso file {} doesn't exist", name(), iso_file.generic_string()));
		}

		iso_file = fs::canonical(iso_file);

		config["iso"] = iso_file.generic_string();
	}

	if (config.count("disk")) {
		auto& disks = config.at("disk");

		for (auto& disk: disks) {
			if (disk.count("source")) {
				fs::path source_file = disk.at("source").get<std::string>();
				if (source_file.is_relative()) {
					fs::path src_file(config.at("src_file").get<std::string>());
					source_file = src_file.parent_path() / source_file;
				}

				if (!fs::exists(source_file)) {
					throw std::runtime_error(fmt::format("Can't construct VmController for vm {}: source disk image {} doesn't exist", name(), source_file.generic_string()));
				}

				source_file = fs::canonical(source_file);
				disk["source"] = source_file;
			}
		}
	}
}

nlohmann::json VM::get_config() const {
	return config;
}

std::string VM::id() const {
	return config.at("prefix").get<std::string>() + config.at("name").get<std::string>();
}

std::string VM::name() const {
	return config.at("name");
}

std::string VM::prefix() const {
	return config.at("prefix");
}

std::set<std::string> VM::nics() const {
	std::set<std::string> result;

	if (config.count("nic")) {
		for (auto& nic: config.at("nic")) {
			result.insert(nic.at("name").get<std::string>());
		}
	}
	return result;
}

std::set<std::string> VM::networks() const {
	std::set<std::string> result;

	if (config.count("nic")) {
		auto nics = config.at("nic");
		for (auto& nic: nics) {
			std::string source_network = nic.at("attached_to").get<std::string>();
			result.insert(source_network);
		}
	}

	return result;
}

