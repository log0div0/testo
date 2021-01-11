
#include "VM.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

VM::VM(const nlohmann::json& config_): config(config_) {
	if (!config.count("name")) {
		throw std::runtime_error("Constructing VM \"" + id() + "\" error: field \"name\" is not specified");
	}

	if (!config.count("ram")) {
		throw std::runtime_error("Constructing VM \"" + id() + "\" error: field \"ram\" is not specified");
	}

	if (!config.count("cpus")) {
		throw std::runtime_error("Constructing VM \"" + id() + "\" error: field \"cpu\" is not specified");
	}

	if (!config.count("disk")) {
		throw std::runtime_error("Constructing VM \"" + id() + "\" error: you must specify at least 1 disk");
	}

	if (config.count("disk")) {
		auto disks = config.at("disk");

		for (auto& disk: disks) {
			if (!(disk.count("size") ^ disk.count("source"))) {
				throw std::runtime_error("Constructing VM \"" + id() + "\" error: either field \"size\" or \"source\" must be specified for the disk \"" +
					disk.at("name").get<std::string>() + "\"");
			}
		}

		for (uint32_t i = 0; i < disks.size(); i++) {
			for (uint32_t j = i + 1; j < disks.size(); j++) {
				if (disks[i].at("name") == disks[j].at("name")) {
					throw std::runtime_error("Constructing VM \"" + id() + "\" error: two identical disk names: \"" +
						disks[i].at("name").get<std::string>() + "\"");
				}
			}
		}
	}

	if (config.count("nic")) {
		auto nics = config.at("nic");
		for (auto& nic: nics) {
			if (!nic.count("attached_to")) {
				throw std::runtime_error("Constructing VM \"" + id() + "\" error: field attached_to is not specified for the nic \"" +
					nic.at("name").get<std::string>() + "\"");
			}

			if (nic.count("mac")) {
				std::string mac = nic.at("mac").get<std::string>();
				if (!is_mac_correct(mac)) {
					throw std::runtime_error("Constructing VM \"" + id() + "\" error: incorrect mac string: \"" + mac + "\"");
				}
			}
		}

		for (uint32_t i = 0; i < nics.size(); i++) {
			for (uint32_t j = i + 1; j < nics.size(); j++) {
				if (nics[i].at("name") == nics[j].at("name")) {
					throw std::runtime_error("Constructing VM \"" + id() + "\" error: two identical NIC names: \"" +
						nics[i].at("name").get<std::string>() + "\"");
				}
			}
		}
	}

	if (config.count("video")) {
		auto videos = config.at("video");

		if (videos.size() > 1) {
			throw std::runtime_error("Constructing VM \"" + id() + "\" error: multiple video devices are not supported at the moment");
		}
	}
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

