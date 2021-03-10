
#include "SharedFolder.hpp"

#include <os/Process.hpp>
#include <os/File.hpp>

#include <regex>

fs::path get_config_path() {
#ifdef __linux__
	return "/etc/testo-guest-additions.conf";
#else
	throw std::runtime_error("Not implemented");
#endif
}

nlohmann::json load_config() {
	if (!fs::exists(get_config_path())) {
		return nlohmann::json::object();
	}
	std::vector<uint8_t> data = os::File::open_for_read(get_config_path()).read_all();
	return nlohmann::json::parse((char*)&data[0], (char*)&data[0] + data.size());
}

void save_config(const nlohmann::json& config) {
	std::string data = config.dump(4);
	os::File::open_for_write(get_config_path()).write((uint8_t*)data.data(), data.size());
#ifdef __linux__
	sync();
#endif
}

void register_shared_folder(const std::string& folder_name, const fs::path& guest_path) {
	nlohmann::json config = load_config();
	if (config.count("permanent_shared_folders")) {
		for (auto& shared_folder: config.at("permanent_shared_folders")) {
			if (shared_folder.at("name") == folder_name) {
				shared_folder["guest_path"] = guest_path;
				save_config(config);
				return;
			}
		}
	}
	config["permanent_shared_folders"].push_back({
		{"name", folder_name},
		{"guest_path", guest_path.string()}
	});
	save_config(config);
}

void unregister_shared_folder(const std::string& folder_name) {
	nlohmann::json config = load_config();
	if (config.count("permanent_shared_folders")) {
		auto& list = config.at("permanent_shared_folders");
		for (auto it = list.begin(); it != list.end(); ++it) {
			if ((*it).at("name") == folder_name) {
				list.erase(it);
				save_config(config);
				return;
			}
		}
	}
}

std::vector<std::string> split_string_by_newline(const std::string& str)
{
	auto result = std::vector<std::string>{};
	auto ss = std::stringstream{str};

	for (std::string line; std::getline(ss, line, '\n');)
		result.push_back(line);

	return result;
}

nlohmann::json get_shared_folder_status(const std::string& folder_name) {
	nlohmann::json result = {
		{"name", folder_name},
		{"is_mounted", false}
	};
#if defined(__QEMU__) && defined(__linux__)
	std::regex re(folder_name + " (.+?) 9p .+");
	std::smatch match;
	std::string output = os::Process::exec("cat /proc/mounts");
	for (auto& line: split_string_by_newline(output)) {
		if (std::regex_match(line, match, re)) {
			result["is_mounted"] = true;
			result["guest_path"] = match[1];
			break;
		}
	}
#else
	throw std::runtime_error("Sorry, shared folders are not supported on this combination of the hypervisor and the operating system");
#endif
	return result;
}

bool mount_shared_folder(const std::string& folder_name, const fs::path& guest_path) {
	if (guest_path.is_relative()) {
		throw std::runtime_error("Guest path must be absolute");
	}

	if (fs::exists(guest_path)) {
		if (!fs::is_directory(guest_path)) {
			throw std::runtime_error("Path " + guest_path.string() + " is not a directory");
		}
	} else {
		fs::create_directories(guest_path);
	}

	auto status = get_shared_folder_status(folder_name);
	if (!status.at("is_mounted")) {
#if defined(__QEMU__) && defined(__linux__)
		os::Process::exec("modprobe 9p");
		os::Process::exec("modprobe virtio");
		os::Process::exec("modprobe 9pnet");
		os::Process::exec("modprobe 9pnet_virtio");
		os::Process::exec("mount " + folder_name + " \"" + guest_path.string() + "\" -t 9p -o trans=virtio");
#else
		throw std::runtime_error("Sorry, shared folders are not supported on this combination of the hypervisor and the operating system");
#endif
		return true;
	} else {
		return false;
	}
}

bool umount_shared_folder(const std::string& folder_name) {
	auto status = get_shared_folder_status(folder_name);
	if (status.at("is_mounted")) {
#if defined(__QEMU__) && defined(__linux__)
		os::Process::exec("umount " + folder_name);
#else
		throw std::runtime_error("Sorry, shared folders are not supported on this combination of the hypervisor and the operating system");
#endif
		return true;
	} else {
		return false;
	}
}

void mount_permanent_shared_folders() {
	nlohmann::json config = load_config();
	if (config.count("permanent_shared_folders")) {
		for (auto& shared_folder: config.at("permanent_shared_folders")) {
			mount_shared_folder(shared_folder.at("name"), shared_folder.at("guest_path").get<std::string>());
		}
	}
}
