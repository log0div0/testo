
#pragma once

#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>
#include "Utils.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <set>
#include <vector>


struct FlashDriveController {
	FlashDriveController() = default;
	FlashDriveController(const FlashDriveController& other) = delete;
	FlashDriveController(const nlohmann::json& config);
	int create();
	bool is_mounted() const;
	int mount() const;
	int umount() const;
	int load_folder() const;

	std::string name() const {
		return config.at("name").get<std::string>();
	}

	fs::path img_path() const {
		auto res = flash_drives_img_dir();
		res += name() + ".vmdk";
		return res;
	}

	bool has_folder() const {
		return config.count("folder");
	}

	std::string current_vm;
	vbox::Medium handle;
private:
	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
	nlohmann::json config;
};
