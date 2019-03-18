
#pragma once

#include "FlashDriveController.hpp"
#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>


struct VboxFlashDriveController: FlashDriveController {
	VboxFlashDriveController() = delete;
	VboxFlashDriveController(const VboxFlashDriveController& other) = delete;
	VboxFlashDriveController(const nlohmann::json& config);
	int create() override;
	bool is_mounted() const override;
	int mount() const override;
	int umount() const override;
	int load_folder() const override;

	fs::path img_path() const override {
		auto res = flash_drives_img_dir();
		res += name() + ".vmdk";
		return res;
	}

	std::string name() const override {
		return config.at("name").get<std::string>();
	}

	bool has_folder() const override {
		return config.count("folder");
	}

	vbox::Medium handle;
private:


	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
	nlohmann::json config;
	//API& api;
};
