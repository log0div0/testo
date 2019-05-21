
#pragma once

#include "FlashDriveController.hpp"
#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>


struct VboxFlashDriveController: FlashDriveController {
	VboxFlashDriveController() = delete;
	VboxFlashDriveController(const VboxFlashDriveController& other) = delete;
	VboxFlashDriveController(const nlohmann::json& config);
	void create() override;
	bool is_mounted() const override;
	void mount() const override;
	void umount() const override;
	void load_folder() const override;

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

	nlohmann::json get_config() const override {
		return config;
	}

	std::string cksum() const {
		return "";
	}


	bool cache_enabled() const {
		return false;
	}

	vbox::Medium handle;
private:


	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
	nlohmann::json config;
	//API& api;
};
