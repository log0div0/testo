
#pragma once

#include "FlashDriveController.hpp"

struct QemuFlashDriveController: FlashDriveController {
	QemuFlashDriveController() = delete;
	QemuFlashDriveController(const QemuFlashDriveController& other) = delete;
	QemuFlashDriveController(const nlohmann::json& config);
	int create() override;
	bool is_mounted() const override;
	int mount() const override;
	int umount() const override;
	int load_folder() const override;

	std::string name() const override {
		return config.at("name").get<std::string>();
	}

	fs::path img_path() const override {
		auto res = flash_drives_img_dir();
		res += name() + ".vmdk";
		return res;
	}

	bool has_folder() const override {
		return config.count("folder");
	}

private:
	nlohmann::json config;
};
