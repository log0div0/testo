
#pragma once

#include "HyperVVM.hpp"
#include "HyperVFlashDriveController.hpp"
#include "../Environment.hpp"

struct HyperVEnvironment: Environment {
	HyperVEnvironment() {}
	~HyperVEnvironment() {}

	fs::path testo_dir() const override;
	fs::path flash_drives_mount_dir() const override;
	fs::path flash_drives_img_dir() const override;
	fs::path metadata_dir() const override;

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) override {
		return std::make_shared<VmController>(std::shared_ptr<VM>(new HyperVVM(config)));
	}

	std::shared_ptr<FlashDrive> create_flash_drive_controller(const nlohmann::json& config) override {
		return std::shared_ptr<FlashDrive>(new HyperVFlashDriveController(config));
	}
};
