
#pragma once

#include "HyperVVM.hpp"
#include "HyperVFlashDrive.hpp"
#include "HyperVNetwork.hpp"
#include "../Environment.hpp"

struct HyperVEnvironment: Environment {
	HyperVEnvironment(const nlohmann::json& config): Environment(config) {}
	~HyperVEnvironment() {}

	fs::path testo_dir() const override;
	fs::path flash_drives_mount_dir() const override;
	fs::path flash_drives_img_dir() const override;
	fs::path vm_metadata_dir() const override;
	fs::path network_metadata_dir() const override;
	fs::path flash_drives_metadata_dir() const override;

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) override {
		return std::make_shared<VmController>(std::shared_ptr<VM>(new HyperVVM(config)));
	}

	std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) override {
		return std::make_shared<FlashDriveController>(std::shared_ptr<FlashDrive>(new HyperVFlashDrive(config)));
	}

	std::shared_ptr<NetworkController> create_network_controller(const nlohmann::json& config) {
		return std::make_shared<NetworkController>(std::shared_ptr<Network>(new HyperVNetwork(config)));
	}
};
