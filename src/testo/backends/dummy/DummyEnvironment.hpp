
#pragma once

#include "DummyVM.hpp"
#include "DummyFlashDrive.hpp"
#include "DummyNetwork.hpp"
#include "../Environment.hpp"

struct DummyEnvironment: Environment {
	DummyEnvironment() {}
	~DummyEnvironment() {}

	fs::path testo_dir() const override;
	fs::path flash_drives_mount_dir() const override;
	fs::path flash_drives_img_dir() const override;
	fs::path vm_metadata_dir() const override;
	fs::path flash_drives_metadata_dir() const override;

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) override {
		return std::make_shared<VmController>(std::shared_ptr<VM>(new DummyVM(config)));
	}

	std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) override {
		return std::make_shared<FlashDriveController>(std::shared_ptr<FlashDrive>(new DummyFlashDrive(config)));
	}

	std::shared_ptr<NetworkController> create_network_controller(const nlohmann::json& config) override {
		return std::make_shared<NetworkController>(std::shared_ptr<Network>(new DummyNetwork(config)));
	}
};
