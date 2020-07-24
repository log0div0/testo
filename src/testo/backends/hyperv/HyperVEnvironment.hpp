
#pragma once

#include "HyperVVM.hpp"
#include "HyperVFlashDrive.hpp"
#include "HyperVNetwork.hpp"
#include "../Environment.hpp"

struct HyperVEnvironment: Environment {
	HyperVEnvironment() {}
	~HyperVEnvironment() {}

	fs::path testo_dir() const override;
	fs::path flash_drives_mount_dir() const override;
	fs::path flash_drives_img_dir() const override;
	fs::path vm_metadata_dir() const override;
	fs::path network_metadata_dir() const override;
	fs::path flash_drives_metadata_dir() const override;

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VM> create_vm(const nlohmann::json& config) override {
		return std::shared_ptr<VM>(new HyperVVM(config));
	}

	std::shared_ptr<FlashDrive> create_flash_drive(const nlohmann::json& config) override {
		return std::shared_ptr<FlashDrive>(new HyperVFlashDrive(config));
	}

	std::shared_ptr<Network> create_network(const nlohmann::json& config) override {
		return std::shared_ptr<Network>(new HyperVNetwork(config));
	}
};
