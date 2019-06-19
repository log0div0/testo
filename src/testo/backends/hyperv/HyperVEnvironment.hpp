
#pragma once

#include "HyperVVM.hpp"
#include "HyperVFlashDriveController.hpp"
#include "../Environment.hpp"

struct HyperVEnvironment: Environment {
	HyperVEnvironment() {}
	~HyperVEnvironment() {}

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VM> create_vm_controller(const nlohmann::json& config) override {
		return std::shared_ptr<VM>(new HyperVVM(config));
	}

	std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) override {
		return std::shared_ptr<FlashDriveController>(new HyperVFlashDriveController(config));
	}
};
