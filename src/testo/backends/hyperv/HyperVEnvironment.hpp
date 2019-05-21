
#pragma once

#include "HyperVVmController.hpp"
#include "HyperVFlashDriveController.hpp"
#include "../Environment.hpp"
#include <hyperv/wmi.hpp>

struct HyperVEnvironment: Environment {
	HyperVEnvironment() {}
	~HyperVEnvironment() {}

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) override {
		return std::shared_ptr<VmController>(new HyperVVmController(config));
	}

	std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) override {
		return std::shared_ptr<FlashDriveController>(new HyperVFlashDriveController(config));
	}

private:
	wmi::CoInitializer initializer;
};
