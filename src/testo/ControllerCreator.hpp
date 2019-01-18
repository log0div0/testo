
#pragma once

#include "VboxVmController.hpp"
#include "VboxFlashDriveController.hpp"

struct ControllerCreator {
	virtual VmController* create_vm_controller(const nlohmann::json& config) = 0;
	virtual FlashDriveController* create_flash_drive_controller(const nlohmann::json& config) = 0;
	virtual ~ControllerCreator() = default;
};

struct VboxControllerCreator: public ControllerCreator {
	VboxVmController* create_vm_controller(const nlohmann::json& config) override {
		return new VboxVmController(config);
	}

	VboxFlashDriveController* create_flash_drive_controller(const nlohmann::json& config) override {
		return new VboxFlashDriveController(config);
	}
};
