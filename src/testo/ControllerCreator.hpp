
#pragma once

#include "VmController.hpp"
#include "FlashDriveController.hpp"

struct ControllerCreator {
	VmController* create_vm_controller(const nlohmann::json& config) {
		return new VmController(config);
	}

	FlashDriveController* create_flash_drive_controller(const nlohmann::json& config) {
		return new FlashDriveController(config);
	}
};