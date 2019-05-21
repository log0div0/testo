
#pragma once

#include "VmController.hpp"
#include "FlashDriveController.hpp"
#include "../Register.hpp"

struct Environment {
	virtual ~Environment() = default;

	virtual void setup() = 0;
	virtual void cleanup() = 0;

	virtual std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) = 0;
	virtual std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) = 0;
};
