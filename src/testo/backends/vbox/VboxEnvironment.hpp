
#pragma once

#include "../Environment.hpp"
#include "VboxVmController.hpp"
#include "VboxFlashDriveController.hpp"

struct VboxEnvironment: public Environment {
	VboxEnvironment() {}
	~VboxEnvironment();

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) override {
		return std::shared_ptr<VmController>(new VboxVmController(config));
	}

	std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) override {
		return std::shared_ptr<FlashDriveController>(new VboxFlashDriveController(config));
	}

	//API& api;
};
