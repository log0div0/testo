
#pragma once

#include "VboxVmController.hpp"
#include "VboxFlashDriveController.hpp"
#include "Register.hpp"

struct Environment {
	virtual ~Environment() = default;

	virtual void setup() = 0;
	virtual void cleanup() = 0;

	virtual VmController* create_vm_controller(const nlohmann::json& config) = 0;
	virtual FlashDriveController* create_flash_drive_controller(const nlohmann::json& config) = 0;
};

struct VboxEnvironment: public Environment {
	VboxEnvironment(): api(API::instance()) {}
	~VboxEnvironment();

	void setup() override;
	void cleanup() override;

	VboxVmController* create_vm_controller(const nlohmann::json& config) override {
		return new VboxVmController(config);
	}

	VboxFlashDriveController* create_flash_drive_controller(const nlohmann::json& config) override {
		return new VboxFlashDriveController(config);
	}

	API& api;
};
