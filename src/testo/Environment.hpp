
#pragma once

#include "VboxVmController.hpp"
#include "VboxFlashDriveController.hpp"
#include "Register.hpp"

struct Environment {
	virtual ~Environment() = default;

	virtual void setup() = 0;
	virtual void cleanup() = 0;

	virtual std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) = 0;
	virtual std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) = 0;
};

struct VboxEnvironment: public Environment {
	VboxEnvironment(): api(API::instance()) {}
	~VboxEnvironment();

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) override {
		return std::shared_ptr<VmController>(new VboxVmController(config));
	}

	std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) override {
		return std::shared_ptr<VboxFlashDriveController>(new VboxFlashDriveController(config));
	}

	API& api;
};
