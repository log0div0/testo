
#pragma once

#include "../Environment.hpp"
#include "QemuVmController.hpp"
#include "QemuFlashDriveController.hpp"

struct QemuEnvironment : public Environment {
	QemuEnvironment() {}
	~QemuEnvironment();

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) override {
		return std::shared_ptr<VmController>(new QemuVmController(config));
	}

	std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) override {
		return std::shared_ptr<FlashDriveController>(new QemuFlashDriveController(config));
	}

private:
	void prepare_storage_pool(const std::string& pool_name);
	vir::Connect qemu_connect;
};
