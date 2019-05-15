
#pragma once

#include "VboxVmController.hpp"
#include "VboxFlashDriveController.hpp"
#ifndef WIN32
#include "QemuVmController.hpp"
#include "QemuFlashDriveController.hpp"
#endif
#include "Register.hpp"

struct Environment {
	virtual ~Environment() = default;

	virtual void setup() = 0;
	virtual void cleanup() = 0;

	virtual std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) = 0;
	virtual std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) = 0;
};

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

#ifdef WIN32
struct HypervEnvironment: public Environment {
	HypervEnvironment() {}
	~HypervEnvironment();

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) override;
	std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) override;
};
#else
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
#endif
