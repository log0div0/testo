
#pragma once

#include "../Environment.hpp"
#include <qemu/Connect.hpp>

struct QemuEnvironment : public Environment {
	static fs::path testo_dir;
	static fs::path flash_drives_mount_dir;
	static fs::path metadata_dir;

	QemuEnvironment();
	~QemuEnvironment();

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) override;
	std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) override;

private:
	void prepare_storage_pool(const std::string& pool_name);
	vir::Connect qemu_connect;
};
