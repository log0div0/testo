
#pragma once

#include "../Environment.hpp"
#include <qemu/Connect.hpp>

struct QemuEnvironment : public Environment {
	fs::path testo_dir() const override {
		return "/var/lib/libvirt/testo";
	}
	fs::path flash_drives_mount_dir() const override {
		return "/var/lib/libvirt/testo/flash_drives/mount_point/";
	}
	fs::path flash_drives_img_dir() const override {
		throw std::runtime_error("Not needed");
	}
	fs::path metadata_dir() const override {
		return "/var/lib/libvirt/testo/metadata";
	}

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
