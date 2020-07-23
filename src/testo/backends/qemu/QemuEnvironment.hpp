
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
	fs::path vm_metadata_dir() const override {
		return "/var/lib/libvirt/testo/vm_metadata";
	}
	fs::path network_metadata_dir() const override {
		return "/var/lib/libvirt/testo/network_metadata";
	}

	fs::path flash_drives_metadata_dir() const override {
		return "/var/lib/libvirt/testo/fd_metadata";
	}

	QemuEnvironment();
	~QemuEnvironment();

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VM> create_vm(const nlohmann::json& config) override;
	std::shared_ptr<FlashDrive> create_flash_drive(const nlohmann::json& config) override;
	std::shared_ptr<Network> create_network(const nlohmann::json& config) override;

private:
	void prepare_storage_pool(const std::string& pool_name);
	vir::Connect qemu_connect;
};
