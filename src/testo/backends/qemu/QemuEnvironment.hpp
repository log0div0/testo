
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

	QemuEnvironment(const std::string& uri);
	~QemuEnvironment();

	void setup() override;
	void cleanup() override;

	std::string uri() const;
	bool is_local_uri() const;

	fs::path resolve_path(const std::string& volume, const std::string& pool);
	std::string get_last_modify_date(const std::string& volume, const std::string& pool);

	std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) override;
	std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) override;
	std::shared_ptr<NetworkController> create_network_controller(const nlohmann::json& config) override;

private:
	const std::string _uri;
	void prepare_storage_pool(const std::string& pool_name);
	vir::Connect qemu_connect;
};
