
#pragma once

#include "../Environment.hpp"
#include <qemu/Connect.hpp>

struct QemuEnvironment : public Environment {

	QemuEnvironment();

	fs::path testo_dir() const override {
		return "/var/lib/libvirt/testo";
	}

	void setup() override;

	std::string hypervisor() const override {
		return "qemu";
	}

	std::shared_ptr<VM> create_vm(const nlohmann::json& config) override;
	std::shared_ptr<FlashDrive> create_flash_drive(const nlohmann::json& config) override;
	std::shared_ptr<Network> create_network(const nlohmann::json& config) override;

	void validate_vm_config(const nlohmann::json& config) override;
	void validate_flash_drive_config(const nlohmann::json& config) override;
	void validate_network_config(const nlohmann::json& config) override;

private:
	void prepare_storage_pool(const std::string& pool_name);
	vir::Connect qemu_connect;
};
