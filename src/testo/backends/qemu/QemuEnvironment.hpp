
#pragma once

#include "../Environment.hpp"
#include <qemu/Connect.hpp>

struct QemuEnvironment : public Environment {
	fs::path testo_dir() const override {
		return "/var/lib/libvirt/testo";
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
