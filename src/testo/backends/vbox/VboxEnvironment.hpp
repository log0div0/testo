
#pragma once

#include "../Environment.hpp"
#include <vbox/api.hpp>

struct VboxEnvironment: public Environment {
	fs::path testo_dir() const override {
		throw std::runtime_error("Implement me");
	}
	fs::path flash_drives_mount_dir() const override {
		return _flash_drives_mount_dir;
	}
	fs::path flash_drives_img_dir() const override {
		return _flash_drives_img_dir;
	}
	fs::path vm_metadata_dir() const override {
		throw std::runtime_error("Implement me");
	}
	fs::path network_metadata_dir() const override {
		throw std::runtime_error("Implement me");
	}
	fs::path flash_drives_metadata_dir() const override {
		throw std::runtime_error("Implement me");
	}

	VboxEnvironment(const nlohmann::json& config);
	~VboxEnvironment();

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VM> create_vm(const nlohmann::json& config) override;
	std::shared_ptr<FlashDrive> create_flash_drive(const nlohmann::json& config) override;
	std::shared_ptr<Network> create_network(const nlohmann::json& config) override;

	vbox::API api;

private:
	fs::path _flash_drives_img_dir;
	fs::path _flash_drives_mount_dir;
};
