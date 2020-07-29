
#pragma once

#include "VM.hpp"
#include "FlashDrive.hpp"
#include "Network.hpp"

struct Environment {
	virtual ~Environment() = default;

	virtual fs::path testo_dir() const = 0;
	virtual fs::path flash_drives_mount_dir() const = 0;
	virtual fs::path flash_drives_img_dir() const = 0;
	virtual fs::path vm_metadata_dir() const = 0;
	virtual fs::path network_metadata_dir() const = 0;
	virtual fs::path flash_drives_metadata_dir() const = 0;

	virtual void setup() = 0;
	virtual void cleanup() = 0;

	virtual std::shared_ptr<VM> create_vm(const nlohmann::json& config) = 0;
	virtual std::shared_ptr<FlashDrive> create_flash_drive(const nlohmann::json& config) = 0;
	virtual std::shared_ptr<Network> create_network(const nlohmann::json& config) = 0;
};

extern std::shared_ptr<Environment> env;
