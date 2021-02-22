
#pragma once

#include "VM.hpp"
#include "FlashDrive.hpp"
#include "Network.hpp"

struct Environment {
	virtual ~Environment() = default;

	virtual fs::path testo_dir() const = 0;

	fs::path vm_metadata_dir() const {
		return testo_dir() / "vm_metadata";
	}
	fs::path network_metadata_dir() const {
		return testo_dir() / "network_metadata";
	}
	fs::path flash_drives_metadata_dir() const {
		return testo_dir() / "fd_metadata";
	}

	virtual void setup();
	virtual std::string hypervisor() const = 0;

	virtual std::shared_ptr<VM> create_vm(const nlohmann::json& config) = 0;
	virtual std::shared_ptr<FlashDrive> create_flash_drive(const nlohmann::json& config) = 0;
	virtual std::shared_ptr<Network> create_network(const nlohmann::json& config) = 0;
};

extern std::shared_ptr<Environment> env;
