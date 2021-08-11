
#pragma once

#include "../NNServiceClient.hpp"

#include "VM.hpp"
#include "FlashDrive.hpp"
#include "Network.hpp"

struct EnvironmentConfig {
	
	std::string nn_service_ip() const;
	std::string nn_service_port() const;

	void validate() const;
	virtual void dump(nlohmann::json& j) const;

	std::string nn_service_endpoint = "127.0.0.1:8156";
};

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

	virtual void setup(const EnvironmentConfig& config);
	virtual std::string hypervisor() const = 0;

	virtual std::shared_ptr<VM> create_vm(const nlohmann::json& config) = 0;
	virtual std::shared_ptr<FlashDrive> create_flash_drive(const nlohmann::json& config) = 0;
	virtual std::shared_ptr<Network> create_network(const nlohmann::json& config) = 0;

	virtual void validate_vm_config(const nlohmann::json& config) = 0;
	virtual void validate_flash_drive_config(const nlohmann::json& config) = 0;
	virtual void validate_network_config(const nlohmann::json& config) = 0;

	NNServiceClient nn_client;
};

extern std::shared_ptr<Environment> env;
