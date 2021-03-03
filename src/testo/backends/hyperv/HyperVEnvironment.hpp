
#pragma once

#include "HyperVVM.hpp"
#include "HyperVFlashDrive.hpp"
#include "HyperVNetwork.hpp"
#include "../Environment.hpp"

struct HyperVEnvironment: Environment {

	fs::path testo_dir() const override;

	void setup() override;

	std::string hypervisor() const override {
		return "hyperv";
	}

	std::shared_ptr<VM> create_vm(const nlohmann::json& config) override {
		return std::shared_ptr<VM>(new HyperVVM(config));
	}

	std::shared_ptr<FlashDrive> create_flash_drive(const nlohmann::json& config) override {
		return std::shared_ptr<FlashDrive>(new HyperVFlashDrive(config));
	}

	std::shared_ptr<Network> create_network(const nlohmann::json& config) override {
		return std::shared_ptr<Network>(new HyperVNetwork(config));
	}

	void validate_vm_config(const nlohmann::json& config) override;
	void validate_flash_drive_config(const nlohmann::json& config) override;
	void validate_network_config(const nlohmann::json& config) override;
};
