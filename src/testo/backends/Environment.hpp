
#pragma once

#include "VmController.hpp"
#include "FlashDriveController.hpp"
#include "NetworkController.hpp"

struct Environment {
	Environment(const nlohmann::json& config) {
		_content_cksum_maxsize = config.at("content_cksum_maxsize").get<uint32_t>() * 1048576;
	}
	
	virtual ~Environment() = default;

	virtual fs::path testo_dir() const = 0;
	virtual fs::path flash_drives_mount_dir() const = 0;
	virtual fs::path flash_drives_img_dir() const = 0;
	virtual fs::path vm_metadata_dir() const = 0;
	virtual fs::path network_metadata_dir() const = 0;
	virtual fs::path flash_drives_metadata_dir() const = 0;

	uint64_t content_cksum_maxsize() const {
		return _content_cksum_maxsize;
	}

	virtual void setup() = 0;
	virtual void cleanup() = 0;

	virtual std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) = 0;
	virtual std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) = 0;
	virtual std::shared_ptr<NetworkController> create_network_controller(const nlohmann::json& config) = 0;

private:
	uint64_t _content_cksum_maxsize;
};

extern std::shared_ptr<Environment> env;
