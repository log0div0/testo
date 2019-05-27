
#pragma once

#include "../Environment.hpp"
#include <vbox/api.hpp>

struct VboxEnvironment: public Environment {
	static fs::path flash_drives_img_dir;
	static fs::path flash_drives_mount_dir;

	VboxEnvironment();
	~VboxEnvironment();

	void setup() override;
	void cleanup() override;

	std::shared_ptr<VmController> create_vm_controller(const nlohmann::json& config) override;
	std::shared_ptr<FlashDriveController> create_flash_drive_controller(const nlohmann::json& config) override;

	vbox::API api;
};
