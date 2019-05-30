
#pragma once

#include "../FlashDriveController.hpp"
#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>

struct VboxFlashDriveController: FlashDriveController {
	VboxFlashDriveController() = delete;
	VboxFlashDriveController(const VboxFlashDriveController& other) = delete;
	VboxFlashDriveController(const nlohmann::json& config);
	void create() override;
	bool is_mounted() const override;
	void mount() const override;
	void umount() const override;

	fs::path img_path() const override;
	fs::path mount_dir() const override;

private:
	void remove_if_exists();

	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
};
