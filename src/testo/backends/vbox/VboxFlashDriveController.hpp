
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
	void load_folder() const override;

	fs::path img_path() const override;

	vbox::Medium handle;
private:
	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
};
