
#pragma once

#include "../FlashDriveController.hpp"

struct HyperVFlashDriveController: FlashDriveController {
	HyperVFlashDriveController() = delete;
	HyperVFlashDriveController(const nlohmann::json& config);
	~HyperVFlashDriveController() override;

	void create() override;
	bool is_mounted() const override;
	void mount() const override;
	void umount() const override;
	fs::path img_path() const override;
	fs::path mount_dir() const override;
};
