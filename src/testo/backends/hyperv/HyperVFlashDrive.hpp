
#pragma once

#include "../FlashDrive.hpp"

struct HyperVFlashDrive: FlashDrive {
	HyperVFlashDrive() = delete;
	HyperVFlashDrive(const nlohmann::json& config);
	~HyperVFlashDrive() override;

	void create() override;
	bool is_mounted() const override;
	void mount() const override;
	void umount() const override;
	fs::path img_path() const override;
	fs::path mount_dir() const override;
};
