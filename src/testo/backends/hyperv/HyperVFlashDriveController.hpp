
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
	void load_folder() const override;

	fs::path img_path() const override;
	std::string name() const override;
	nlohmann::json get_config() const override;
	bool has_folder() const override;
	std::string cksum() const override;
	bool cache_enabled() const override;
};
