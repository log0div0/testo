
#pragma once

#include "../FlashDrive.hpp"

struct HyperVFlashDrive: FlashDrive {
	HyperVFlashDrive() = delete;
	HyperVFlashDrive(const nlohmann::json& config);
	~HyperVFlashDrive() override;

	bool is_defined() override;
	void create() override;
	void undefine() override;
	void load_folder(const fs::path& folder) override;
	bool has_snapshot(const std::string& snapshot) override;
	void make_snapshot(const std::string& snapshot) override;
	void delete_snapshot(const std::string& snapshot) override;
	void rollback(const std::string& snapshot) override;
	fs::path img_path() const override;
};
