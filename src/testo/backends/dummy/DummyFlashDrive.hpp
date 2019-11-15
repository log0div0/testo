
#pragma once

#include "../FlashDrive.hpp"

struct DummyFlashDrive: FlashDrive {
	DummyFlashDrive() = delete;
	DummyFlashDrive(const nlohmann::json& config);
	~DummyFlashDrive() override;

	bool is_defined() override;
	void create() override;
	void undefine() override;
	bool is_mounted() const override;
	void mount() const override;
	void umount() const override;
	bool has_snapshot(const std::string& snapshot) override;
	void make_snapshot(const std::string& snapshot) override;
	void delete_snapshot(const std::string& snapshot) override;
	void rollback(const std::string& snapshot) override;
	fs::path img_path() const override;

private:
	fs::path metadata_file() const {
		fs::path result = "./dummy_hypervisor_files";
		result = result / id();
		return result;
	};
};
