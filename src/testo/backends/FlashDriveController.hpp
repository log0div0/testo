
#pragma once

#include "Controller.hpp"
#include "FlashDrive.hpp"

struct FlashDriveController: public Controller {
	FlashDriveController() = delete;
	FlashDriveController(std::shared_ptr<FlashDrive> fd): Controller(), fd(fd) {}

	std::string name() const override;
	bool is_defined() override;
	void create() override;
	void create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed) override;
	void restore_snapshot(const std::string& snapshot) override;
	void delete_snapshot_with_children(const std::string& snapshot) override;

	bool check_config_relevance() override;

	fs::path get_metadata_dir() const override;

	std::shared_ptr<FlashDrive> fd;
};
