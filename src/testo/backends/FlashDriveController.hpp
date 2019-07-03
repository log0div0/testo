
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
	bool has_snapshot(const std::string& snapshot) override;
	std::string get_snapshot_cksum(const std::string& snapshot) override;

	bool has_key(const std::string& key) override;
	std::string get_metadata(const std::string& key) override;
	void set_metadata(const std::string& key, const std::string& value) override;

	bool check_config_relevance() override;

	std::shared_ptr<FlashDrive> fd;
};
