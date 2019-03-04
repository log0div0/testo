
#pragma once

#include "FlashDriveController.hpp"
#include <nlohmann/json.hpp>

struct VmController {
	virtual ~VmController() = default;
	virtual int install() = 0;
	virtual int make_snapshot(const std::string& snapshot, const std::string& cksum) = 0;
	virtual int set_metadata(const nlohmann::json& metadata) = 0;
	virtual int set_metadata(const std::string& key, const std::string& value) = 0;
	virtual nlohmann::json get_config() const = 0;
	virtual std::string get_metadata(const std::string& key) = 0;
	virtual std::string get_snapshot_cksum(const std::string& snapshot) = 0;
	virtual int rollback(const std::string& snapshot) = 0;
	virtual int press(const std::vector<std::string>& buttons) = 0;
	virtual int set_nic(const std::string& nic, bool is_enabled) = 0;
	virtual bool is_link_plugged(const std::string& nic) const = 0;
	virtual int set_link(const std::string& nic, bool is_connected) = 0;
	virtual int plug_flash_drive(std::shared_ptr<FlashDriveController> fd) = 0;
	virtual int unplug_flash_drive(std::shared_ptr<FlashDriveController> fd) = 0;
	virtual void unplug_all_flash_drives() = 0;
	virtual bool is_dvd_plugged() const = 0;
	virtual int plug_dvd(fs::path path) = 0;
	virtual int unplug_dvd() = 0;
	virtual int start() = 0;
	virtual int stop() = 0;
	virtual int type(const std::string& text) = 0;
	virtual int wait(const std::string& text, const std::string& time) = 0;
	virtual int run(const fs::path& exe, std::vector<std::string> args) = 0;

	virtual bool is_plugged(std::shared_ptr<FlashDriveController> fd) = 0;
	virtual bool has_snapshot(const std::string& snapshot) = 0;
	virtual std::vector<std::string> keys() = 0;
	virtual bool has_key(const std::string& key) = 0;
	virtual bool is_defined() const = 0;
	virtual bool is_running() = 0;
	virtual bool is_additions_installed() = 0;

	virtual std::string name() const = 0;

	virtual int copy_to_guest(const fs::path& src, const fs::path& dst) = 0;
	virtual int remove_from_guest(const fs::path& obj) = 0;

	virtual std::set<std::string> nics() const = 0;
};
