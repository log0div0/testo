
#pragma once

#include "FlashDriveController.hpp"
#include <nlohmann/json.hpp>

struct VmController {
	virtual ~VmController() = default;
	virtual void install() = 0;
	virtual void make_snapshot(const std::string& snapshot, const std::string& cksum) = 0;
	virtual void set_metadata(const nlohmann::json& metadata) = 0;
	virtual void set_metadata(const std::string& key, const std::string& value) = 0;
	virtual nlohmann::json get_config() const = 0;
	virtual std::string get_metadata(const std::string& key) = 0;
	virtual std::string get_snapshot_cksum(const std::string& snapshot) = 0;
	virtual void rollback(const std::string& snapshot) = 0;
	virtual void press(const std::vector<std::string>& buttons) = 0;
	virtual bool is_nic_plugged(const std::string& nic) const = 0;
	virtual void set_nic(const std::string& nic, bool is_enabled) = 0;
	virtual bool is_link_plugged(const std::string& nic) const = 0;
	virtual void set_link(const std::string& nic, bool is_connected) = 0;
	virtual void plug_flash_drive(std::shared_ptr<FlashDriveController> fd) = 0;
	virtual void unplug_flash_drive(std::shared_ptr<FlashDriveController> fd) = 0;
	virtual bool is_dvd_plugged() const = 0;
	virtual void plug_dvd(fs::path path) = 0;
	virtual void unplug_dvd() = 0;
	virtual void start() = 0;
	virtual void stop() = 0;
	virtual void type(const std::string& text) = 0;
	virtual bool wait(const std::string& text, const nlohmann::json& params, const std::string& time) = 0;
	virtual bool check(const std::string& text, const nlohmann::json& params) = 0;
	virtual int run(const fs::path& exe, std::vector<std::string> args) = 0;

	virtual bool is_flash_plugged(std::shared_ptr<FlashDriveController> fd) = 0;
	virtual bool has_snapshot(const std::string& snapshot) = 0;
	virtual std::vector<std::string> keys() = 0;
	virtual bool has_key(const std::string& key) = 0;
	virtual bool is_defined() const = 0;
	virtual bool is_running() = 0;
	virtual bool is_additions_installed() = 0;

	virtual std::string name() const = 0;

	virtual void copy_to_guest(const fs::path& src, const fs::path& dst) = 0;
	virtual void copy_from_guest(const fs::path& src, const fs::path& dst) = 0;
	virtual void remove_from_guest(const fs::path& obj) = 0;

	virtual std::set<std::string> nics() const = 0;
};
