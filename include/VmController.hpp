
#pragma once

#include <FlashDriveController.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <set>
#include <vector>

struct VmController {
	VmController() = default;
	virtual ~VmController() {}
	VmController(const VmController& other) = delete;
	VmController(const nlohmann::json& config);
	virtual int install() = 0;
	virtual int make_snapshot(const std::string& snapshot) = 0;
	virtual int set_config_cksum(const std::string& cksum) = 0;
	virtual std::string get_config_cksum() = 0;
	virtual int set_snapshot_cksum(const std::string& snapshot, const std::string& cksum) = 0;
	virtual std::string get_snapshot_cksum(const std::string& snapshot) = 0;
	virtual int rollback(const std::string& snapshot) = 0;
	virtual int press(const std::vector<std::string>& buttons) = 0;
	virtual int plug_flash_drive(std::shared_ptr<FlashDriveController> fd) = 0;
	virtual int unplug_flash_drive(std::shared_ptr<FlashDriveController> fd) = 0;
	virtual int start() = 0;
	virtual int stop() = 0;
	virtual int type(const std::string& text) = 0;
	virtual int wait(const std::string& text, const std::string& time) = 0;

	virtual bool has_snapshot(const std::string& snapshot) = 0;
	virtual bool is_defined() const = 0;
	virtual bool is_running() = 0;

	std::string config_cksum() const;

	std::string name() const {
		return config.at("name").get<std::string>();
	}

	std::set<std::string> nics() const;
	std::set<std::string> networks() const;

protected:
	nlohmann::json config;
};
