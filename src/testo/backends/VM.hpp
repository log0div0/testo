
#pragma once

#include "FlashDrive.hpp"
#include <stb/Image.hpp>
#include <nlohmann/json.hpp>

enum class VmState {
	Stopped,
	Running,
	Suspended,
	Other
};

enum MouseButton {
	Left = 1,
	Right = 2,
	Middle = 4
};

struct VM {
	VM() = delete;
	VM(const nlohmann::json& config_);
	virtual ~VM() = default;
	virtual void install() = 0;
	virtual void undefine() = 0;
	virtual void make_snapshot(const std::string& snapshot) = 0;
	virtual void rollback(const std::string& snapshot) = 0;
	virtual void press(const std::vector<std::string>& buttons) = 0;
	virtual void mouse_move(const std::string& x, const std::string& y) = 0;
	virtual void mouse_set_buttons(uint32_t button_mask) = 0;
	virtual bool is_nic_plugged(const std::string& nic) const = 0;
	virtual void set_nic(const std::string& nic, bool is_enabled) = 0;
	virtual bool is_link_plugged(const std::string& nic) const = 0;
	virtual void set_link(const std::string& nic, bool is_connected) = 0;
	virtual void plug_flash_drive(std::shared_ptr<FlashDrive> fd) = 0;
	virtual void unplug_flash_drive(std::shared_ptr<FlashDrive> fd) = 0;
	virtual bool is_dvd_plugged() const = 0;
	virtual void plug_dvd(fs::path path) = 0;
	virtual void unplug_dvd() = 0;
	virtual void start() = 0;
	virtual void stop() = 0;
	virtual void power_button() = 0;
	virtual void suspend() = 0;
	virtual void resume() = 0;
	virtual stb::Image screenshot() = 0;
	virtual int run(const fs::path& exe, std::vector<std::string> args, uint32_t timeout_milliseconds) = 0;

	virtual bool is_flash_plugged(std::shared_ptr<FlashDrive> fd) = 0;
	virtual bool has_snapshot(const std::string& snapshot) = 0;
	virtual void delete_snapshot(const std::string& snapshot) = 0;
	virtual bool is_defined() const = 0;
	virtual VmState state() const = 0;
	virtual bool is_additions_installed() = 0;

	virtual void copy_to_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds) = 0;
	virtual void copy_from_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds) = 0;
	virtual void remove_from_guest(const fs::path& obj) = 0;

	std::set<std::string> nics() const;
	std::set<std::string> networks() const;

	std::string id() const;
	std::string name() const;
	std::string prefix() const;
	nlohmann::json get_config() const;

protected:
	nlohmann::json config;
};
