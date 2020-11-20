
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
	None = 0,
	Left = 1,
	Right = 2,
	Middle = 3,
	WheelUp = 4,
	WheelDown = 5
};

struct VM {
	VM() = delete;
	VM(const nlohmann::json& config_);
	virtual ~VM() = default;
	virtual void install() = 0;
	virtual void undefine() = 0;
	virtual void remove_disks() = 0;
	virtual nlohmann::json make_snapshot(const std::string& snapshot) = 0;
	virtual void rollback(const std::string& snapshot, const nlohmann::json& opaque) = 0;
	virtual void press(const std::vector<std::string>& buttons) = 0;
	virtual void hold(const std::vector<std::string>& buttons) = 0;
	virtual void release(const std::vector<std::string>& buttons) = 0;
	virtual void mouse_move_abs(uint32_t x, uint32_t y) = 0;
	virtual void mouse_move_abs(const std::string& axis, uint32_t value) = 0;
	virtual void mouse_move_rel(int x, int y) = 0;
	virtual void mouse_move_rel(const std::string& axis, int value) = 0;
	virtual void mouse_hold(const std::vector<MouseButton>& buttons) = 0;
	virtual void mouse_release(const std::vector<MouseButton>& buttons) = 0;
	virtual bool is_nic_plugged(const std::string& pci_addr) const = 0;
	virtual std::string attach_nic(const std::string& nic) = 0;
	virtual void detach_nic(const std::string& pci_addr) = 0;
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
	virtual stb::Image<stb::RGB> screenshot() = 0;
	virtual int run(const fs::path& exe, std::vector<std::string> args,
		const std::function<void(const std::string&)>& callback) = 0;

	virtual bool is_flash_plugged(std::shared_ptr<FlashDrive> fd) = 0;
	virtual bool has_snapshot(const std::string& snapshot) = 0;
	virtual void delete_snapshot(const std::string& snapshot) = 0;
	virtual bool is_defined() const = 0;
	virtual VmState state() const = 0;
	virtual bool is_additions_installed() = 0;

	virtual void copy_to_guest(const fs::path& src, const fs::path& dst) = 0;
	virtual void copy_from_guest(const fs::path& src, const fs::path& dst) = 0;
	virtual void remove_from_guest(const fs::path& obj) = 0;
	virtual std::string get_tmp_dir() = 0;

	std::set<std::string> nics() const;
	std::set<std::string> networks() const;

	std::string id() const;
	std::string name() const;
	std::string prefix() const;

protected:
	nlohmann::json config;
};
