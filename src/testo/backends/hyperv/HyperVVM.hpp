
#pragma once

#include "../VM.hpp"
#include <hyperv/Connect.hpp>

struct HyperVVM: VM {
	HyperVVM() = delete;
	HyperVVM(const nlohmann::json& config);
	~HyperVVM() override;
	void install() override;
	void undefine() override;
	void remove_disks() override;
	nlohmann::json make_snapshot(const std::string& snapshot) override;
	void rollback(const std::string& snapshot, const nlohmann::json& opaque) override;
	void press(const std::vector<std::string>& buttons) override;
	void hold(const std::vector<std::string>& buttons) override;
	void release(const std::vector<std::string>& buttons) override;
	void mouse_move_abs(uint32_t x, uint32_t y) override;
	void mouse_move_rel(int x, int y) override;
	void mouse_hold(const std::vector<MouseButton>& buttons) override;
	void mouse_release(const std::vector<MouseButton>& buttons) override;
	bool is_nic_plugged(const std::string& nic) const override;
	void plug_nic(const std::string& nic) override;
	void unplug_nic(const std::string& nic) override;
	bool is_link_plugged(const std::string& nic) const override;
	void set_link(const std::string& nic, bool is_connected) override;
	void plug_flash_drive(std::shared_ptr<FlashDrive> fd) override;
	void unplug_flash_drive(std::shared_ptr<FlashDrive> fd) override;
	bool is_hostdev_plugged() override;
	void plug_hostdev_usb(const std::string& addr) override;
	void unplug_hostdev_usb(const std::string& addr) override;
	bool is_dvd_plugged() const override;
	void plug_dvd(fs::path path) override;
	void unplug_dvd() override;
	void start() override;
	void stop() override;
	void suspend() override;
	void resume() override;
	void power_button() override;
	stb::Image<stb::RGB> screenshot() override;

	bool is_flash_plugged(std::shared_ptr<FlashDrive> fd) override;
	bool has_snapshot(const std::string& snapshot) override;
	void delete_snapshot(const std::string& snapshot) override;
	bool is_defined() const override;
	VmState state() const override;

	std::shared_ptr<GuestAdditions> guest_additions() override;

private:
	hyperv::Connect connect;
	std::unordered_map<std::string, std::vector<uint8_t>> scancodes;
};
