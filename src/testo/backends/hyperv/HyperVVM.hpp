
#pragma once

#include "../VM.hpp"
#include <hyperv/Connect.hpp>

struct HyperVVM: VM {
	HyperVVM() = delete;
	HyperVVM(const nlohmann::json& config);
	~HyperVVM() override;
	void install() override;
	void make_snapshot(const std::string& snapshot) override;
	void rollback(const std::string& snapshot) override;
	void press(const std::vector<std::string>& buttons) override;
	void mouse_move(const std::string& x, const std::string& y) override;
	void mouse_set_buttons(uint32_t button_mask) override;
	bool is_nic_plugged(const std::string& nic) const override;
	void set_nic(const std::string& nic, bool is_enabled) override;
	bool is_link_plugged(const std::string& nic) const override;
	void set_link(const std::string& nic, bool is_connected) override;
	void plug_flash_drive(std::shared_ptr<FlashDrive> fd) override;
	void unplug_flash_drive(std::shared_ptr<FlashDrive> fd) override;
	bool is_dvd_plugged() const override;
	void plug_dvd(fs::path path) override;
	void unplug_dvd() override;
	void start() override;
	void stop() override;
	void suspend() override;
	void resume() override;
	void power_button() override;
	stb::Image screenshot() override;
	int run(const fs::path& exe, std::vector<std::string> args, uint32_t timeout_milliseconds) override;

	bool is_flash_plugged(std::shared_ptr<FlashDrive> fd) override;
	bool has_snapshot(const std::string& snapshot) override;
	void delete_snapshot(const std::string& snapshot) override;
	bool is_defined() const override;
	VmState state() const override;
	bool is_additions_installed() override;

	void copy_to_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds) override;
	void copy_from_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds) override;
	void remove_from_guest(const fs::path& obj) override;

	std::set<std::string> nics() const override;

private:
	hyperv::Connect connect;
	std::unordered_map<std::string, std::vector<uint8_t>> scancodes;
};
