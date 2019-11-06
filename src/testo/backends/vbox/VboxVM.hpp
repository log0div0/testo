
#pragma once

#include "../VM.hpp"
#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>

struct VboxVM: public VM {
	VboxVM() = delete;
	VboxVM(const nlohmann::json& config);
	VboxVM(const VboxVM& other) = delete;
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
	void power_button() override;
	void suspend() override;
	void resume() override;
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

private:
	void copy_dir_to_guest(const fs::path& src, const fs::path& dst, vbox::GuestSession& gsession);
	void delete_snapshot_with_children(vbox::Snapshot& snapshot);
	void remove_if_exists();
	void create_vm();
	void wait_state(std::initializer_list<MachineState> states);

	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
	vbox::Session start_session;
	vbox::Session work_session;

	std::set<std::shared_ptr<FlashDrive>> plugged_fds;
	std::unordered_map<std::string, std::vector<uint8_t>> scancodes;
};
