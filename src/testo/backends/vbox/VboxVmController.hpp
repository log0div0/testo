
#pragma once

#include "../VmController.hpp"
#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>

struct VboxVmController: public VmController {
	VboxVmController() = delete;
	VboxVmController(const nlohmann::json& config);
	VboxVmController(const VboxVmController& other) = delete;
	void install() override;
	void make_snapshot(const std::string& snapshot, const std::string& cksum) override;
	void set_metadata(const nlohmann::json& metadata) override;
	void set_metadata(const std::string& key, const std::string& value) override;

	std::string get_metadata(const std::string& key) override;
	std::string get_snapshot_cksum(const std::string& snapshot) override;
	void rollback(const std::string& snapshot) override;
	void press(const std::vector<std::string>& buttons) override;
	bool is_nic_plugged(const std::string& nic) const override;
	void set_nic(const std::string& nic, bool is_enabled) override;
	bool is_link_plugged(const std::string& nic) const override;
	void set_link(const std::string& nic, bool is_connected) override;
	void plug_flash_drive(std::shared_ptr<FlashDriveController> fd) override;
	void unplug_flash_drive(std::shared_ptr<FlashDriveController> fd) override;
	bool is_dvd_plugged() const override;
	void plug_dvd(fs::path path) override;
	void unplug_dvd() override;
	void start() override;
	void stop() override;
	void shutdown(uint32_t timeout_seconds) override;
	void suspend() override;
	void resume() override;
	stb::Image screenshot() override;
	int run(const fs::path& exe, std::vector<std::string> args, uint32_t timeout_seconds) override;

	bool is_flash_plugged(std::shared_ptr<FlashDriveController> fd) override;
	bool has_snapshot(const std::string& snapshot) override;
	std::vector<std::string> keys() override;
	bool has_key(const std::string& key) override;
	bool is_defined() const override;
	bool is_running() override;
	bool is_additions_installed() override;

	void copy_to_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_seconds) override;
	void copy_from_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_seconds) override;
	void remove_from_guest(const fs::path& obj) override;

	std::set<std::string> nics() const override;

private:
	void copy_dir_to_guest(const fs::path& src, const fs::path& dst, vbox::GuestSession& gsession);
	void delete_snapshot_with_children(vbox::Snapshot& snapshot);
	void remove_if_exists();
	void create_vm();

	void set_snapshot_cksum(const std::string& snapshot, const std::string& cksum);

	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
	vbox::Session start_session;
	vbox::Session work_session;

	std::set<std::shared_ptr<FlashDriveController>> plugged_fds;

	//API& api;
};
