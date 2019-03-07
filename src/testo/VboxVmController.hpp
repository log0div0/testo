
#pragma once

#include "VmController.hpp"
#include "API.hpp"
#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>

struct VboxVmController: public VmController {
	VboxVmController() = delete;
	VboxVmController(const nlohmann::json& config);
	VboxVmController(const VboxVmController& other) = delete;
	int install() override;
	int make_snapshot(const std::string& snapshot, const std::string& cksum) override;
	int set_metadata(const nlohmann::json& metadata) override;
	int set_metadata(const std::string& key, const std::string& value) override;

	nlohmann::json get_config() const override {
		return config;
	}

	std::string get_metadata(const std::string& key) override;
	std::string get_snapshot_cksum(const std::string& snapshot) override;
	int rollback(const std::string& snapshot) override;
	int press(const std::vector<std::string>& buttons) override;
	bool is_nic_plugged(const std::string& nic) const override;
	int set_nic(const std::string& nic, bool is_enabled) override;
	bool is_link_plugged(const std::string& nic) const override;
	int set_link(const std::string& nic, bool is_connected) override;
	int plug_flash_drive(std::shared_ptr<FlashDriveController> fd) override;
	int unplug_flash_drive(std::shared_ptr<FlashDriveController> fd) override;
	bool is_dvd_plugged() const override;
	int plug_dvd(fs::path path) override;
	int unplug_dvd() override;
	int start() override;
	int stop() override;
	int type(const std::string& text) override;
	int wait(const std::string& text, const std::string& time) override;
	int run(const fs::path& exe, std::vector<std::string> args) override;

	bool is_flash_plugged(std::shared_ptr<FlashDriveController> fd) override;
	bool has_snapshot(const std::string& snapshot) override;
	std::vector<std::string> keys() override;
	bool has_key(const std::string& key) override;
	bool is_defined() const override;
	bool is_running() override;
	bool is_additions_installed() override;

	std::string name() const override {
		return config.at("name").get<std::string>();
	}

	int copy_to_guest(const fs::path& src, const fs::path& dst) override;
	int remove_from_guest(const fs::path& obj) override;

	std::set<std::string> nics() const override;

private:
	void copy_dir_to_guest(const fs::path& src, const fs::path& dst, vbox::GuestSession& gsession);
	void delete_snapshot_with_children(vbox::Snapshot& snapshot);
	void remove_if_exists();
	void create_vm();

	int set_snapshot_cksum(const std::string& snapshot, const std::string& cksum);

	nlohmann::json config;
	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
	vbox::Session start_session;
	vbox::Session work_session;
	std::unordered_map<char, std::vector<std::string>> charmap;

	std::set<std::shared_ptr<FlashDriveController>> plugged_fds;

	API& api;
};
