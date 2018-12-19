
#pragma once

#include <FlashDriveController.hpp>
#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>
#include <nlohmann/json.hpp>

struct VmController {
	VmController() = default;
	VmController(const nlohmann::json& config);
	int install();
	int make_snapshot(const std::string& snapshot);
	int set_config_cksum(const std::string& cksum);
	std::string get_config_cksum();
	int set_snapshot_cksum(const std::string& snapshot, const std::string& cksum);
	std::string get_snapshot_cksum(const std::string& snapshot);
	int rollback(const std::string& snapshot);
	int press(const std::vector<std::string>& buttons);
	int set_nic(const std::string& nic, bool is_enabled);
	int set_link(const std::string& nic, bool is_connected);
	int plug_flash_drive(std::shared_ptr<FlashDriveController> fd);
	int unplug_flash_drive(std::shared_ptr<FlashDriveController> fd);
	int plug_dvd(fs::path path);
	int unplug_dvd();
	int start();
	int stop();
	int type(const std::string& text);
	int wait(const std::string& text, const std::string& time);
	int run(const fs::path& exe, std::vector<std::string> args);

	bool is_plugged(std::shared_ptr<FlashDriveController> fd);
	bool has_snapshot(const std::string& snapshot);
	bool is_defined() const;
	bool is_running();
	bool is_additions_installed();

	std::string config_cksum() const;
	std::string name() const {
		return config.at("name").get<std::string>();
	}

	int copy_to_guest(const fs::path& src, const fs::path& dst);
	int remove_from_guest(const fs::path& obj);

	std::set<std::string> nics() const;
	std::set<std::shared_ptr<FlashDriveController>> plugged_fds;
	nlohmann::json attrs;

private:
	void copy_dir_to_guest(const fs::path& src, const fs::path& dst, vbox::GuestSession& gsession);
	void delete_snapshot_with_children(vbox::Snapshot& snapshot);
	void remove_if_exists();
	void create_vm();

	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
	vbox::Session start_session;
	vbox::Session work_session;
	std::unordered_map<char, std::vector<std::string>> charmap;

	nlohmann::json config;
};
