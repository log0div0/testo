
#pragma once

#include "pugixml/pugixml.hpp"
#include "VmController.hpp"
#include "qemu/Host.hpp"

struct QemuVmController: public VmController {
	QemuVmController() = delete;
	QemuVmController(const nlohmann::json& config);
	QemuVmController(const QemuVmController& other) = delete;
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
	void prepare_networks();
	void remove_disks(const pugi::xml_document& config);
	void create_disks();

	void delete_snapshot_with_children(vir::Snapshot& snapshot);

	std::vector<std::string> keys(vir::Snapshot& snapshot);

	std::string get_dvd_path();
	std::string get_dvd_path(vir::Snapshot& snapshot);

	bool is_link_plugged(const pugi::xml_node& devices, const std::string& nic) const;
	bool is_link_plugged(vir::Snapshot& snapshot, const std::string& nic);

	bool is_nic_plugged(vir::Snapshot& snapshot, const std::string& nic);

	void attach_nic(const std::string& nic);
	void detach_nic(const std::string& nic);

	std::string get_flash_img();
	std::string get_flash_img(vir::Snapshot& snapshot);
	void attach_flash_drive(const std::string& img_path);
	void detach_flash_drive();

	nlohmann::json config;
	vir::Connect qemu_connect;
	std::unordered_map<std::string, uint32_t> scancodes;
	std::unordered_map<char, std::vector<std::string>> charmap;
};
