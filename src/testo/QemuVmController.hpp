
#pragma once

#include "pugixml/pugixml.hpp"
#include "VmController.hpp"
#include "qemu/Host.hpp"
#include "StinkingPileOfShit.hpp"

struct QemuVmController: public VmController {
	QemuVmController() = delete;
	QemuVmController(const nlohmann::json& config);
	~QemuVmController();
	QemuVmController(const QemuVmController& other) = delete;
	void install() override;
	void make_snapshot(const std::string& snapshot, const std::string& cksum) override;
	void set_metadata(const nlohmann::json& metadata) override;
	void set_metadata(const std::string& key, const std::string& value) override;

	nlohmann::json get_config() const override {
		return config;
	}

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
	void type(const std::string& text) override;
	bool wait(const std::string& text, const nlohmann::json& params, const std::string& time) override;
	bool check(const std::string& text, const nlohmann::json& params) override;
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

	void copy_to_guest(const fs::path& src, const fs::path& dst) override;
	void copy_from_guest(const fs::path& src, const fs::path& dst) override;
	void remove_from_guest(const fs::path& obj) override;

	std::set<std::string> nics() const override;

private:
	void prepare_networks();
	void remove_disk();
	void create_disk();

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
	void attach_flash_drive(const std::string& img_path);
	void detach_flash_drive();

	nlohmann::json config;
	vir::Connect qemu_connect;
	std::unordered_map<std::string, uint32_t> scancodes;
	std::unordered_map<char, std::vector<std::string>> charmap;
	StinkingPileOfShit shit;
	std::vector<uint8_t> screenshot_buffer;
};
