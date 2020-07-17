
#pragma once

#include <pugixml/pugixml.hpp>
#include "../VM.hpp"
#include <qemu/Host.hpp>
#include <unordered_map>

struct QemuVM: public VM {
	QemuVM() = delete;
	QemuVM(const nlohmann::json& config);
	~QemuVM();
	QemuVM(const QemuVM& other) = delete;
	void install() override;
	void undefine() override;
	void remove_disks() override;
	void make_snapshot(const std::string& snapshot) override;

	void rollback(const std::string& snapshot) override;
	void press(const std::vector<std::string>& buttons) override;
	void hold(const std::vector<std::string>& buttons) override;
	void release(const std::vector<std::string>& buttons) override;
	void mouse_move_abs(uint32_t x, uint32_t y) override;
	void mouse_move_abs(const std::string& axis, uint32_t value) override;
	void mouse_move_rel(int x, int y) override;
	void mouse_move_rel(const std::string& axis, int value) override;
	void mouse_hold(const std::vector<MouseButton>& buttons) override;
	void mouse_release(const std::vector<MouseButton>& buttons) override;
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
	int run(const fs::path& exe, std::vector<std::string> args, uint32_t timeout_milliseconds,
		const std::function<void(const std::string&)>& callback) override;

	bool is_flash_plugged(std::shared_ptr<FlashDrive> fd) override;
	bool has_snapshot(const std::string& snapshot) override;
	void delete_snapshot(const std::string& snapshot) override;
	bool is_defined() const override;
	VmState state() const override;
	bool is_additions_installed() override;

	void copy_to_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds) override;
	void copy_from_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds) override;
	void remove_from_guest(const fs::path& obj) override;
	std::string get_tmp_dir() override;

private:
	void import_disk(const std::string& name, const fs::path& source);
	void create_new_disk(const std::string& name, uint32_t size);
	void create_disks();

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

	std::string preferable_video_model();
	std::string mouse_button_to_str(MouseButton btn);

	vir::Connect qemu_connect;
	std::unordered_map<std::string, uint32_t> scancodes;
	std::vector<std::string> disk_targets; //10 + a cdrom
	std::vector<uint8_t> screenshot_buffer;

};
