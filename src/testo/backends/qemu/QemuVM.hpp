
#pragma once

#include <pugixml/pugixml.hpp>
#include "../VM.hpp"
#include <qemu/Host.hpp>

struct QemuVM: public VM {
	QemuVM() = delete;
	QemuVM(const nlohmann::json& config);
	~QemuVM();
	QemuVM(const QemuVM& other) = delete;
	void install() override;
	void undefine() override;
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
	void plug_dvd(IsoId iso) override;
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
	std::string find_free_guest_additions_port() const;

	void remove_disk();
	void create_disk();

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

	fs::path upload_iso(const fs::path& iso_path);

	fs::path iso_path;

	vir::Connect qemu_connect;
	std::unordered_map<std::string, uint32_t> scancodes;
	std::vector<uint8_t> screenshot_buffer;
};
