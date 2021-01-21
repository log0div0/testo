
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
	nlohmann::json make_snapshot(const std::string& snapshot) override;
	void rollback(const std::string& snapshot, const nlohmann::json& opaque) override;
	void press(const std::vector<std::string>& buttons) override;
	void hold(const std::vector<std::string>& buttons) override;
	void release(const std::vector<std::string>& buttons) override;
	void mouse_move_abs(uint32_t x, uint32_t y) override;
	void mouse_move_rel(int x, int y) override;
	void mouse_hold(const std::vector<MouseButton>& buttons) override;
	void mouse_release(const std::vector<MouseButton>& buttons) override;
	bool is_nic_plugged(const std::string& pci_addr) const override;
	std::string attach_nic(const std::string& nic) override;
	void detach_nic(const std::string& pci_addr) override;
	bool is_link_plugged(const std::string& pci_addr) const override;
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
	void power_button() override;
	void suspend() override;
	void resume() override;
	stb::Image<stb::RGB> screenshot() override;
	int run(const fs::path& exe, std::vector<std::string> args,
		const std::function<void(const std::string&)>& callback) override;

	bool is_flash_plugged(std::shared_ptr<FlashDrive> fd) override;
	bool has_snapshot(const std::string& snapshot) override;
	void delete_snapshot(const std::string& snapshot) override;
	bool is_defined() const override;
	VmState state() const override;
	bool is_additions_installed() override;

	void copy_to_guest(const fs::path& src, const fs::path& dst) override;
	void copy_from_guest(const fs::path& src, const fs::path& dst) override;
	void remove_from_guest(const fs::path& obj) override;
	std::string get_tmp_dir() override;

private:
	void import_disk(const std::string& name, const fs::path& source);
	void create_new_disk(const std::string& name, uint32_t size);
	void create_disks();
	void attach_flash_drive(const std::string& img_path);
	void detach_flash_drive();

	std::string preferable_video_model();
	std::string mouse_button_to_str(MouseButton btn);

	std::set<std::string> plugged_nics() const;

	vir::Connect qemu_connect;
	std::unordered_map<std::string, uint32_t> scancodes;
	std::vector<std::string> disk_targets; //10 + a cdrom
	std::vector<uint8_t> screenshot_buffer;

};
