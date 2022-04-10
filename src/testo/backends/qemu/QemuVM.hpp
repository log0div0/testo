
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
	void hold(KeyboardButton button) override;
	void release(KeyboardButton button) override;
	void mouse_move_abs(uint32_t x, uint32_t y) override;
	void mouse_move_rel(int x, int y) override;
	void mouse_hold(MouseButton buttons) override;
	void mouse_release(MouseButton buttons) override;
	bool is_nic_plugged(const std::string& nic) const override;
	void plug_nic(const std::string& nic) override;
	void unplug_nic(const std::string& nic) override;
	bool is_link_plugged(const std::string& nic) const override;
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

	bool is_flash_plugged(std::shared_ptr<FlashDrive> fd) override;
	bool has_snapshot(const std::string& snapshot) override;
	void delete_snapshot(const std::string& snapshot) override;
	bool is_defined() const override;
	VmState state() const override;

	std::shared_ptr<GuestAdditions> guest_additions() override;

	static std::string preferable_video_model(const vir::Connect& qemu_connect);

private:
	bool is_my_disk(const fs::path& disk_path) const;

	std::string compose_config() const;

	void import_disk(const std::string& name, const fs::path& source);
	void create_new_disk(const std::string& name, uint32_t size);
	void create_disks();
	void attach_flash_drive(const std::string& img_path);
	void detach_flash_drive();

	bool is_target_free(const std::string& target);

	std::string mouse_button_to_str(MouseButton btn);

	std::set<std::string> plugged_nics() const;

	vir::Connect qemu_connect;
	std::unordered_map<std::string, std::string> nic_pci_map;
	std::vector<uint8_t> screenshot_buffer;

	bool use_external_snapshots() const;

	fs::path disks_dir() const;
	fs::path memory_dir() const;
	fs::path nvram_dir() const;

	fs::path current_nvram_path() const;
	fs::path nvram_snapshot_path(const std::string& snapshot_name) const;
	fs::path source_nvram_path() const;

	fs::path memory_snapshot_path(const std::string& snapshot_name) const;

	std::string disk_file_name(const std::string& name) const;
	fs::path disk_path(const std::string& name) const;
	std::string disk_snapshot_file_name(const std::string& disk_name, const std::string& snapshot_name) const;
	fs::path disk_snapshot_path(const std::string& disk_name, const std::string& snapshot_name) const;
};
