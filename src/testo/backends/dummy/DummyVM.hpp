
#pragma once

#include "../VM.hpp"

struct DummyVM: VM {
	DummyVM() = delete;
	DummyVM(const nlohmann::json& config);
	~DummyVM() override;
	void install() override;
	void undefine() override;
	void make_snapshot(const std::string& snapshot) override;
	void rollback(const std::string& snapshot) override;
	void press(const std::vector<std::string>& buttons) override;
	void hold(const std::vector<std::string>& buttons) override;
	void mouse_move_abs(uint32_t x, uint32_t y) override;
	void mouse_move_abs(const std::string& axis, uint32_t value) override;
	void mouse_move_rel(int x, int y) override;
	void mouse_move_rel(const std::string& axis, int value) override;
	void mouse_press(const std::vector<MouseButton>& buttons) override;
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
	void suspend() override;
	void resume() override;
	void power_button() override;
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
	std::string get_tmp_dir() override;

private:
	fs::path metadata_file() const {
		fs::path result = "./dummy_hypervisor_files";
		result = result / id();
		return result;
	};
};
