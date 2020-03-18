
#pragma once

#include "../FlashDrive.hpp"
#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>

struct VboxFlashDrive: FlashDrive {
	VboxFlashDrive() = delete;
	VboxFlashDrive(const VboxFlashDrive& other) = delete;
	VboxFlashDrive(const nlohmann::json& config);
	bool is_defined() override;
	void create() override;
	void undefine() override;
	bool is_mounted() const override;
	void mount() override;
	void umount() override;
	bool has_snapshot(const std::string& snapshot) override;
	void make_snapshot(const std::string& snapshot) override;
	void delete_snapshot(const std::string& snapshot) override;
	void rollback(const std::string& snapshot) override;

	fs::path img_path() const override;

private:
	void remove_if_exists();

	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
};
