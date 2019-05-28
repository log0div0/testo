
#pragma once

#include "../FlashDriveController.hpp"
#include <qemu/Host.hpp>

struct QemuFlashDriveController: FlashDriveController {
	QemuFlashDriveController() = delete;
	QemuFlashDriveController(const QemuFlashDriveController& other) = delete;
	QemuFlashDriveController(const nlohmann::json& config);
	void create() override;
	bool is_mounted() const override;
	void mount() const override;
	void umount() const override;

	fs::path img_path() const override;
	fs::path mount_dir() const override;

private:
	void remove_if_exists();

	vir::Connect qemu_connect;
};
