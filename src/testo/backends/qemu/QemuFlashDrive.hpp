
#pragma once

#include "../FlashDrive.hpp"
#include <qemu/Host.hpp>

struct QemuFlashDrive: FlashDrive {
	QemuFlashDrive() = delete;
	QemuFlashDrive(const QemuFlashDrive& other) = delete;
	QemuFlashDrive(const nlohmann::json& config);
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
