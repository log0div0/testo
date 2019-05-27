
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
	void load_folder() const override;

	fs::path img_path() const override {
		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-flash-drives-pool");
		return pool.path() / (name() + ".img");
	}

private:
	fs::path img_dir() const {
		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-flash-drives-pool");
		return pool.path();
	}
	void remove_if_exists();

	vir::Connect qemu_connect;
};
