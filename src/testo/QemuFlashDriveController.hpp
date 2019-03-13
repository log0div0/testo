
#pragma once

#include "FlashDriveController.hpp"
#include "qemu/Host.hpp"

struct QemuFlashDriveController: FlashDriveController {
	QemuFlashDriveController() = delete;
	QemuFlashDriveController(const QemuFlashDriveController& other) = delete;
	QemuFlashDriveController(const nlohmann::json& config);
	int create() override;
	bool is_mounted() const override;
	int mount() const override;
	int umount() const override;
	int load_folder() const override;

	fs::path img_path() const override {
		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-flash-drives-pool");
		return pool.path() / (name() + ".img");
	}

	std::string name() const override {
		return config.at("name").get<std::string>();
	}

	bool has_folder() const override {
		return config.count("folder");
	}

private:
	void remove_if_exists();

	nlohmann::json config;
	vir::Connect qemu_connect;
};
