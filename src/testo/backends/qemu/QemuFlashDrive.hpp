
#pragma once

#include "../FlashDrive.hpp"
#include <qemu/Host.hpp>

struct QemuFlashDrive: FlashDrive {
	QemuFlashDrive() = delete;
	QemuFlashDrive(const QemuFlashDrive& other) = delete;
	QemuFlashDrive(const nlohmann::json& config);
	bool is_defined() override;
	void create() override;
	bool is_mounted() const override;
	void mount() const override;
	void umount() const override;
	bool has_snapshot(const std::string& snapshot) override;
	void make_snapshot(const std::string& snapshot) override;
	void delete_snapshot(const std::string& snapshot) override;
	void rollback(const std::string& snapshot) override;

	fs::path img_path() const override;

private:
	void remove_if_exists();

	vir::Connect qemu_connect;
};
