
#pragma once

#include "../FlashDrive.hpp"
#include <qemu/Host.hpp>

struct QemuFlashDrive: FlashDrive {
	QemuFlashDrive() = delete;
	QemuFlashDrive(const QemuFlashDrive& other) = delete;
	QemuFlashDrive(const nlohmann::json& config);
	~QemuFlashDrive() {}
	bool is_defined() override;
	void create() override;
	void undefine() override;
	void upload(const fs::path& from, const fs::path& to) override;
	void download(const fs::path& from, const fs::path& to) override;
	bool has_snapshot(const std::string& snapshot) override;
	void make_snapshot(const std::string& snapshot) override;
	void delete_snapshot(const std::string& snapshot) override;
	void rollback(const std::string& snapshot) override;

	fs::path img_path() const override;

private:
	vir::Connect qemu_connect;
};
