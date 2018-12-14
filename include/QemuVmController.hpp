
#pragma once

#include <VmController.hpp>

struct QemuVmController: public VmController {
	QemuVmController(): VmController() {}
	QemuVmController(const nlohmann::json& config): VmController(config) {}
	int install();
	int make_snapshot(const std::string& snapshot);
	int set_config_cksum(const std::string& cksum);
	std::string get_config_cksum();
	int set_snapshot_cksum(const std::string& snapshot, const std::string& cksum);
	std::string get_snapshot_cksum(const std::string& snapshot);
	int rollback(const std::string& snapshot);
	int press(const std::vector<std::string>& buttons);
	int plug_flash_drive(std::shared_ptr<FlashDriveController> fd);
	int unplug_flash_drive(std::shared_ptr<FlashDriveController> fd);
	int start();
	int stop();
	int type(const std::string& text);
	int wait(const std::string& text, const std::string& time);

	bool has_snapshot(const std::string& snapshot);
	bool is_defined() const;
	bool is_running();
};
