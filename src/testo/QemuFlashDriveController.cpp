
#include "QemuFlashDriveController.hpp"
#include "Utils.hpp"

QemuFlashDriveController::QemuFlashDriveController(const nlohmann::json& config):
config(config)
{}

int QemuFlashDriveController::create() {
	return 0;
}

bool QemuFlashDriveController::is_mounted() const {
	return true;
}

int QemuFlashDriveController::mount() const {
	return 0;
}

int QemuFlashDriveController::umount() const {
	return 0;
}

int QemuFlashDriveController::load_folder() const {
	return 0;
}
