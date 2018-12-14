
#include <QemuVmController.hpp>
#include <functional>

static std::string exec_and_read(const std::string& cmd) {
	std::array<char, 128> buffer;
	std::string result;
	std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
	if (!pipe) throw std::runtime_error("popen() failed!");
	while (!feof(pipe.get())) {
		if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
			result += buffer.data();
	}
	return result;
}

static void replaceAll(std::string& str, const std::string& from, const std::string& to) {
	if(from.empty())
		return;
	size_t start_pos = 0;
	while((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
	}
}

int QemuVmController::install() {
	std::string script_file("install.py ");
	std::string cmd = script_file + config.dump();

	replaceAll(cmd, "\"", "\\\"");
	int result = std::system(cmd.c_str());
	if (result == 0) {
		set_config_cksum(config_cksum());
	}

	return result;
}

int QemuVmController::make_snapshot(const std::string& snapshot) {
	std::string script_file("snapshot.py ");
	std::string cmd = script_file + name() + " " + snapshot;
	return std::system(cmd.c_str());
}

int QemuVmController::set_config_cksum(const std::string& cksum) {
	std::string script_file("set_config_cksum.py ");
	std::string cmd = script_file + name() + " " + cksum;
	return std::system(cmd.c_str());
}

std::string QemuVmController::get_config_cksum() {
	std::string script_file("get_config_cksum.py ");
	std::string cmd = script_file + name();

	std::string result = exec_and_read(cmd);
	return result.substr(0, result.length() - 1);
}


int QemuVmController::set_snapshot_cksum(const std::string& snapshot, const std::string& cksum) {
	std::string script_file("set_snapshot_cksum.py ");
	std::string cmd = script_file + name() + " " + snapshot + " " + cksum;
	return std::system(cmd.c_str());
}

std::string QemuVmController::get_snapshot_cksum(const std::string& snapshot) {
	std::string script_file("get_snapshot_cksum.py ");
	std::string cmd = script_file + name() + " " + snapshot;

	std::string result = exec_and_read(cmd);
	return result.substr(0, result.length() - 1);
}

int QemuVmController::rollback(const std::string& snapshot) {
	std::string script_file("rollback_new.py ");
	std::string cmd = script_file + name() + " " + snapshot;
	return std::system(cmd.c_str());
}

int QemuVmController::press(const std::vector<std::string>& buttons) {
	std::string script_file("press.py ");
	
	std::string cmd = script_file + name();
	for (auto& button: buttons) {
		cmd += " " + button;
	}

	return std::system(cmd.c_str());
}

int QemuVmController::plug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	//flash drive must be unplugged and mounted on host
	//first we umount it...
	if (fd->umount()) {
		throw std::runtime_error("Error while umounting flash drive " + fd->name() + " from host");
	}

	//now we attach it mazafaka
	std::string script_file("attach_flashdrive.py ");
	std::string cmd = script_file + name() + " " + fd->name();

	int result = std::system(cmd.c_str());
	if (result == 0) {
		//fd->is_plugged = true;
		fd->current_vm = name();
	}

	return result;
}

int QemuVmController::unplug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
		std::string script_file("detach_flashdrive.py ");
	std::string cmd = script_file + name() + " " + fd->name();

	int result = std::system(cmd.c_str());
	if (result == 0) {
		//fd->is_plugged = false;
		fd->current_vm = "";

		if (fd->mount()) {
			throw std::runtime_error("Error while mounting flash drive " + fd->name() + " to host");
		}
	}
	
	return result;
}

int QemuVmController::start() {
	std::string script_file("start.py ");
	std::string cmd = script_file + name();
	return std::system(cmd.c_str());
}

int QemuVmController::stop() {
	std::string script_file("stop.py ");
	std::string cmd = script_file + name();
	return std::system(cmd.c_str());
}

int QemuVmController::type(const std::string& text) {
	std::string script_file("type.py ");
	
	std::string cmd = script_file + name() + " \"" + text + "\"";
	return std::system(cmd.c_str());
}

int QemuVmController::wait(const std::string& text, const std::string& time) {
	std::string script_file("wait.py ");
	
	std::string cmd = script_file + "--timeout " + time + " " + name() + " \"" + text + "\"";
	return std::system(cmd.c_str());
}

bool QemuVmController::has_snapshot(const std::string& snapshot) {
	std::string script_file("has_snapshot.py ");
	std::string cmd = script_file + name() + " " + snapshot;
	return std::system(cmd.c_str());
}

bool QemuVmController::is_defined() const {
	std::string script_file("is_defined.py ");
	std::string cmd = script_file + name();
	return std::system(cmd.c_str());
}

bool QemuVmController::is_running() {
	std::string script_file("is_running.py ");
	std::string cmd = script_file + name();
	return std::system(cmd.c_str());
}
