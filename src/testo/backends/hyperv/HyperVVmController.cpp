
#include "HyperVVmController.hpp"
#include <iostream>

HyperVVmController::HyperVVmController(const nlohmann::json& config_): VmController(config_) {
	std::cout << "HyperVVmController " << config.dump(4) << std::endl;

	scancodes.insert({
		{"ESC", 1},
		{"ONE", 2},
		{"TWO", 3},
		{"THREE", 4},
		{"FOUR", 5},
		{"FIVE", 6},
		{"SIX", 7},
		{"SEVEN", 8},
		{"EIGHT", 9},
		{"NINE", 10},
		{"ZERO", 11},
		{"MINUS", 12},
		{"EQUALSIGN", 13},
		{"BACKSPACE", 14},
		{"TAB", 15},
		{"Q", 16},
		{"W", 17},
		{"E", 18},
		{"R", 19},
		{"T", 20},
		{"Y", 21},
		{"U", 22},
		{"I", 23},
		{"O", 24},
		{"P", 25},
		{"LEFTBRACE", 26},
		{"RIGHTBRACE", 27},
		{"ENTER", 28},
		{"LEFTCTRL", 29},
		{"A", 30},
		{"S", 31},
		{"D", 32},
		{"F", 33},
		{"G", 34},
		{"H", 35},
		{"J", 36},
		{"K", 37},
		{"L", 38},
		{"SEMICOLON", 39},
		{"APOSTROPHE", 40},
		{"GRAVE", 41},
		{"LEFTSHIFT", 42},
		{"BACKSLASH", 43},
		{"Z", 44},
		{"X", 45},
		{"C", 46},
		{"V", 47},
		{"B", 48},
		{"N", 49},
		{"M", 50},
		{"COMMA", 51},
		{"DOT", 52},
		{"SLASH", 53},
		{"RIGHTSHIFT", 54},
		{"LEFTALT", 56},
		{"SPACE", 57},
		{"CAPSLOCK", 58},
		{"F1", 59},
		{"F2", 60},
		{"F3", 61},
		{"F4", 62},
		{"F5", 63},
		{"F6", 64},
		{"F7", 65},
		{"F8", 66},
		{"F9", 67},
		{"F10", 68},
		{"F11", 87},
		{"F12", 88},
		{"NUMLOCK", 69},
		{"SCROLLLOCK", 70},
		{"RIGHTCTRL", 97},
		{"RIGHTALT", 100},
		{"HOME", 102},
		{"UP", 103},
		{"PAGEUP", 104},
		{"LEFT", 105},
		{"RIGHT", 106},
		{"END", 107},
		{"DOWN", 108},
		{"PAGEDOWN", 109},
		{"INSERT", 110},
		{"DELETE", 111},
		{"SCROLLUP", 177},
		{"SCROLLDOWN", 178},
	});
}

HyperVVmController::~HyperVVmController() {

}

void HyperVVmController::install() {
	try {
		for (auto& machine: connect.machines()) {
			if (machine.name() == name()) {
				if (machine.is_running()) {
					machine.stop();
				}
				machine.destroy();
			}
		}

		auto machine = connect.defineMachine(name());
		auto controllers = machine.ideControllers();
		controllers.at(0).addDVDDrive(0).mountISO(config.at("iso"));
		// controllers.at(1).addDiskDrive(0);

		if (config.count("nic")) {
			for (auto& nic_cfg: config.at("nic")) {
				auto bridges = connect.bridges();
				auto it = std::find_if(bridges.begin(), bridges.end(), [&](auto bridge) {
					return bridge.name() == nic_cfg.at("network");
				});
				if (it == bridges.end()) {
					connect.defineBridge(nic_cfg.at("network"));
				}
				auto nic = machine.addNIC(nic_cfg.at("name"));
				if (nic_cfg.count("mac")) {
					nic.setMAC(nic_cfg.at("mac"));
				}
				auto bridge = connect.bridge(nic_cfg.at("network"));
				nic.connect(bridge);
			}
		}


		std::cout << "TODO: " << __FUNCSIG__ << std::endl;
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVmController::make_snapshot(const std::string& snapshot, const std::string& cksum) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::set_metadata(const std::string& key, const std::string& value) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

std::string HyperVVmController::get_metadata(const std::string& key) {
	try {
		auto machine = connect.machine(name());
		auto json = nlohmann::json::parse(machine.notes().at(0));
		return json.at(key);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string HyperVVmController::get_snapshot_cksum(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::rollback(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void HyperVVmController::press(const std::vector<std::string>& buttons) {
	try {
		std::vector<uint8_t> codes;
		for (auto button: buttons) {
			std::transform(button.begin(), button.end(), button.begin(), toupper);
			codes.push_back(scancodes.at(button));
		}
		connect.machine(name()).keyboard().typeScancodes(codes);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

bool HyperVVmController::is_nic_plugged(const std::string& nic) const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::set_nic(const std::string& nic, bool is_enabled) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVmController::is_link_plugged(const std::string& nic) const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::set_link(const std::string& nic, bool is_connected) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::plug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::unplug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVmController::is_dvd_plugged() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::plug_dvd(fs::path path) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::unplug_dvd() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void HyperVVmController::start() {
	try {
		connect.machine(name()).start();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVmController::stop() {
	try {
		connect.machine(name()).stop();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVmController::suspend() {
	try {
		connect.machine(name()).pause();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}
void HyperVVmController::resume() {
	try {
		connect.machine(name()).start();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVmController::power_button() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

uint8_t Table5[1 << 5] = {0, 8, 16, 25, 33, 41, 49, 58, 66, 74, 82, 90, 99, 107, 115, 123, 132,
 140, 148, 156, 165, 173, 181, 189, 197, 206, 214, 222, 230, 239, 247, 255};

uint8_t Table6[1 << 6] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 45, 49, 53, 57, 61, 65, 69,
 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 130, 134, 138,
 142, 146, 150, 154, 158, 162, 166, 170, 174, 178, 182, 186, 190, 194, 198,
 202, 206, 210, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255};

stb::Image HyperVVmController::screenshot() {
	try {
		auto machine = connect.machine(name());

		if (!machine.is_running()) {
			return {};
		}
		auto display = machine.display();

		size_t height = display.height();
		size_t width = display.width();
		std::vector<uint8_t> screenshot = display.screenshot();

		stb::Image result(width, height, 3);

		for (size_t h = 0; h < height; ++h) {
			for (size_t w = 0; w < width; ++w) {
				size_t dst_index = h*width*3 + w*3;
				size_t src_index = h*width*2 + w*2;
				uint16_t word = *(uint16_t*)(screenshot.data() + src_index);
				uint8_t r5 = word >> 11;
				uint8_t g6 = (word >> 5) & 0b00111111;
				uint8_t b5 = word & 0b00011111;
				uint8_t r8 = Table5[r5];
				uint8_t g8 = Table6[g6];
				uint8_t b8 = Table5[b5];

				result._data[dst_index + 0] = r8;
				result._data[dst_index + 1] = g8;
				result._data[dst_index + 2] = b8;
			}
		}

		return result;
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

int HyperVVmController::run(const fs::path& exe, std::vector<std::string> args, uint32_t timeout_seconds) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVmController::is_flash_plugged(std::shared_ptr<FlashDriveController> fd) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVmController::has_snapshot(const std::string& snapshot) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
std::vector<std::string> HyperVVmController::keys() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVmController::has_key(const std::string& key) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

bool HyperVVmController::is_defined() const {
	try {
		for (auto& machine: connect.machines()) {
			if (machine.name() == name()) {
				return true;
			}
		}
		return false;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

bool HyperVVmController::is_running() {
	try {
		return connect.machine(name()).is_running();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

bool HyperVVmController::is_additions_installed() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::copy_to_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_seconds) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::copy_from_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_seconds) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVmController::remove_from_guest(const fs::path& obj) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
std::set<std::string> HyperVVmController::nics() const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
