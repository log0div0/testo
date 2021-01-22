
#include "HyperVVM.hpp"
#include <iostream>

HyperVVM::HyperVVM(const nlohmann::json& config_): VM(config_) {
	if (config.count("nic")) {
		auto nics = config.at("nic");

		for (auto& nic: nics) {
			if (nic.count("adapter_type")) {
				std::string driver = nic.at("adapter_type").get<std::string>();
				throw std::runtime_error("Constructing VM \"" + id() + "\" error: nic \"" +
					nic.at("name").get<std::string>() + "\" has unsupported adapter type: \"" + driver + "\"");
			}
		}
	}

	scancodes.insert({
		{"ESC", {1}},
		{"ONE", {2}},
		{"TWO", {3}},
		{"THREE", {4}},
		{"FOUR", {5}},
		{"FIVE", {6}},
		{"SIX", {7}},
		{"SEVEN", {8}},
		{"EIGHT", {9}},
		{"NINE", {10}},
		{"ZERO", {11}},
		{"MINUS", {12}},
		{"EQUAL", {13}},
		{"BACKSPACE", {14}},
		{"TAB", {15}},
		{"Q", {16}},
		{"W", {17}},
		{"E", {18}},
		{"R", {19}},
		{"T", {20}},
		{"Y", {21}},
		{"U", {22}},
		{"I", {23}},
		{"O", {24}},
		{"P", {25}},
		{"LEFTBRACE", {26}},
		{"RIGHTBRACE", {27}},
		{"ENTER", {28}},
		{"LEFTCTRL", {29}},
		{"A", {30}},
		{"S", {31}},
		{"D", {32}},
		{"F", {33}},
		{"G", {34}},
		{"H", {35}},
		{"J", {36}},
		{"K", {37}},
		{"L", {38}},
		{"SEMICOLON", {39}},
		{"APOSTROPHE", {40}},
		{"GRAVE", {41}},
		{"LEFTSHIFT", {42}},
		{"BACKSLASH", {43}},
		{"Z", {44}},
		{"X", {45}},
		{"C", {46}},
		{"V", {47}},
		{"B", {48}},
		{"N", {49}},
		{"M", {50}},
		{"COMMA", {51}},
		{"DOT", {52}},
		{"SLASH", {53}},
		{"RIGHTSHIFT", {54}},
		{"LEFTALT", {56}},
		{"SPACE", {57}},
		{"CAPSLOCK", {58}},
		{"NUMLOCK", {69}}, //TODO: recheck
		{"SCROLLLOCK", {70}},

		{"F1", {59}},
		{"F2", {60}},
		{"F3", {61}},
		{"F4", {62}},
		{"F5", {63}},
		{"F6", {64}},
		{"F7", {65}},
		{"F8", {66}},
		{"F9", {67}},
		{"F10", {68}},
		{"F11", {87}},
		{"F12", {88}},

		{"RIGHTCTRL", {97}},
		{"RIGHTALT", {100}},

		{"HOME", {224,71}},
		{"UP", {224, 72}},
		{"PAGEUP", {224,73}},
		{"LEFT", {224,75}},
		{"RIGHT", {224,77}},
		{"END", {224,79}},
		{"DOWN", {224,80}},
		{"PAGEDOWN", {224,81}},
		{"INSERT", {224,82}},
		{"DELETE", {224,83}},

		{"SCROLLUP", {177}},
		{"SCROLLDOWN", {178}},
	});
}

HyperVVM::~HyperVVM() {

}

void HyperVVM::install() {
	try {
		for (auto& machine: connect.machines()) {
			if (machine.name() == id()) {
				if (machine.state() != hyperv::Machine::State::Disabled) {
					machine.disable();
				}
				machine.destroy();
			}
		}

		fs::path hhd_dir = connect.defaultVirtualHardDiskPath();
		fs::path hhd_path = hhd_dir / (id() + ".vhd");
		if (fs::exists(hhd_path)) {
			fs::remove(hhd_path);
		}

		auto machine = connect.defineMachine(id());

		machine.processor().setVirtualQuantity(config.at("cpus"));
		machine.memory().setVirtualQuantity(config.at("ram"));

		auto controllers = machine.ideControllers();
		controllers.at(0).addDVDDrive(0).mountISO(config.at("iso"));

		auto& disks = config.at("disk");
		for (size_t i = 0; i < disks.size(); ++i) {
			auto& disk = disks.at(i);
			size_t disk_size = disk.at("size").get<uint32_t>();
			disk_size = disk_size * 1024 * 1024;
			connect.createHDD(hhd_path, disk_size);
			controllers.at(1).addDiskDrive(i).mountHDD(hhd_path);
		}
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::undefine() {
	try {
		auto machine = connect.machine(id());
		if (machine.state() != hyperv::Machine::State::Disabled) {
			machine.disable();
		}
		machine.destroy();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::remove_disks() {
	try {
		std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

nlohmann::json HyperVVM::make_snapshot(const std::string& snapshot_name) {
	try {
		connect.machine(id()).createSnapshot().setName(snapshot_name);
		return nlohmann::json::object();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

bool HyperVVM::has_snapshot(const std::string& snapshot_name) {
	try {
		auto machine = connect.machine(id());
		for (auto& snapshot: machine.snapshots()) {
			if (snapshot.name() == snapshot_name) {
				return true;
			}
		}
		return false;
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::delete_snapshot(const std::string& snapshot_name) {
	try {
		connect.machine(id()).snapshot(snapshot_name).destroy();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::rollback(const std::string& snapshot_name, const nlohmann::json& opaque) {
	try {
		{
			auto machine = connect.machine(id());
			if (machine.state() != hyperv::Machine::State::Disabled) {
				machine.disable();
			}
			machine.snapshot(snapshot_name).apply();
		}
		{
			auto machine = connect.machine(id());
			if (machine.state() == hyperv::Machine::State::Offline) {
				machine.enable();
			}
		}
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::hold(const std::vector<std::string>& buttons) {
	try {
		std::vector<uint8_t> codes;
		for (auto button: buttons) {
			std::transform(button.begin(), button.end(), button.begin(), toupper);
			for (auto code: scancodes.at(button)) {
				codes.push_back(code);
			}
		}
		auto keyboard = connect.machine(id()).keyboard();
		keyboard.typeScancodes(codes);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::release(const std::vector<std::string>& buttons) {
	try {
		std::vector<uint8_t> codes;
		for (auto button: buttons) {
			std::transform(button.begin(), button.end(), button.begin(), toupper);
			for (auto code: scancodes.at(button)) {
				codes.push_back(code | 0x80);
			}
		}
		auto keyboard = connect.machine(id()).keyboard();
		keyboard.typeScancodes(codes);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::mouse_move_abs(uint32_t x, uint32_t y) {
	try {
		auto mouse = connect.machine(id()).synthetic_mouse();
		mouse.set_absolute_position(x, y);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::mouse_move_rel(int x, int y) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void HyperVVM::mouse_hold(const std::vector<MouseButton>& buttons) {
	try {
		auto mouse = connect.machine(id()).synthetic_mouse();
		for (auto button: buttons) {
			mouse.set_button_state((uint32_t)button, true);
		}
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::mouse_release(const std::vector<MouseButton>& buttons) {
	try {
		auto mouse = connect.machine(id()).synthetic_mouse();
		for (auto button: buttons) {
			mouse.set_button_state((uint32_t)button, false);
		}
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

bool HyperVVM::is_nic_plugged(const std::string& pci_addr) const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

std::string HyperVVM::attach_nic(const std::string& nic_name) {
	try {
		for (auto& nic_json: config.at("nic")) {
			if (nic_json.at("name") == nic_name) {
				auto machine = connect.machine(id());
				auto nic = machine.addNIC(nic_name);
				if (nic_json.count("mac")) {
					nic.setMAC(nic_json.at("mac"));
				}
				std::string net_name = prefix() + nic_json.at("attached_to").get<std::string>();
				auto bridge = connect.bridge(net_name);
				nic.connect(bridge);
				return nic_name;
			}
		}
		throw std::runtime_error("NIC " + nic_name + " not found");
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::detach_nic(const std::string& pci_addr) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVM::is_link_plugged(const std::string& nic) const {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVM::set_link(const std::string& nic, bool is_connected) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVM::plug_flash_drive(std::shared_ptr<FlashDrive> fd) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVM::unplug_flash_drive(std::shared_ptr<FlashDrive> fd) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVM::is_hostdev_plugged() {
	return false;
}
void HyperVVM::plug_hostdev_usb(const std::string& addr) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
void HyperVVM::unplug_hostdev_usb(const std::string& addr) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
bool HyperVVM::is_dvd_plugged() const {
	try {
		auto machine = connect.machine(id());
		auto controller = machine.ideControllers().at(0);
		auto drive = controller.drives().at(0);
		return drive.disks().size();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}
void HyperVVM::plug_dvd(fs::path path) {
	try {
		auto machine = connect.machine(id());
		auto controller = machine.ideControllers().at(0);
		auto drive = controller.drives().at(0);
		drive.mountISO(path);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}
void HyperVVM::unplug_dvd() {
	try {
		auto machine = connect.machine(id());
		auto controller = machine.ideControllers().at(0);
		auto drive = controller.drives().at(0);
		drive.disks().at(0).umount();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::start() {
	try {
		connect.machine(id()).enable();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::stop() {
	try {
		connect.machine(id()).disable();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::suspend() {
	try {
		connect.machine(id()).requestStateChange(hyperv::Machine::State::Quiesce);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}
void HyperVVM::resume() {
	try {
		connect.machine(id()).enable();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::power_button() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

uint8_t Table5[1 << 5] = {0, 8, 16, 25, 33, 41, 49, 58, 66, 74, 82, 90, 99, 107, 115, 123, 132,
 140, 148, 156, 165, 173, 181, 189, 197, 206, 214, 222, 230, 239, 247, 255};

uint8_t Table6[1 << 6] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 45, 49, 53, 57, 61, 65, 69,
 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 130, 134, 138,
 142, 146, 150, 154, 158, 162, 166, 170, 174, 178, 182, 186, 190, 194, 198,
 202, 206, 210, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255};

stb::Image<stb::RGB> HyperVVM::screenshot() {
	try {
		auto machine = connect.machine(id());

		if (machine.state() != hyperv::Machine::State::Enabled) {
			return {};
		}

		auto display = machine.display();

		if (display.state() != hyperv::Display::State::Enabled) {
			return {};
		}

		size_t height = display.height();
		size_t width = display.width();

		if (!width || !height) {
			return {};
		}

		std::vector<uint8_t> screenshot = display.screenshot();

		stb::Image<stb::RGB> result(width, height);

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

				result.data[dst_index + 0] = r8;
				result.data[dst_index + 1] = g8;
				result.data[dst_index + 2] = b8;
			}
		}

		return result;
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

bool HyperVVM::is_flash_plugged(std::shared_ptr<FlashDrive> fd) {
	try {
		std::cout << "TODO: " << __FUNCSIG__ << std::endl;
		return false;
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

bool HyperVVM::is_defined() const {
	try {
		for (auto& machine: connect.machines()) {
			if (machine.name() == id()) {
				return true;
			}
		}
		return false;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

VmState HyperVVM::state() const {
	try {
		auto state = connect.machine(id()).state();
		if (state == hyperv::Machine::State::Disabled) {
			return VmState::Stopped;
		} else if (state == hyperv::Machine::State::Enabled) {
			return VmState::Running;
		} else if (state == hyperv::Machine::State::Quiesce) {
			return VmState::Suspended;
		} else {
			return VmState::Other;
		}
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::shared_ptr<GuestAdditions> HyperVVM::guest_additions() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
