
#include <coro/Timer.h>
#include "HyperVGuestAdditions.hpp"
#include "HyperVVM.hpp"
#include <iostream>

using namespace std::chrono_literals;

HyperVVM::HyperVVM(const nlohmann::json& config_): VM(config_) {

	scancodes.insert({
		{KeyboardButton::ESC, {1}},
		{KeyboardButton::ONE, {2}},
		{KeyboardButton::TWO, {3}},
		{KeyboardButton::THREE, {4}},
		{KeyboardButton::FOUR, {5}},
		{KeyboardButton::FIVE, {6}},
		{KeyboardButton::SIX, {7}},
		{KeyboardButton::SEVEN, {8}},
		{KeyboardButton::EIGHT, {9}},
		{KeyboardButton::NINE, {10}},
		{KeyboardButton::ZERO, {11}},
		{KeyboardButton::MINUS, {12}},
		{KeyboardButton::EQUALSIGN, {13}},
		{KeyboardButton::BACKSPACE, {14}},
		{KeyboardButton::TAB, {15}},
		{KeyboardButton::Q, {16}},
		{KeyboardButton::W, {17}},
		{KeyboardButton::E, {18}},
		{KeyboardButton::R, {19}},
		{KeyboardButton::T, {20}},
		{KeyboardButton::Y, {21}},
		{KeyboardButton::U, {22}},
		{KeyboardButton::I, {23}},
		{KeyboardButton::O, {24}},
		{KeyboardButton::P, {25}},
		{KeyboardButton::LEFTBRACE, {26}},
		{KeyboardButton::RIGHTBRACE, {27}},
		{KeyboardButton::ENTER, {28}},
		{KeyboardButton::LEFTCTRL, {29}},
		{KeyboardButton::A, {30}},
		{KeyboardButton::S, {31}},
		{KeyboardButton::D, {32}},
		{KeyboardButton::F, {33}},
		{KeyboardButton::G, {34}},
		{KeyboardButton::H, {35}},
		{KeyboardButton::J, {36}},
		{KeyboardButton::K, {37}},
		{KeyboardButton::L, {38}},
		{KeyboardButton::SEMICOLON, {39}},
		{KeyboardButton::APOSTROPHE, {40}},
		{KeyboardButton::GRAVE, {41}},
		{KeyboardButton::LEFTSHIFT, {42}},
		{KeyboardButton::BACKSLASH, {43}},
		{KeyboardButton::Z, {44}},
		{KeyboardButton::X, {45}},
		{KeyboardButton::C, {46}},
		{KeyboardButton::V, {47}},
		{KeyboardButton::B, {48}},
		{KeyboardButton::N, {49}},
		{KeyboardButton::M, {50}},
		{KeyboardButton::COMMA, {51}},
		{KeyboardButton::DOT, {52}},
		{KeyboardButton::SLASH, {53}},
		{KeyboardButton::RIGHTSHIFT, {54}},
		{KeyboardButton::LEFTALT, {56}},
		{KeyboardButton::SPACE, {57}},
		{KeyboardButton::CAPSLOCK, {58}},
		{KeyboardButton::NUMLOCK, {69}}, //TODO: recheck
		{KeyboardButton::SCROLLLOCK, {70}},

		{KeyboardButton::F1, {59}},
		{KeyboardButton::F2, {60}},
		{KeyboardButton::F3, {61}},
		{KeyboardButton::F4, {62}},
		{KeyboardButton::F5, {63}},
		{KeyboardButton::F6, {64}},
		{KeyboardButton::F7, {65}},
		{KeyboardButton::F8, {66}},
		{KeyboardButton::F9, {67}},
		{KeyboardButton::F10, {68}},
		{KeyboardButton::F11, {87}},
		{KeyboardButton::F12, {88}},

		{KeyboardButton::RIGHTCTRL, {97}},
		{KeyboardButton::RIGHTALT, {100}},

		{KeyboardButton::HOME, {224,71}},
		{KeyboardButton::UP, {224, 72}},
		{KeyboardButton::PAGEUP, {224,73}},
		{KeyboardButton::LEFT, {224,75}},
		{KeyboardButton::RIGHT, {224,77}},
		{KeyboardButton::END, {224,79}},
		{KeyboardButton::DOWN, {224,80}},
		{KeyboardButton::PAGEDOWN, {224,81}},
		{KeyboardButton::INSERT, {224,82}},
		{KeyboardButton::DELETE, {224,83}},

		{KeyboardButton::SCROLLUP, {177}},
		{KeyboardButton::SCROLLDOWN, {178}},
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

		auto machine = connect.defineMachine(id());

		machine.processor().setVirtualQuantity(config.at("cpus"));
		machine.memory().setVirtualQuantity(config.at("ram"));

		auto controller = machine.addSCSIController();
		auto dvd = controller.addDVDDrive(0);
		if (config.count("iso")) {
			dvd.mountISO(config.at("iso"));
		}

		auto& disks = config.at("disk");
		for (size_t i = 0; i < disks.size(); ++i) {
			auto& disk = disks.at(i);
			fs::path disk_dir = fs::path(connect.defaultVirtualHardDiskPath()) / id();
			std::string disk_name = disk.at("name");
			fs::path disk_path;
			if (disk.count("size")) {
				disk_path = disk_dir / (disk_name + ".vhdx");
				size_t disk_size = disk.at("size");
				disk_size = disk_size * 1024 * 1024;
				connect.createDynamicHardDisk(disk_path, disk_size, hyperv::HardDiskFormat::VHDX);
			} else if (disk.count("source")) {
				fs::path source = disk.at("source").get<std::string>();
				hyperv::HardDiskFormat format;
				if (source.extension() == ".vhd") {
					format = hyperv::HardDiskFormat::VHD;
					disk_path = disk_dir / (disk_name + ".vhd");
				} else if (source.extension() == ".vhdx") {
					format = hyperv::HardDiskFormat::VHDX;
					disk_path = disk_dir / (disk_name + ".vhdx");
				} else {
					throw std::runtime_error("Unsupported disk format: " + source.extension().string());
				}
				connect.createDifferencingHardDisk(disk_path, disk.at("source"), format);
			} else {
				throw std::runtime_error("Shoud not be there");
			}
			controller.addDiskDrive(i + 1).mountHDD(disk_path);
		}

		if (config.count("nic")) {
			auto nics = config.at("nic");
			for (auto& nic: nics) {
				plug_nic(nic.at("name").get<std::string>());
			}
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
		fs::path disk_dir = fs::path(connect.defaultVirtualHardDiskPath()) / id();
		for (size_t i = 0; i < 30; ++i) {
			try {
				if (fs::exists(disk_dir)) {
					fs::remove_all(disk_dir);
				}
				return;
			} catch (const std::system_error& error) {
				if (error.code() == std::error_code(ERROR_SHARING_VIOLATION, std::system_category())) {
					coro::Timer timer;
					timer.waitFor(1s);
					continue;
				} else {
					throw;
				}
			}
		}
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
		fs::path disk_dir = fs::path(connect.defaultVirtualHardDiskPath()) / id();
		if (!fs::exists(disk_dir)) {
			return false;
		}
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

void HyperVVM::hold(KeyboardButton button) {
	try {
		auto keyboard = connect.machine(id()).keyboard();
		keyboard.typeScancodes(scancodes.at(button));
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::release(KeyboardButton button) {
	try {
		std::vector<uint8_t> codes;
		for (auto code: scancodes.at(button)) {
			codes.push_back(code | 0x80);
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

void HyperVVM::mouse_hold(MouseButton button) {
	try {
		auto mouse = connect.machine(id()).synthetic_mouse();
		mouse.set_button_state((uint32_t)button, true);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::mouse_release(MouseButton button) {
	try {
		auto mouse = connect.machine(id()).synthetic_mouse();
		mouse.set_button_state((uint32_t)button, false);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

bool HyperVVM::is_nic_plugged(const std::string& nic_name) const {
	try {
		auto machine = connect.machine(id());
		for (auto& nic: machine.nics()) {
			if (nic.name() == nic_name) {
				return true;
			}
		}
		return false;
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::plug_nic(const std::string& nic_name) {
	try {
		for (auto& nic_json: config.at("nic")) {
			if (nic_json.count("attached_to_dev")) {
				throw std::runtime_error("attached_to_dev mode is not implemented yet");
			}

			if (nic_json.at("name") == nic_name) {
				auto machine = connect.machine(id());
				auto nic = machine.addNIC(nic_name);
				if (nic_json.count("mac")) {
					nic.setMAC(nic_json.at("mac"));
				}
				std::string net_name;
				if (nic_json.at("network_mode") == "nat") {
					net_name = "Default Switch";
				} else {
					net_name = prefix() + nic_json.at("attached_to").get<std::string>();
				}
				auto bridge = connect.bridge(net_name);
				nic.connect(bridge);
				return;
			}
		}
		throw std::runtime_error("NIC " + nic_name + " not found");
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::unplug_nic(const std::string& nic_name) {
	try {
		auto machine = connect.machine(id());
		for (auto& nic: machine.nics()) {
			if (nic.name() == nic_name) {
				nic.destroy();
				return;
			}
		}
		throw std::runtime_error("NIC " + nic_name + " not found");
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

bool HyperVVM::is_link_plugged(const std::string& nic_name) const {
	try {
		auto machine = connect.machine(id());
		for (auto& nic: machine.nics()) {
			if (nic.name() == nic_name) {
				return nic.is_connected();
			}
		}
		throw std::runtime_error("NIC " + nic_name + " not found");
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::set_link(const std::string& nic_name, bool is_connected) {
	try {
		auto machine = connect.machine(id());
		for (auto& nic: machine.nics()) {
			if (nic.name() == nic_name) {
				if (is_connected) {
					for (auto& nic_json: config.at("nic")) {
						if (nic_json.at("name") == nic_name) {
							std::string net_name;
							if (nic_json.at("network_mode") == "nat") {
								net_name = "Default Switch";
							} else {
								net_name = prefix() + nic_json.at("attached_to").get<std::string>();
							}
							auto bridge = connect.bridge(net_name);
							nic.connect(bridge);
							return;
						}
					}
				} else {
					nic.disconnect();
					return;
				}
			}
		}
		throw std::runtime_error("NIC " + nic_name + " not found");
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
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
		auto controller = machine.scsiControllers().at(0);
		auto drive = controller.drives().at(0);
		return drive.disks().size();
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::plug_dvd(fs::path path) {
	try {
		auto machine = connect.machine(id());
		auto controller = machine.scsiControllers().at(0);
		auto drive = controller.drives().at(0);
		drive.mountISO(path);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void HyperVVM::unplug_dvd() {
	try {
		auto machine = connect.machine(id());
		auto controller = machine.scsiControllers().at(0);
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
	try {
		connect.machine(id()).requestStateChange(hyperv::Machine::State::ShutDown);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
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
	try {
		auto machine = connect.machine(id());
		return std::make_shared<HyperVGuestAdditions>(machine);
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Connecting to guest additions channel"));
	}
}
