
#include "VboxVM.hpp"
#include "VboxFlashDrive.hpp"
#include <vbox/lock.hpp>
#include <functional>

#include "../../Utils.hpp"

#include <chrono>
#include <thread>
#include <regex>

using namespace std::chrono_literals;

VboxVM::VboxVM(const nlohmann::json& config_): VM(config_) {
	if (!config.count("name")) {
		throw std::runtime_error("Constructing VboxVM error: field NAME is not specified");
	}

	if (!config.count("ram")) {
		throw std::runtime_error("Constructing VboxVM error: field RAM is not specified");
	}

	if (!config.count("cpus")) {
		throw std::runtime_error("Constructing VboxVM error: field CPUS is not specified");
	}

	if (!config.count("iso")) {
		throw std::runtime_error("Constructing VboxVM error: field ISO is not specified");
	}

	if (!config.count("disk_size")) {
		throw std::runtime_error("Constructing VboxVM error: field DISK SIZE is not specified");
	}

	if (config.count("nic")) {
		auto nics = config.at("nic");
		for (auto& nic: nics) {
			if (!nic.count("slot")) {
				throw std::runtime_error("Constructing VboxVM error: field slot is not specified for the nic " +
					nic.at("name").get<std::string>());
			}

			if (!nic.count("attached_to")) {
				throw std::runtime_error("Constructing VboxVM error: field attached_to is not specified for the nic " +
					nic.at("name").get<std::string>());
			}

			if (nic.at("attached_to").get<std::string>() == "internal") {
				if (!nic.count("network")) {
					throw std::runtime_error("Constructing VboxVM error: nic " +
					nic.at("name").get<std::string>() + " has type internal, but field network is not specified");
				}
			}

			if (nic.at("attached_to").get<std::string>() == "nat") {
				if (nic.count("network")) {
					throw std::runtime_error("Constructing VboxVM error: nic " +
					nic.at("name").get<std::string>() + " has type NAT, you must not specify field network");
				}
			}
		}

		for (uint32_t i = 0; i < nics.size(); i++) {
			for (uint32_t j = i + 1; j < nics.size(); j++) {
				if (nics[i].at("name") == nics[j].at("name")) {
					throw std::runtime_error("Constructing VboxVM error: two identical NIC names: " +
						nics[i].at("name").get<std::string>());
				}

				if (nics[i].at("slot") == nics[j].at("slot")) {
					throw std::runtime_error("Constructing VboxVM error: two identical SLOTS: " +
						std::to_string(nics[i].at("slot").get<uint32_t>()));
				}
			}
		}
	}


	virtual_box = virtual_box_client.virtual_box();
	start_session = virtual_box_client.session();
	work_session = virtual_box_client.session();

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

void VboxVM::remove_if_exists() {
	try {
		std::vector<vbox::Machine> machines = virtual_box.machines();
		for (auto& machine: machines) {
			if (machine.name() == id()) {
				vbox::Session session = virtual_box_client.session();
				{
					vbox::Lock lock(machine, session, LockType_Shared);

					auto machine = session.machine();
					if (machine.state() != MachineState_PoweredOff) {
						session.console().power_down().wait_and_throw_if_failed();
					}
				}

				while (machine.session_state() != SessionState_Unlocked) {
					std::this_thread::sleep_for(std::chrono::milliseconds(40));
				}

				machine.delete_config(machine.unregister(CleanupMode_DetachAllReturnHardDisksOnly)).wait_and_throw_if_failed();
			}
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::create_vm() {
	try {
		{
			vbox::GuestOSType guest_os_type = virtual_box.get_guest_os_type(config.value("vbox_os_type", "Other_64"));
			std::string settings_file_path = virtual_box.compose_machine_filename(id(), "/", {}, {});
			vbox::Machine machine = virtual_box.create_machine(settings_file_path, id(), {"/"}, guest_os_type.id(), {});

			machine.memory_size(config.at("ram").get<std::uint32_t>()); //for now, but we need to change it
			machine.vram_size(guest_os_type.recommended_vram() > 16 ? guest_os_type.recommended_vram() : 16);
			machine.cpus(config.at("cpus").get<uint32_t>());

			machine.add_usb_controller("OHCI", USBControllerType_OHCI);

			vbox::StorageController ide = machine.add_storage_controller("IDE", StorageBus_IDE);
			vbox::StorageController sata = machine.add_storage_controller("SATA", StorageBus_SATA);
			vbox::StorageController usb = machine.add_storage_controller("USB", StorageBus_USB);
			ide.port_count(2);
			sata.port_count(1);
			usb.port_count(8);

			machine.save_settings();
			virtual_box.register_machine(machine);
		}

		uint32_t nic_count = 0;

		{
			//TODO: CLEANUP IF SOMETHING WENT WRONG
			auto lock_machine = virtual_box.find_machine(id());
			vbox::Lock lock(lock_machine, work_session, LockType_Write);

			auto machine = work_session.machine();

			machine.setExtraData("VBoxInternal/Devices/VMMDev/0/Config/GetHostTimeDisabled", "1");

			std::experimental::filesystem::path iso_path(config.at("iso").get<std::string>());
			auto abs_iso_path = std::experimental::filesystem::absolute(iso_path);
			vbox::Medium dvd = virtual_box.open_medium(abs_iso_path.generic_string(),
				DeviceType_DVD, AccessMode_ReadOnly, false);
			machine.attach_device("IDE", 1, 0, DeviceType_DVD, dvd);

			std::string hard_disk_path = std::regex_replace(machine.settings_file_path(), std::regex("\\.vbox$"), ".vdi");
			vbox::Medium hard_disk = virtual_box.create_medium("vdi", hard_disk_path, AccessMode_ReadWrite, DeviceType_HardDisk);
			size_t disk_size = config.at("disk_size").get<uint32_t>();
			disk_size = disk_size * 1024 * 1024;
			hard_disk.create_base_storage(disk_size, MediumVariant_Standard).wait_and_throw_if_failed();
			machine.attach_device("SATA", 0, 0, DeviceType_HardDisk, hard_disk);

			machine.getNetworkAdapter(0).setEnabled(false);
			machine.getNetworkAdapter(0).setAttachmentType(NetworkAttachmentType_Null);

			//todo: NICS unordered_map or smth
			if (config.count("nic")) {
				auto& nics = config.at("nic");
				for (auto& nic: nics) {
					auto network_adapter = machine.getNetworkAdapter(nic.at("slot").get<uint32_t>());
					if (nic.at("attached_to") == "nat") {
						network_adapter.setAttachmentType(NetworkAttachmentType_NAT);
					} else if (nic.at("attached_to") == "internal") {
						network_adapter.setAttachmentType(NetworkAttachmentType_Internal);
						network_adapter.setInternalNetwork(nic.at("network").get<std::string>());
					} else {
						throw std::runtime_error("Unknown network attached_to type");
					}

					if (nic.count("mac")) {
						std::string mac = nic.at("mac").get<std::string>();
						if (!is_mac_correct(mac)) {
							throw std::runtime_error(std::string("Incorrect mac string: ") + mac);
						}

						network_adapter.setMAC(normalized_mac(mac));
					}

					if (nic.count("adapter_type")) {
						std::string type = nic.at("adapter_type").get<std::string>();
						if (type == "Am79C970A") {
							network_adapter.setAdapterType(NetworkAdapterType_Am79C970A);
						} else if (type == "Am79C973") {
							network_adapter.setAdapterType(NetworkAdapterType_Am79C973);
						} else if ((type == "82540EM") || (type == "e1000")) {
							network_adapter.setAdapterType(NetworkAdapterType_I82540EM);
						} else if (type == "82543GC") {
							network_adapter.setAdapterType(NetworkAdapterType_I82543GC);
						} else if (type == "82545EM") {
							network_adapter.setAdapterType(NetworkAdapterType_I82545EM);
						} else if (type == "virtio-net") {
							network_adapter.setAdapterType(NetworkAdapterType_Virtio);
						} else {
							throw std::runtime_error(std::string("Unknown adapter type: ") + type);
						}
					}

					network_adapter.setEnabled(true);
					nic_count++;
				}

			}
			machine.save_settings();
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::install() {
	try {
		remove_if_exists();
		create_vm();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::undefine() {
	throw std::runtime_error("Implement me");
}

void VboxVM::make_snapshot(const std::string& snapshot) {
	try {
		auto lock_machine = virtual_box.find_machine(id());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		if (machine.hasSnapshot(snapshot)) {
			auto existing_snapshot = machine.findSnapshot(snapshot);
			delete_snapshot_with_children(existing_snapshot);
		}
		machine.takeSnapshot(snapshot, "", true).wait_and_throw_if_failed();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::rollback(const std::string& snapshot) {
	try {
		stop();

		{
			auto lock_machine = virtual_box.find_machine(id());
			vbox::Lock lock(lock_machine, work_session, LockType_Shared);
			auto machine = work_session.machine();
			auto snap = machine.findSnapshot(snapshot);
			machine.restoreSnapshot(snap).wait_and_throw_if_failed();
		}

		start();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::press(const std::vector<std::string>& buttons) {
	try {
		auto machine = virtual_box.find_machine(id());
		vbox::Lock lock(machine, work_session, LockType_Shared);
		auto keyboard = work_session.console().keyboard();
		std::vector<uint8_t> codes;
		for (auto button: buttons) {
			std::transform(button.begin(), button.end(), button.begin(), toupper);
			for (auto code: scancodes.at(button)) {
				codes.push_back(code);
			}
		}
		for (auto code: codes) {
			keyboard.putScancode(code);
		}
		for (auto code: codes) {
			keyboard.putScancode(code | 0x80);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::mouse_move(const std::string& x, const std::string& y) {
	throw std::runtime_error("Implement me");
}

void VboxVM::mouse_set_buttons(uint32_t button_mask) {
	throw std::runtime_error("Implement me");
}

bool VboxVM::is_nic_plugged(const std::string& nic) const {
	try {
		if (!config.count("nic")) {
			throw std::runtime_error("There's no nics in this vm");
		}

		auto& nics = config.at("nic");

		for (auto& nic_it: nics) {
			if (nic_it.at("name") == nic) {
				auto machine = virtual_box.find_machine(id());
				auto network_adapter = machine.getNetworkAdapter(nic_it.at("slot").get<uint32_t>());
				return network_adapter.enabled();
			}
		}

		throw std::runtime_error(std::string("There's no nic with name ") + nic);
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::set_nic(const std::string& nic, bool is_enabled) {
	try {
		if (!config.count("nic")) {
			throw std::runtime_error("There's no nics in this vm");
		}

		auto& nics = config.at("nic");

		for (auto& nic_it: nics) {
			if (nic_it.at("name") == nic) {
				auto lock_machine = virtual_box.find_machine(id());

				vbox::Lock lock(lock_machine, work_session, LockType_Write);
				auto machine = work_session.machine();
				auto network_adapter = machine.getNetworkAdapter(nic_it.at("slot").get<uint32_t>());
				network_adapter.setEnabled(is_enabled);
				machine.save_settings();
				return;
			}
		}

		throw std::runtime_error(std::string("There's no nic with name ") + nic);
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool VboxVM::is_link_plugged(const std::string& nic) const {
	try {
		if (!config.count("nic")) {
			throw std::runtime_error("There's no nics in this vm");
		}

		auto& nics = config.at("nic");

		for (auto& nic_it: nics) {
			if (nic_it.at("name") == nic) {
				auto machine = virtual_box.find_machine(id());
				auto network_adapter = machine.getNetworkAdapter(nic_it.at("slot").get<uint32_t>());
				return network_adapter.cableConnected();
			}
		}

		throw std::runtime_error(std::string("There's no nic with name ") + nic);
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::set_link(const std::string& nic, bool is_connected) {
	try {
		if (!config.count("nic")) {
			throw std::runtime_error("There's no nics in this vm");
		}

		auto& nics = config.at("nic");

		for (auto& nic_it: nics) {
			if (nic_it.at("name") == nic) {
				auto lock_machine = virtual_box.find_machine(id());

				vbox::Lock lock(lock_machine, work_session, LockType_Shared);
				auto machine = work_session.machine();
				auto network_adapter = machine.getNetworkAdapter(nic_it.at("slot").get<uint32_t>());
				network_adapter.setCableConnected(is_connected);
				return;
			}
		}

		throw std::runtime_error(std::string("There's no nic with name ") + nic);
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool VboxVM::is_flash_plugged(std::shared_ptr<FlashDrive> fd) {
	return (plugged_fds.find(fd) != plugged_fds.end());
}

void VboxVM::plug_flash_drive(std::shared_ptr<FlashDrive> fd) {
	try {
		if (plugged_fds.find(fd) != plugged_fds.end()) {
			throw std::runtime_error("This flash drive is already attached to this vm");
		}

		auto lock_machine = virtual_box.find_machine(id());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);

		auto machine = work_session.machine();

		auto usb = machine.storage_controller_by_name("USB");
		auto attachments = machine.medium_attachments_of_controller("USB");

		std::sort(attachments.begin(), attachments.end());

		int32_t empty_slot = -1;
		int32_t port_count = usb.port_count();
		for (int32_t i = 0; i < port_count; i++) {
			if ((int32_t)attachments.size() < i) {
				if (attachments[i].port() != i) { //empty slot
					empty_slot = i;
					break;
				}
			} else {
				empty_slot = i;
				break;
			}
		}

		vbox::Medium medium = virtual_box.open_medium(fd->img_path().generic_string(), DeviceType_HardDisk, AccessMode_ReadWrite, false);
		machine.attach_device("USB", empty_slot, 0, DeviceType_HardDisk, medium);
		machine.save_settings();
		plugged_fds.insert(fd);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::unplug_flash_drive(std::shared_ptr<FlashDrive> fd) {
	try {
		if (plugged_fds.find(fd) == plugged_fds.end()) {
			throw std::runtime_error("This flash drive is not plugged to this vm");
		}

		auto lock_machine = virtual_box.find_machine(id());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);

		auto machine = work_session.machine();

		auto attachments = machine.medium_attachments_of_controller("USB");

		for (auto& attachment: attachments) {
			if (attachment.medium().location() == fd->img_path()) {
				machine.detach_device("USB", attachment.port(), attachment.device());
			}
		}
		machine.save_settings();
		plugged_fds.erase(fd);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool VboxVM::is_dvd_plugged() const {
	try {
		auto machine = virtual_box.find_machine(id());
		auto mediums = machine.medium_attachments_of_controller("IDE");
		for (auto& medium: mediums) {
			if (medium.port() == 1) {
				return true;
			}
		}
		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::plug_dvd(fs::path path) {
	try {
		if (path.is_relative()) {
			path = fs::absolute(path);
		}

		vbox::Medium dvd = virtual_box.open_medium(path.generic_string(),
				DeviceType_DVD, AccessMode_ReadOnly, false);

		auto lock_machine = virtual_box.find_machine(id());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		machine.mount_medium("IDE", 1, 0, dvd, false);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::unplug_dvd() {
	try {
		auto lock_machine = virtual_box.find_machine(id());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		machine.unmount_medium("IDE", 1, 0, false);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::start() {
	try {
		auto machine = virtual_box.find_machine(id());
		if (machine.state() == MachineState_Running) {
			return;
		}
		wait_state({MachineState_PoweredOff, MachineState_Saved});
		auto deadline = std::chrono::system_clock::now() + 10s;
		do {
			if (machine.session_state() == SessionState_Unlocked)
			{
				machine.launch_vm_process(start_session, "headless").wait_and_throw_if_failed();
				start_session.unlock_machine();
				wait_state({MachineState_Running});
				return;
			}
			std::this_thread::sleep_for(100ms);
		} while (std::chrono::system_clock::now() < deadline);

		throw std::runtime_error("Failed to start machine, because it's locked");
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::stop() {
	try {
		auto machine = virtual_box.find_machine(id());
		if ((machine.state() == MachineState_PoweredOff) ||
			(machine.state() == MachineState_Saved) ||
			(machine.state() == MachineState_Aborted)) {
			return;
		}
		wait_state({MachineState_Running, MachineState_Paused});
		vbox::Lock lock(machine, work_session, LockType_Shared);
		work_session.console().power_down().wait_and_throw_if_failed();
		wait_state({MachineState_PoweredOff});
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::power_button() {
	try {
		auto machine = virtual_box.find_machine(id());
		vbox::Lock lock(machine, work_session, LockType_Shared);
		work_session.console().power_button();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::suspend() {
	try {
		auto machine = virtual_box.find_machine(id());
		if (machine.state() == MachineState_Paused) {
			return;
		}
		wait_state({MachineState_Running});
		vbox::Lock lock(machine, work_session, LockType_Shared);
		work_session.console().pause();
		wait_state({MachineState_Paused});
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::resume() {
	try {
		auto machine = virtual_box.find_machine(id());
		if (machine.state() == MachineState_Running) {
			return;
		}
		wait_state({MachineState_Paused});
		vbox::Lock lock(machine, work_session, LockType_Shared);
		work_session.console().resume();
		wait_state({MachineState_Running});
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

stb::Image VboxVM::screenshot() {
	try {
		auto lock_machine = virtual_box.find_machine(id());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto display = work_session.console().display();

		ULONG width = 0;
		ULONG height = 0;
		ULONG bits_per_pixel = 0;
		LONG x_origin = 0;
		LONG y_origin = 0;
		GuestMonitorStatus guest_monitor_status = GuestMonitorStatus_Disabled;

		display.get_screen_resolution(0, &width, &height, &bits_per_pixel, &x_origin, &y_origin, &guest_monitor_status);

		if (!width || !height) {
			return {};
		}

		stb::Image result(width, height, 3);

		vbox::SafeArray safe_array = display.take_screen_shot_to_array(0, width, height, BitmapFormat_BGRA);
		vbox::ArrayOut array_out = safe_array.copy_out(VT_UI1);

		for(size_t h = 0; h < height; ++h){
			for(size_t w = 0; w < width; ++w){
				for(size_t c = 0; c < 3; ++c){
					size_t src_index = h*width*4 + w*4 + c;
					size_t dst_index = h*width*3 + w*3 + c;
					result.data[dst_index] = array_out[src_index];
				}
			}
		}

		return result;
	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return {};
	}
}

int VboxVM::run(const fs::path& exe, std::vector<std::string> args, uint32_t timeout_milliseconds) {
	try {
		args.insert(args.begin(), "--");
		uint32_t timeout = timeout_milliseconds * 1000 * 1000;

		auto lock_machine = virtual_box.find_machine(id());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		//1) Open the session

		auto machine = work_session.machine();
		// auto login = machine.getExtraData("login");
		std::string login = "root";
		auto password = machine.getExtraData("password");

		if (!login.length()) {
			throw std::runtime_error("Attribute login is not specified");
		}

		if (!password.length()) {
			throw std::runtime_error("Attribute login is not specified");
		}

		auto gsession = work_session.console().guest().create_session(login, password);

		std::vector<ProcessCreateFlag> create_flags = {ProcessCreateFlag_WaitForStdOut, ProcessCreateFlag_WaitForStdErr};
		std::vector<ProcessWaitForFlag> wait_for_flags = {ProcessWaitForFlag_StdOut, ProcessWaitForFlag_StdErr, ProcessWaitForFlag_Terminate};
		auto gprocess = gsession.process_create(exe.generic_string(), args, {}, create_flags, timeout);

		bool completed = false;
		while (!completed) {
			auto result = gprocess.wait_for(wait_for_flags, 200); //wait for 500 ms
			switch (result) {
				case ProcessWaitResult_Terminate:
					completed = true;
					break;
				case ProcessWaitResult_StdOut:
				case ProcessWaitResult_StdErr:
				case ProcessWaitResult_WaitFlagNotSupported:
				{
					//print both streams
					auto stderr_read = gprocess.read(2, 0x00010000, 0);
					if (stderr_read.size() > 1) {
						std::string err(stderr_read.begin(), stderr_read.end());
						std::cout << err;
					}
					auto stdout_read = gprocess.read(1, 0x00010000, 0); //read 64 KBytes
					if (stdout_read.size() > 1) {
						std::string out(stdout_read.begin(), stdout_read.end());
						std::cout << out;
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(50));
					break;
				}
				case ProcessWaitResult_Timeout:
					if (!gprocess.is_alive()) {
						completed = true;
					}
					break;
				case ProcessWaitResult_Error:
					throw std::runtime_error("Got ProcessWaitResult_Error");
					break;
				default:
					break;
			}
		}

		auto status = gprocess.status();
		if ((status == ProcessStatus_TimedOutKilled) || (status == ProcessStatus_TimedOutAbnormally)) {
			throw std::runtime_error("Guest process timeout out");
		}
		if ((status != ProcessStatus_TerminatedNormally) && (status != ProcessStatus_TerminatedSignal)) {
			throw std::runtime_error("Guest process terminated abnormally");
		}
		//okay
		return gprocess.exit_code();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool VboxVM::has_snapshot(const std::string& snapshot) {
	try {
		auto lock_machine = virtual_box.find_machine(id());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		return machine.hasSnapshot(snapshot);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::delete_snapshot(const std::string& snapshot) {
	try {
		auto lock_machine = virtual_box.find_machine(id());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		if (machine.hasSnapshot(snapshot)) {
			auto existing_snapshot = machine.findSnapshot(snapshot);
			machine.deleteSnapshot(existing_snapshot);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool VboxVM::is_defined() const {
	std::vector<vbox::Machine> machines = virtual_box.machines();
	for (auto& machine: machines) {
		if (machine.name() == id()) {
			return true;
		}
	}
	return false;
}

VmState VboxVM::state() const {
	try {
		auto machine = virtual_box.find_machine(id());
		auto state = machine.state();
		if (state == MachineState_PoweredOff) {
			return VmState::Stopped;
		} else if (state == MachineState_Running) {
			return VmState::Running;
		} else if (state == MachineState_Paused) {
			return VmState::Suspended;
		} else {
			return VmState::Other;
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::delete_snapshot_with_children(vbox::Snapshot& snapshot) {
	auto children = snapshot.children();

	if (children.size()) {
		for (auto& snap: children) {
			delete_snapshot_with_children(snap);
		}
	}

	auto machine = work_session.machine();
	machine.deleteSnapshot(snapshot).wait_and_throw_if_failed();
}

bool VboxVM::is_additions_installed() {
	try {
		auto lock_machine = virtual_box.find_machine(id());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto facilities = work_session.console().guest().facilities();

		for (auto& facility: facilities) {
			if (facility.type() == AdditionsFacilityType_VBoxService) {
				return true;
			}
		}
		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::copy_dir_to_guest(const fs::path& src, const fs::path& dst, vbox::GuestSession& gsession) {
	gsession.directory_create(dst.generic_string());

	for (auto& file: fs::directory_iterator(src)) {
		if (fs::is_regular_file(file)) {
			gsession.file_copy_to_guest(file.path().generic_string(), (dst / file.path().filename()).generic_string()).wait_and_throw_if_failed();
		} else if (fs::is_directory(file)) {
			copy_dir_to_guest(file.path().generic_string(), (dst / file.path().filename()).generic_string(), gsession);
		} //else continue
	}
}

void VboxVM::copy_to_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds) {
	try {
		//1) if there's no src on host - fuck you
		if (!fs::exists(src)) {
			throw std::runtime_error("Source file/folder doens't exist on host: " + src.generic_string());
		}

		auto lock_machine = virtual_box.find_machine(id());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);

		auto machine = work_session.machine();
		// auto login = machine.getExtraData("login");
		std::string login = "root";
		auto password = machine.getExtraData("password");

		if (!login.length()) {
			throw std::runtime_error("Attribute login is not specified");
		}

		if (!password.length()) {
			throw std::runtime_error("Attribute login is not specified");
		}

		auto gsession = work_session.console().guest().create_session(login, password);

		if (fs::is_regular_file(src)) {
			gsession.file_copy_to_guest(src.generic_string(), dst.generic_string()).wait_and_throw_if_failed();
		} else if (fs::is_directory(src)) {
			copy_dir_to_guest(src, dst, gsession);
		} else {
			throw std::runtime_error("Unknown type of file: " + dst.generic_string());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::copy_from_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds) {
	throw std::runtime_error("Implement me");
}

void VboxVM::remove_from_guest(const fs::path& obj) {
	try {
		auto lock_machine = virtual_box.find_machine(id());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);

		auto machine = work_session.machine();
		// auto login = machine.getExtraData("login");
		std::string login = "root";
		auto password = machine.getExtraData("password");

		if (!login.length()) {
			throw std::runtime_error("Attribute login is not specified");
		}

		if (!password.length()) {
			throw std::runtime_error("Attribute login is not specified");
		}

		auto gsession = work_session.console().guest().create_session(login, password);

		//directory handling differs from file handling
		if (gsession.directory_exists(obj.generic_string())) {
			gsession.directory_remove_recursive(obj.generic_string());
		} else if (gsession.file_exists(obj.generic_string())) {
			gsession.file_remove(obj.generic_string());
		} else {
			throw std::runtime_error("Target object doesn't exist on vm: " + obj.generic_string());
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VboxVM::wait_state(std::initializer_list<MachineState> states) {
	auto machine = virtual_box.find_machine(id());
	auto deadline = std::chrono::system_clock::now() + 10s;
	do {
		auto it = std::find(states.begin(), states.end(), machine.state());
		if (it != states.end()) {
			return;
		}
		std::this_thread::sleep_for(100ms);
	} while (std::chrono::system_clock::now() < deadline);

	std::stringstream ss;
	ss << "Machine is not in one of this states: ";
	for (auto it = states.begin(); it != states.end(); ++it) {
		if (it != states.begin()) {
			ss << ", ";
		}
		ss << *it;
	}

	throw std::runtime_error(ss.str());
}
