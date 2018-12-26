
#include <VmController.hpp>
#include <vbox/lock.hpp>
#include <functional>

#include <API.hpp>
#include <Utils.hpp>
#include <leptonica/allheaders.h>

#include <chrono>
#include <thread>
#include <regex>

#include <experimental/filesystem>

VmController::VmController(const nlohmann::json& config): config(config) {
	if (!config.count("name")) {
		throw std::runtime_error("Constructing VmController error: field NAME is not specified");
	}

	if (!config.count("ram")) {
		throw std::runtime_error("Constructing VmController error: field RAM is not specified");
	}

	if (!config.count("cpus")) {
		throw std::runtime_error("Constructing VmController error: field CPUS is not specified");
	}

	if (!config.count("iso")) {
		throw std::runtime_error("Constructing VmController error: field ISO is not specified");
	}

	if (!config.count("disk_size")) {
		throw std::runtime_error("Constructing VmController error: field DISK SIZE is not specified");
	}

	if (config.count("nic")) {
		auto nics = config.at("nic");
		for (auto& nic: nics) {
			if (!nic.count("slot")) {
				throw std::runtime_error("Constructing VmController error: field slot is not specified for the nic " + 
					nic.at("name").get<std::string>());
			}
			
			if (!nic.count("attached_to")) {
				throw std::runtime_error("Constructing VmController error: field attached_to is not specified for the nic " + 
					nic.at("name").get<std::string>());
			}

			if (nic.at("attached_to").get<std::string>() == "internal") {
				if (!nic.count("network")) {
					throw std::runtime_error("Constructing VmController error: nic " + 
					nic.at("name").get<std::string>() + " has type internal, but field network is not specified");
				}
			}

			if (nic.at("attached_to").get<std::string>() == "nat") {
				if (nic.count("network")) {
					throw std::runtime_error("Constructing VmController error: nic " + 
					nic.at("name").get<std::string>() + " has type NAT, you must not specify field network");
				}
			}
		}

		for (uint32_t i = 0; i < nics.size(); i++) {
			for (uint32_t j = i + 1; j < nics.size(); j++) {
				if (nics[i].at("name") == nics[j].at("name")) {
					throw std::runtime_error("Constructing VmController error: two identical NIC names: " + 
						nics[i].at("name").get<std::string>()); 
				}

				if (nics[i].at("slot") == nics[j].at("slot")) {
					throw std::runtime_error("Constructing VmController error: two identical SLOTS: " + 
						nics[i].at("slot").get<uint32_t>()); 
				}
			}
		}
	}


	virtual_box = virtual_box_client.virtual_box();
	start_session = virtual_box_client.session();
	work_session = virtual_box_client.session();

	if (!config.count("os_type")) {
		throw std::runtime_error("Constructing VmController error: field OSType is not specified");
	}

	charmap.insert({
		{'0', {"ZERO"}},
		{'1', {"ONE"}},
		{'2', {"TWO"}},
		{'3', {"THREE"}},
		{'4', {"FOUR"}},
		{'5', {"FIVE"}},
		{'6', {"SIX"}},
		{'7', {"SEVEN"}},
		{'8', {"EIGHT"}},
		{'9', {"NINE"}},

		{')', {"LEFTSHIFT", "ZERO"}},
		{'!', {"LEFTSHIFT", "ONE"}},
		{'@', {"LEFTSHIFT", "TWO"}},
		{'#', {"LEFTSHIFT", "THREE"}},
		{'$', {"LEFTSHIFT", "FOUR"}},
		{'%', {"LEFTSHIFT", "FIVE"}},
		{'^', {"LEFTSHIFT", "SIX"}},
		{'&', {"LEFTSHIFT", "SEVEN"}},
		{'*', {"LEFTSHIFT", "EIGHT"}},
		{'(', {"LEFTSHIFT", "NINE"}},

		{'a', {"A"}},
		{'b', {"B"}},
		{'c', {"C"}},
		{'d', {"D"}},
		{'e', {"E"}},
		{'f', {"F"}},
		{'g', {"G"}},
		{'h', {"H"}},
		{'i', {"I"}},
		{'j', {"J"}},
		{'k', {"K"}},
		{'l', {"L"}},
		{'m', {"M"}},
		{'n', {"N"}},
		{'o', {"O"}},
		{'p', {"P"}},
		{'q', {"Q"}},
		{'r', {"R"}},
		{'s', {"S"}},
		{'t', {"T"}},
		{'u', {"U"}},
		{'v', {"V"}},
		{'w', {"W"}},
		{'x', {"X"}},
		{'y', {"Y"}},
		{'z', {"Z"}},

		{'A', {"LEFTSHIFT", "A"}},
		{'B', {"LEFTSHIFT", "B"}},
		{'C', {"LEFTSHIFT", "C"}},
		{'D', {"LEFTSHIFT", "D"}},
		{'E', {"LEFTSHIFT", "E"}},
		{'F', {"LEFTSHIFT", "F"}},
		{'G', {"LEFTSHIFT", "G"}},
		{'H', {"LEFTSHIFT", "H"}},
		{'I', {"LEFTSHIFT", "I"}},
		{'J', {"LEFTSHIFT", "J"}},
		{'K', {"LEFTSHIFT", "K"}},
		{'L', {"LEFTSHIFT", "L"}},
		{'M', {"LEFTSHIFT", "M"}},
		{'N', {"LEFTSHIFT", "N"}},
		{'O', {"LEFTSHIFT", "O"}},
		{'P', {"LEFTSHIFT", "P"}},
		{'Q', {"LEFTSHIFT", "Q"}},
		{'R', {"LEFTSHIFT", "R"}},
		{'S', {"LEFTSHIFT", "S"}},
		{'T', {"LEFTSHIFT", "T"}},
		{'U', {"LEFTSHIFT", "U"}},
		{'V', {"LEFTSHIFT", "V"}},
		{'W', {"LEFTSHIFT", "W"}},
		{'X', {"LEFTSHIFT", "X"}},
		{'Y', {"LEFTSHIFT", "Y"}},
		{'Z', {"LEFTSHIFT", "Z"}},

		{'-', {"MINUS"}},
		{'_', {"LEFTSHIFT", "MINUS"}},

		{'=', {"EQUAL"}},
		{'+', {"LEFTSHIFT", "EQUAL"}},

		{'\'', {"APOSTROPHE"}},
		{'\"', {"LEFTSHIFT", "APOSTROPHE"}},

		{'\\', {"BACKSLASH"}},
		{'|', {"LEFTSHIFT", "BACKSLASH"}},

		{',', {"COMMA"}},
		{'<', {"LEFTSHIFT", "COMMA"}},

		{'.', {"DOT"}},
		{'>', {"LEFTSHIFT", "DOT"}},

		{'/', {"SLASH"}},
		{'?', {"LEFTSHIFT", "SLASH"}},

		{';', {"SEMICOLON"}},
		{':', {"LEFTSHIFT", "SEMICOLON"}},

		{'[', {"LEFTBRACE"}},
		{'{', {"LEFTSHIFT", "LEFTBRACE"}},

		{']', {"RIGHTBRACE"}},
		{'}', {"LEFTSHIFT", "RIGHTBRACE"}},

		{'`', {"GRAVE"}},
		{'~', {"LEFTSHIFT", "GRAVE"}},

		{' ', {"SPACE"}},
	});
}

void VmController::remove_if_exists() {
	try {
		std::vector<vbox::Machine> machines = virtual_box.machines();
		for (auto& machine: machines) {
			if (machine.name() == name()) {
				vbox::Session session = virtual_box_client.session();
				{
					vbox::Lock lock(machine, session, LockType_Shared);

					auto machine = session.machine();
					if (machine.state() == MachineState_Running) {
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

void VmController::create_vm() {
	try {
		{
			vbox::GuestOSType guest_os_type = virtual_box.get_guest_os_type(config.at("os_type").get<std::string>());
			std::string settings_file_path = virtual_box.compose_machine_filename(name(), "/", {}, {});
			vbox::Machine machine = virtual_box.create_machine(settings_file_path, name(), {"/"}, guest_os_type.id(), {});

			machine.memory_size(config.at("ram").get<std::uint32_t>()); //for now, but we need to change it
			machine.vram_size(guest_os_type.recommended_vram());
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
		
		{
			//TODO: CLEANUP IF SOMETHING WENT WRONG
			auto lock_machine = virtual_box.find_machine(name());
			vbox::Lock lock(lock_machine, work_session, LockType_Write);

			auto machine = work_session.machine();
			
			std::experimental::filesystem::path iso_path(config.at("iso").get<std::string>());
			auto abs_iso_path = std::experimental::filesystem::absolute(iso_path);
			vbox::Medium dvd = virtual_box.open_medium(abs_iso_path,
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

					network_adapter.setEnabled(true);
				}

			}
			machine.save_settings();
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

int VmController::set_metadata(const nlohmann::json& metadata) {
	try {
		for (auto key_value = metadata.begin(); key_value != metadata.end(); ++key_value) {
			auto lock_machine = virtual_box.find_machine(name());
			vbox::Lock lock(lock_machine, work_session, LockType_Shared);
			auto machine = work_session.machine();
			machine.setExtraData(key_value.key(), key_value.value());
		}

		return 0;		
	}
	catch (const std::exception& error) {
		std::cout << "Setting metadata on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int VmController::set_metadata(const std::string& key, const std::string& value) {
	try {
		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		machine.setExtraData(key, value);
		return 0;
	}
	catch (const std::exception& error) {
		std::cout << "Setting metadata on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int VmController::install() {	
	try {
		remove_if_exists();
		create_vm();
		if (config.count("metadata")) {
			set_metadata(config.at("metadata"));
		}
		set_config_cksum(config_cksum());
		if (start()) {
			throw std::runtime_error("Start while performing install action");
		}
		return 0;
	}
	catch (const std::exception& error) {
		std::cout << "Performing install on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int VmController::make_snapshot(const std::string& snapshot) {
	try {
		{
			auto lock_machine = virtual_box.find_machine(name());
			vbox::Lock lock(lock_machine, work_session, LockType_Shared);
			auto machine = work_session.machine();
			if (machine.hasSnapshot(snapshot)) {
				auto existing_snapshot = machine.findSnapshot(snapshot);
				delete_snapshot_with_children(existing_snapshot);
			}
		}

		{
			auto lock_machine = virtual_box.find_machine(name());
			vbox::Lock lock(lock_machine, work_session, LockType_Shared);

			auto machine = work_session.machine();
			machine.takeSnapshot(snapshot).wait_and_throw_if_failed();
		}

		if (!is_running()) {
			auto lock_machine = virtual_box.find_machine(name());
			lock_machine.launch_vm_process(start_session, "headless").wait_and_throw_if_failed();
			start_session.unlock_machine();
		}
		
		return 0;
	}
	catch (const std::exception& error) {
		std::cout << "Taking snapshot on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

std::string VmController::config_cksum() const {
	std::hash<std::string> h;
	return std::to_string(h(config.dump()));
}

std::set<std::string> VmController::nics() const {
	std::set<std::string> result;

	for (auto& nic: config.at("nic")) {
		result.insert(nic.at("name").get<std::string>());
	}
	return result;
}

int VmController::set_config_cksum(const std::string& cksum) {
	try {
		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		machine.setExtraData("config_cksum", cksum);
		return 0;
	}
	catch (const std::exception& error) {
		std::cout << "Setting config cksum on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

std::string VmController::get_config_cksum() {
	try {
		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		return machine.getExtraData("config_cksum");
	}
	catch (const std::exception& error) {
		std::cout << "Getting config cksum on vm " << name() << ": " << error << std::endl;
		return "";
	}
}


int VmController::set_snapshot_cksum(const std::string& snapshot, const std::string& cksum) {
	try {
		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		machine.findSnapshot(snapshot).setDescription(cksum);
		return 0;
	}
	catch (const std::exception& error) {
		std::cout << "Setting snapshot cksum on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

std::string VmController::get_snapshot_cksum(const std::string& snapshot) {
	try {
		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		return machine.findSnapshot(snapshot).getDescription();
	}
	catch (const std::exception& error) {
		std::cout << "getting snapshot cksum on vm " << name() << ": " << error << std::endl;
		return "";
	}
}

int VmController::rollback(const std::string& snapshot) {
	try {
		auto lock_machine = virtual_box.find_machine(name());
		if (is_running()) {
			stop();
		}

		{
			vbox::Lock lock(lock_machine, work_session, LockType_Shared);
			auto machine = work_session.machine();
			auto snap = machine.findSnapshot(snapshot);
			machine.restoreSnapshot(snap).wait_and_throw_if_failed();
		}

		lock_machine.launch_vm_process(start_session, "headless").wait_and_throw_if_failed();
		start_session.unlock_machine();

		return 0;
	}
	catch (const std::exception& error) {
		std::cout << "Performing rollback on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int VmController::press(const std::vector<std::string>& buttons) {
	try {
		auto machine = virtual_box.find_machine(name());
		vbox::Lock lock(machine, work_session, LockType_Shared);
		auto keyboard = work_session.console().keyboard();
		keyboard.putScancodes(buttons);
		keyboard.releaseKeys(buttons);
		return 0;
	} catch (const std::exception& error) {
		std::cout << "Pressing button on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int VmController::set_nic(const std::string& nic, bool is_enabled) {
	try {
		if (!config.count("nic")) {
			throw std::runtime_error("There's no nics in this vm");
		}

		auto& nics = config.at("nic");

		for (auto& nic_it: nics) {
			if (nic_it.at("name") == nic) {
				auto lock_machine = virtual_box.find_machine(name());

				vbox::Lock lock(lock_machine, work_session, LockType_Write);
				auto machine = work_session.machine();
				auto network_adapter = machine.getNetworkAdapter(nic_it.at("slot").get<uint32_t>());
				network_adapter.setEnabled(is_enabled);
				machine.save_settings();			
				return 0;
			}
		}

		throw std::runtime_error(std::string("There's no nic with name ") + nic);
	}
	catch (const std::exception& error) {
		std::cout << "(Un)Plugging nic in vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int VmController::set_link(const std::string& nic, bool is_connected) {
	try {
		if (!config.count("nic")) {
			throw std::runtime_error("There's no nics in this vm");
		}

		auto& nics = config.at("nic");

		for (auto& nic_it: nics) {
			if (nic_it.at("name") == nic) {
				auto lock_machine = virtual_box.find_machine(name());

				vbox::Lock lock(lock_machine, work_session, LockType_Shared);
				auto machine = work_session.machine();
				auto network_adapter = machine.getNetworkAdapter(nic_it.at("slot").get<uint32_t>());
				network_adapter.setCableConnected(is_connected);
				return 0;
			}
		}

		throw std::runtime_error(std::string("There's no nic with name ") + nic);
	}
	catch (const std::exception& error) {
		std::cout << "(Un)Plugging link on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

bool VmController::is_plugged(std::shared_ptr<FlashDriveController> fd) {
	return (plugged_fds.find(fd) != plugged_fds.end());
}

int VmController::plug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	try {
		if (plugged_fds.find(fd) != plugged_fds.end()) {
			throw std::runtime_error("This flash drive is already attached to this vm");
		}

		auto lock_machine = virtual_box.find_machine(name());
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

		machine.attach_device("USB", empty_slot, 0, DeviceType_HardDisk, fd->handle);
		machine.save_settings();
		plugged_fds.insert(fd);
		return 0;
	} catch (const std::exception& error) {
		std::cout << "Plugging flash drive on vm " << name() << ": " << error << std::endl;
		return 1;
	}
}

int VmController::unplug_flash_drive(std::shared_ptr<FlashDriveController> fd) {
	try {
		if (plugged_fds.find(fd) == plugged_fds.end()) {
			throw std::runtime_error("This flash drive is not plugged to this vm");
		}

		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);

		auto machine = work_session.machine();

		auto attachments = machine.medium_attachments_of_controller("USB");

		for (auto& attachment: attachments) {
			if (attachment.medium().handle == fd->handle.handle) {
				machine.detach_device("USB", attachment.port(), attachment.device());
			}
		}
		machine.save_settings();
		plugged_fds.erase(fd);
		return 0;
	} catch (const std::exception& error) {
		std::cout << "Unplugging flash drive from vm " << name() << ": " << error << std::endl;
		return 1;
	}
}

int VmController::plug_dvd(fs::path path) {
	try {
		auto lock_machine = virtual_box.find_machine(name());
		
		//auto machine = work_session.machine();
		auto mediums = lock_machine.medium_attachments_of_controller("IDE");
		for (auto& medium: mediums) {
			if (medium.port() == 1) {
				if (unplug_dvd()) {
					throw std::runtime_error("error while unplugging already attached dvd");
				}
			}
		}
		
		if (path.is_relative()) {
			path = fs::absolute(path);
		}

		vbox::Medium dvd = virtual_box.open_medium(path,
				DeviceType_DVD, AccessMode_ReadOnly, false);

		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		machine.mount_medium("IDE", 1, 0, dvd, false);

		return 0;
	} catch (const std::exception& error) {
		std::cout << "Plugging dvd " << path << " to vm " << name() << ": " << error << std::endl;
		return 1;
	}
}

int VmController::unplug_dvd() {
	try {
		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);

		auto machine = work_session.machine();
		auto mediums = machine.medium_attachments_of_controller("IDE");
		bool found = false;
		for (auto& medium: mediums) {
			if (medium.port() == 1) {
				found = true;
				break;
			}
		}

		if (!found) {
			throw std::runtime_error("No dvd is attached");
		}

		machine.unmount_medium("IDE", 1, 0, false);

		return 0;
	} catch (const std::exception& error) {
		std::cout << "Unplugging dvd from vm " << name() << ": " << error << std::endl;
		return 1;
	}
}

int VmController::start() {
	try {
		auto machine = virtual_box.find_machine(name());
		machine.launch_vm_process(start_session, "headless").wait_and_throw_if_failed();
		start_session.unlock_machine();
		return 0;
	}
	catch (const std::exception& error) {
		std::cout << "Starting vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int VmController::stop() {
	try {
		//In the end of stop we should enter session state UNLOCKED (even if the vm was being viewed by the user in GUI)
		//So we lock our machine, then destroy lock and wait for session state to become unlocked
		auto machine = virtual_box.find_machine(name());
		{
			vbox::Lock lock(machine, work_session, LockType_Shared);
			work_session.console().power_down().wait_and_throw_if_failed();
		}
		for (int i = 0; i < 10; i++) {
			if (machine.session_state() == SessionState_Unlocked) {
				return 0;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		throw std::runtime_error("timeout for stop has expired");
	}
	catch (const std::exception& error) {
		std::cout << "Stopping vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int VmController::type(const std::string& text) {
	try {
		auto machine = virtual_box.find_machine(name());
		vbox::Lock lock(machine, work_session, LockType_Shared);
		auto keyboard = work_session.console().keyboard();

		for (auto c: text) {
			auto buttons = charmap.find(c);
			if (buttons == charmap.end()) {
				throw std::runtime_error("Unknown character to type");
			}

			keyboard.putScancodes(buttons->second);
			keyboard.releaseKeys(buttons->second);
			std::this_thread::sleep_for(std::chrono::milliseconds(20)); //Fuck, it's even in vboxmanage sources
		}
		return 0;
	} catch (const std::exception& error) {
		std::cout << "Typing on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int VmController::wait(const std::string& text, const std::string& time) {
	try {
		auto lock_machine = virtual_box.find_machine(name());
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
			std::this_thread::sleep_for(std::chrono::seconds(1));
			display.get_screen_resolution(0, &width, &height, &bits_per_pixel, &x_origin, &y_origin, &guest_monitor_status);

			if (!width || !height) {
				return -1;
			}
		}

		auto timeout = std::chrono::system_clock::now() + std::chrono::seconds(time_to_seconds(time));

		while (std::chrono::system_clock::now() < timeout) {
			vbox::SafeArray safe_array = display.take_screen_shot_to_array(0, width, height, BitmapFormat_PNG);
			vbox::ArrayOut array_out = safe_array.copy_out(VT_UI1);

			PIX* img = pixReadMemPng(array_out.data, array_out.data_size);
			API::instance().tesseract_api->SetImage(img);

			auto out_text = API::instance().tesseract_api->GetUTF8Text();
			std::string out_string(out_text);
			delete [] out_text;
			pixDestroy(&img);

			std::cout << out_string << std::endl;

			if (out_string.find(text) != std::string::npos) {
				return 0;
			}

			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
		return -1;
	} catch (const std::exception& error) {
		std::cout << "Waiting on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}

int VmController::run(const fs::path& exe, std::vector<std::string> args) {
	try {
		args.insert(args.begin(), "--");
		uint32_t timeout = 10 * 60 * 1000; //10 mins

		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		//1) Open the session

		auto machine = work_session.machine();
		auto login = machine.getExtraData("login");
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
		auto gprocess = gsession.process_create(exe, args, {}, create_flags, timeout);

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
						std::string stderr(stderr_read.begin(), stderr_read.end());
						std::cout << stderr;
					}
					auto stdout_read = gprocess.read(1, 0x00010000, 0); //read 64 KBytes
					if (stdout_read.size() > 1) {
						std::string stdout(stdout_read.begin(), stdout_read.end());
						std::cout << stdout;
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
		std::cout << "Run guest process error: " << error << std::endl;
		return -1;
	}
}

bool VmController::has_snapshot(const std::string& snapshot) {
	try {
		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		return machine.hasSnapshot(snapshot);
	} catch (const std::exception& error) {
		std::cout << "Has snapshot on vm " << name() << ": " << error << std::endl;
		return false;
	}
}

bool VmController::is_defined() const {
	std::vector<vbox::Machine> machines = virtual_box.machines();
	for (auto& machine: machines) {
		if (machine.name() == name()) {
			return true;
		}
	}
	return false;
}

bool VmController::is_running() {
	try {
		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto machine = work_session.machine();
		return (machine.state() == MachineState_Running);
	} catch (const std::exception& error) {
		std::cout << "Is running on vm " << name() << ": " << error << std::endl;
		return false;
	}
}

void VmController::delete_snapshot_with_children(vbox::Snapshot& snapshot) {
	auto children = snapshot.children();

	if (children.size()) {
		for (auto& snap: children) {
			delete_snapshot_with_children(snap);
		}
	}

	auto machine = work_session.machine();
	machine.deleteSnapshot(snapshot).wait_and_throw_if_failed();
}

bool VmController::is_additions_installed() {
	try {
		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);
		auto facilities = work_session.console().guest().facilities();

		for (auto& facility: facilities) {
			if (facility.type() == AdditionsFacilityType_VBoxService) {
				return true;
			}
		}
		return false;
	} catch (const std::exception& error) {
		std::cout << "Is additions installed on vm " << name() << ": " << error << std::endl;
		return false;
	}
}

void VmController::copy_dir_to_guest(const fs::path& src, const fs::path& dst, vbox::GuestSession& gsession) {
	gsession.directory_create(dst);

	for (auto& file: fs::directory_iterator(src)) {
		if (fs::is_regular_file(file)) {
			gsession.file_copy_to_guest(file, dst / "/").wait_and_throw_if_failed();
		} else if (fs::is_directory(file)) {
			copy_dir_to_guest(file, dst / file.path().filename(), gsession);
		} //else continue
	}
}

int VmController::copy_to_guest(const fs::path& src, const fs::path& dst) {
	try {
		//1) if there's no src on host - fuck you
		if (!fs::exists(src)) {
			throw std::runtime_error(std::string("Source file/folder doens't exist on host: ") + std::string(src));
		}

		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);

		auto machine = work_session.machine();
		auto login = machine.getExtraData("login");
		auto password = machine.getExtraData("password");

		if (!login.length()) {
			throw std::runtime_error("Attribute login is not specified");
		}

		if (!password.length()) {
			throw std::runtime_error("Attribute login is not specified");
		}
		//1) Open the session
		auto gsession = work_session.console().guest().create_session(login, password);
		
		//2) if dst doesn't exist on guest - fuck you
		if (!gsession.directory_exists(dst)) {
			throw std::runtime_error(std::string("Directory to copy doesn't exist on guest: ") + std::string(dst));
		}

		//3) If target folder already exists on guest - fuck you
		fs::path target_name = dst / src.filename();
		if (gsession.directory_exists(target_name) || gsession.file_exists(target_name)) {
			throw std::runtime_error(std::string("Directory or file already exists on guest: ") + std::string(target_name));
		}

		//4) Now we're all set
		if (fs::is_regular_file(src)) {
			gsession.file_copy_to_guest(src, dst / "/").wait_and_throw_if_failed();
		} else if (fs::is_directory(src)) {
			copy_dir_to_guest(src, target_name, gsession);
		} else {
			throw std::runtime_error(std::string("Unknown type of file: ") + std::string(target_name));
		}
		return 0;
	} catch (const std::exception& error) {
		std::cout << "copy_to_guest on vm " << name() << ": " << error << std::endl;
		return 1;
	}
}

int VmController::remove_from_guest(const fs::path& obj) {
	try {
		auto lock_machine = virtual_box.find_machine(name());
		vbox::Lock lock(lock_machine, work_session, LockType_Shared);

		auto machine = work_session.machine();
		auto login = machine.getExtraData("login");
		auto password = machine.getExtraData("password");

		if (!login.length()) {
			throw std::runtime_error("Attribute login is not specified");
		}

		if (!password.length()) {
			throw std::runtime_error("Attribute login is not specified");
		}

		auto gsession = work_session.console().guest().create_session(login, password);

		//directory handling differs from file handling
		if (gsession.directory_exists(obj)) {
			gsession.directory_remove_recursive(obj);
		} else if (gsession.file_exists(obj)) {
			gsession.file_remove(obj);
		} else {
			throw std::runtime_error(std::string("Target object doesn't exist on vm: ") + std::string(obj));
		}

		return 0;
	} catch (const std::exception& error) {
		std::cout << "remove_from_guest on vm " << name() << ": " << error << std::endl;
		return -1;
	}
}
