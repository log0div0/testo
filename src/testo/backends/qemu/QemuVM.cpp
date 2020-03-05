
#include "QemuVM.hpp"
#include "QemuFlashDrive.hpp"
#include "QemuGuestAdditions.hpp"
#include "QemuEnvironment.hpp"

#include <fmt/format.h>
#include <thread>

QemuVM::QemuVM(const nlohmann::json& config_): VM(config_),
	qemu_connect(vir::connect_open("qemu:///system"))
{
	if (!config.count("name")) {
		throw std::runtime_error("Constructing QemuVM " + id() + " error: field NAME is not specified");
	}

	if (!config.count("ram")) {
		throw std::runtime_error("Constructing QemuVM " + id() + " error: field RAM is not specified");
	}

	if (!config.count("cpus")) {
		throw std::runtime_error("Constructing QemuVM " + id() + " error: field CPUS is not specified");
	}

	if (!config.count("disk_size")) {
		throw std::runtime_error("Constructing QemuVM error: field DISK SIZE is not specified");
	}

	if (config.count("nic")) {
		auto nics = config.at("nic");
		for (auto& nic: nics) {
			if (!nic.count("attached_to")) {
				throw std::runtime_error("Constructing QemuVM error: field attached_to is not specified for the nic " +
					nic.at("name").get<std::string>());
			}

			if (nic.at("attached_to").get<std::string>() == "internal") {
				if (!nic.count("network")) {
					throw std::runtime_error("Constructing QemuVM error: nic " +
					nic.at("name").get<std::string>() + " has type internal, but field network is not specified");
				}
			}

			if (nic.count("mac")) {
				std::string mac = nic.at("mac").get<std::string>();
				if (!is_mac_correct(mac)) {
					throw std::runtime_error(std::string("Incorrect mac string: ") + mac);
				}
			}

			if (nic.at("attached_to").get<std::string>() == "nat") {
				if (nic.count("network")) {
					throw std::runtime_error("Constructing QemuVM error: nic " +
					nic.at("name").get<std::string>() + " has type NAT, you must not specify field network");
				}
			}

			if (nic.count("adapter_type")) {
				//ne2k_pci,i82551,i82557b,i82559er,rtl8139,e1000,pcnet,virtio,sungem
				std::string driver = nic.at("adapter_type").get<std::string>();
				if (driver != "ne2k_pci" &&
					driver != "i82551" &&
					driver != "i82557b" &&
					driver != "i82559er" &&
					driver != "rtl8139" &&
					driver != "e1000" &&
					driver != "pcnet" &&
					driver != "virtio" &&
					driver != "sungem")
				{
					throw std::runtime_error("Constructing QemuVM error: nic " +
						nic.at("name").get<std::string>() + " has unsupported adaptertype internal: " + driver);
				}
			}
		}

		for (uint32_t i = 0; i < nics.size(); i++) {
			for (uint32_t j = i + 1; j < nics.size(); j++) {
				if (nics[i].at("name") == nics[j].at("name")) {
					throw std::runtime_error("Constructing QemuVM error: two identical NIC names: " +
						nics[i].at("name").get<std::string>());
				}
			}
		}
	}

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
		{"LEFTMETA", 125},
		{"RIGHTMETA", 126},
		{"SCROLLUP", 177},
		{"SCROLLDOWN", 178},
	});
}

QemuVM::~QemuVM() {
	if (!is_defined()) {
		remove_disk();
	}
}

void QemuVM::install() {
	try {
		if (is_defined()) {
			undefine();
		}

		//now create disks
		create_disk();

		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");
		fs::path volume_path = pool.path() / (id() + ".img");

		std::string string_config = fmt::format(R"(
			<domain type='kvm'>
				<name>{}</name>
				<memory unit='MiB'>{}</memory>
				<vcpu placement='static'>{}</vcpu>
				<resource>
					<partition>/machine</partition>
				</resource>
				<os>
					<type>hvm</type>
					<boot dev='cdrom'/>
					<boot dev='hd'/>
				</os>
				<features>
					<acpi/>
					<apic/>
					<vmport state='off'/>
				</features>
				<cpu mode='host-model'>
					<model fallback='forbid'/>
				</cpu>
				<clock offset='utc'>
					<timer name='rtc' tickpolicy='catchup'/>
					<timer name='pit' tickpolicy='delay'/>
					<timer name='hpet' present='yes'/>
				</clock>
				<on_poweroff>destroy</on_poweroff>
				<on_reboot>restart</on_reboot>
				<on_crash>destroy</on_crash>
				<pm>
				</pm>
				<metadata>
					<testo:is_testo_related xmlns:testo='http://testo' value='true'/>
				</metadata>
				<devices>
					<disk type='file' device='disk'>
						<driver name='qemu' type='qcow2'/>
						<source file='{}'/>
						<target dev='hda' bus='ide'/>
					</disk>
					<disk type='file' device='cdrom'>
						<driver name='qemu' type='raw'/>
						<source file='{}'/>
						<target dev='hdb' bus='ide'/>
						<readonly/>
					</disk>
					<controller type='usb' index='0' model='ich9-ehci1'>
					</controller>
					<controller type='usb' index='0' model='ich9-uhci1'>
					</controller>
					<controller type='usb' index='0' model='ich9-uhci2'>
					</controller>
					<controller type='usb' index='0' model='ich9-uhci3'>
					</controller>
					<controller type='ide' index='0'>
					</controller>
					<controller type='virtio-serial' index='0'>
					</controller>
					<controller type='pci' index='0' model='pci-root'/>
					<serial type='pty'>
						<target type='isa-serial' port='0'>
							<model name='isa-serial'/>
						</target>
					</serial>
					<console type='pty'>
						<target type='serial' port='0'/>
					</console>
					<channel type='unix'>
						<target type='virtio' name='negotiator.0'/>
					</channel>
					<input type='tablet' bus='usb'>
					</input>
					<input type='mouse' bus='ps2'/>
					<input type='keyboard' bus='ps2'/>
					<graphics type='spice' autoport='yes'>
						<listen type='address'/>
						<image compression='off'/>
					</graphics>
					<sound model='ich6'>
					</sound>
					<video>
						<model type='qxl' heads='1' primary='yes'/>
					</video>
					<redirdev bus='usb' type='spicevmc'>
					</redirdev>
					<redirdev bus='usb' type='spicevmc'>
					</redirdev>
					<memballoon model='virtio'>
					</memballoon>
		)", id(), config.at("ram").get<uint32_t>(), config.at("cpus").get<uint32_t>(), volume_path.generic_string(), config.at("iso").get<std::string>(), id());

		uint32_t nic_count = 0;

		if (config.count("nic")) {
			auto nics = config.at("nic");
			for (auto& nic: nics) {
				//Complete redo
				std::string source_network = config.at("prefix").get<std::string>();

				source_network += nic.at("attached_to").get<std::string>();

				string_config += fmt::format(R"(
					<interface type='network'>
						<source network='{}'/>
				)", source_network);

				if (nic.count("mac")) {
					string_config += fmt::format("\n<mac address='{}'/>", nic.at("mac").get<std::string>());
				}

				if (nic.count("adapter_type")) {
					string_config += fmt::format("\n<model type='{}'/>", nic.at("adapter_type").get<std::string>());
				}

				//libvirt suggests that everything you do in aliases must be prefixed with "ua-nic-"
				std::string nic_name = std::string("ua-nic-");
				nic_name += nic.at("name").get<std::string>();
				string_config += fmt::format("\n<link state='up'/>");
				string_config += fmt::format("\n<alias name='{}'/>", nic_name);
				string_config += fmt::format("\n</interface>");

				nic_count++;
			}
		}

		string_config += "\n </devices> \n </domain>";

		pugi::xml_document xml_config;
		xml_config.load_string(string_config.c_str());
		qemu_connect.domain_define_xml(xml_config);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Performing install")));
	}
}

void QemuVM::undefine() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());


		if (state() != VmState::Stopped) {
			stop();
		}
		//delete the storage
		remove_disk();

		domain.undefine();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Undefining vm {}", id())));
	}
}

void QemuVM::make_snapshot(const std::string& snapshot) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		pugi::xml_document xml_config;
		xml_config.load_string(fmt::format(R"(
			<domainsnapshot>
				<name>{}</name>
			</domainsnapshot>
			)", snapshot).c_str());

		domain.snapshot_create_xml(xml_config);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Taking snapshot {}", snapshot)));
	}

}

void QemuVM::rollback(const std::string& snapshot) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto snap = domain.snapshot_lookup_by_name(snapshot);

		//Now let's take care of possible dvd discontingency
		std::string current_dvd = get_dvd_path();
		std::string snapshot_dvd = get_dvd_path(snap);

		if (current_dvd != snapshot_dvd) {
			//Possible variations:
			//If we have something plugged - let's unplug it
			if (current_dvd.length()) {
				if (domain.state() != VIR_DOMAIN_SHUTOFF) {
					stop();
				}
				unplug_dvd();
			}

			if (snapshot_dvd.length()) {
				plug_dvd(snapshot_dvd);
			}
		}

		//nics contingency
		if (config.count("nic")) {
			for (auto& nic: config.at("nic")) {

				std::string nic_name = nic.at("name").get<std::string>();
				auto currently_plugged = is_nic_plugged(nic_name);
				auto snapshot_plugged = is_nic_plugged(snap, nic_name);
				if (currently_plugged != snapshot_plugged) {
					if (domain.state() != VIR_DOMAIN_SHUTOFF) {
						stop();
					}

					set_nic(nic_name, snapshot_plugged);
				}
			}
		}

		//links contingency
		for (auto& nic: nics()) {
			auto currently_plugged = is_link_plugged(nic);
			auto snapshot_plugged = is_link_plugged(snap, nic);
			if (currently_plugged != snapshot_plugged) {
				set_link(nic, snapshot_plugged);
			}
		}

		std::string flash_attached = get_flash_img();
		if (flash_attached.length()) {
			detach_flash_drive();
		}

		domain.revert_to_snapshot(snap);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Performing rollback error"));
	}
}

void QemuVM::press(const std::vector<std::string>& buttons) {
	try {
		std::vector<uint32_t> keycodes;
		for (auto button: buttons) {
			std::transform(button.begin(), button.end(), button.begin(), toupper);
			keycodes.push_back(scancodes[button]);
		}
		qemu_connect.domain_lookup_by_name(id()).send_keys(VIR_KEYCODE_SET_LINUX, 0, keycodes);
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Pressing buttons error"));
	}
}

void QemuVM::mouse_move_abs(uint32_t x, uint32_t y) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto tmp_screen = screenshot();

		double x_pos = double(32768) / double(tmp_screen.width) * double(x);
		double y_pos = double(32768) / double(tmp_screen.height) * double(y);

		if ((int)x_pos == 0) {
			x_pos = 1;
		}

		if ((int)y_pos == 0) {
			y_pos = 1;
		}

		nlohmann::json json_command = nlohmann::json::parse(fmt::format(R"(
			{{
				"execute": "input-send-event",
				"arguments": {{
					"events": [
						{{
							"type": "abs",
							"data": {{
								"axis": "x",
								"value": {}
							}}

						}},
						{{
							"type": "abs",
							"data": {{
								"axis": "y",
								"value": {}
							}}
						}}
					]
				}}
			}}
		)", (int)x_pos, (int)y_pos));

		auto result = domain.monitor_command(json_command.dump());

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse move error"));
	}
}

void QemuVM::mouse_move_abs(const std::string& axis, uint32_t value) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto tmp_screen = screenshot();

		double pos;

		if (axis == "x") {
			pos = double(32768) / double(tmp_screen.width) * double(value);
		} else if (axis == "y") {
			pos = double(32768) / double(tmp_screen.height) * double(value);
		} else {
			throw std::runtime_error("Unknown axis: " + axis);
		}

		if ((int)pos == 0) {
			pos = 1;
		}

		nlohmann::json json_command = nlohmann::json::parse(fmt::format(R"(
			{{
				"execute": "input-send-event",
				"arguments": {{
					"events": [
						{{
							"type": "abs",
							"data": {{
								"axis": "{}",
								"value": {}
							}}
						}}
					]
				}}
			}}
		)", axis, (int)pos));

		auto result = domain.monitor_command(json_command.dump());

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse move error"));
	}
}

void QemuVM::mouse_move_rel(int x, int y) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		nlohmann::json json_command = nlohmann::json::parse(fmt::format(R"(
			{{
				"execute": "input-send-event",
				"arguments": {{
					"events": [
						{{
							"type": "rel",
							"data": {{
								"axis": "x",
								"value": {}
							}}
						}},
						{{
							"type": "rel",
							"data": {{
								"axis": "y",
								"value": {}
							}}
						}}
					]
				}}
			}}
		)", x, y));

		auto result = domain.monitor_command(json_command.dump());

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse move error"));
	}
}

void QemuVM::mouse_move_rel(const std::string& axis, int value) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		if (axis != "x" && axis != "y") {
			throw std::runtime_error("Unknown axis: " + axis);
		}

		nlohmann::json json_command = nlohmann::json::parse(fmt::format(R"(
			{{
				"execute": "input-send-event",
				"arguments": {{
					"events": [
						{{
							"type": "rel",
							"data": {{
								"axis": "{}",
								"value": {}
							}}
						}}
					]
				}}
			}}
		)", axis, value));

		auto result = domain.monitor_command(json_command.dump());

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse move error"));
	}
}

void QemuVM::mouse_press(const std::vector<MouseButton>& buttons) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		nlohmann::json json_command = R"(
			{
				"execute": "input-send-event",
				"arguments": {
					"events": [
					]
				}
			}
		)"_json;

		for (auto& button: buttons) {
			json_command.at("arguments").at("events").push_back({
				{"type", "btn"},
				{"data", {
					{"down", true},
					{"button", mouse_button_to_str(button)}
				}}
			});
		}

		auto result = domain.monitor_command(json_command.dump());

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse press buttons error"));
	}
}


void QemuVM::mouse_release(const std::vector<MouseButton>& buttons) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		nlohmann::json json_command = R"(
			{
				"execute": "input-send-event",
				"arguments": {
					"events": [
					]
				}
			}
		)"_json;

		for (auto& button: buttons) {
			json_command.at("arguments").at("events").push_back({
				{"type", "btn"},
				{"data", {
					{"down", false},
					{"button", mouse_button_to_str(button)}
				}}
			});
		}

		auto result = domain.monitor_command(json_command.dump());

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse release buttons error"));
	}
}

bool QemuVM::is_nic_plugged(const std::string& nic) const {
	try {
		auto nic_name = std::string("ua-nic-") + nic;
		auto config = qemu_connect.domain_lookup_by_name(id()).dump_xml();
		auto devices = config.first_child().child("devices");

		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			if (std::string(nic_node.child("alias").attribute("name").value()) == nic_name) {
				return true;
			}
		}
		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking if nic {} is plugged", nic)));
	}
}

bool QemuVM::is_nic_plugged(vir::Snapshot& snapshot, const std::string& nic) {
	try {
		auto nic_name = std::string("ua-nic-") + nic;
		auto config = snapshot.dump_xml();
		auto devices = config.first_child().child("domain").child("devices");

		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			if (std::string(nic_node.child("alias").attribute("name").value()) == nic_name) {
				return true;
			}
		}

		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking if nic {} is plugged from snapshot", nic)));
	}

}

void QemuVM::attach_nic(const std::string& nic) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		std::string string_config;

		for (auto& nic_json: config.at("nic")) {
			if (nic_json.at("name") == nic) {
				std::string source_network("testo-");

				if (nic_json.at("attached_to").get<std::string>() == "internal") {
					source_network += nic_json.at("network").get<std::string>();
				}

				if (nic_json.at("attached_to").get<std::string>() == "nat") {
					source_network += "nat";
				}

				string_config = fmt::format(R"(
					<interface type='network'>
						<source network='{}'/>
				)", source_network);

				if (nic_json.count("mac")) {
					string_config += fmt::format("\n<mac address='{}'/>", nic_json.at("mac").get<std::string>());
				}

				if (nic_json.count("adapter_type")) {
					string_config += fmt::format("\n<model type='{}'/>", nic_json.at("adapter_type").get<std::string>());
				}

				//libvirt suggests that everything you do in aliases must be prefixed with "ua-nic-"
				std::string nic_name = std::string("ua-nic-");
				nic_name += nic_json.at("name").get<std::string>();
				string_config += fmt::format("\n<link state='up'/>");
				string_config += fmt::format("\n<alias name='{}'/>", nic_name);
				string_config += fmt::format("\n</interface>");

				break;
			}
		}

		//TODO: check if CURRENT is enough
		std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		pugi::xml_document nic_config;
		nic_config.load_string(string_config.c_str());

		domain.attach_device(nic_config, flags);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Attaching nic {}", nic)));
	}

}

void QemuVM::detach_nic(const std::string& nic) {
	try {
		auto nic_name = std::string("ua-nic-") + nic;
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		//TODO: check if CURRENT is enough
		std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			if (std::string(nic_node.child("alias").attribute("name").value()) == nic_name) {
				domain.detach_device(nic_node, flags);
				return;
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Detaching nic {}", nic)));
	}
}

void QemuVM::set_nic(const std::string& nic, bool is_enabled) {
	try {
		if (is_enabled) {
			attach_nic(nic);
		} else {
			detach_nic(nic);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting nic {}", nic)));
	}
}

bool QemuVM::is_link_plugged(const pugi::xml_node& devices, const std::string& nic) const {
	try {
		std::string nic_name = std::string("ua-nic-") + nic;

		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			if (std::string(nic_node.child("alias").attribute("name").value()) == nic_name) {
				if (nic_node.child("link").empty()) {
					return false;
				}

				std::string state = nic_node.child("link").attribute("state").value();

				if (state == "up") {
					return true;
				} else if (state == "down") {
					return false;
				} else {
					throw std::runtime_error("Unknown link state");
				}
			}
		}
		return false;
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(fmt::format("Checking if nic {} is plugged", nic)));
	}
}

bool QemuVM::is_link_plugged(vir::Snapshot& snapshot, const std::string& nic) {
	try {
		auto config = snapshot.dump_xml();
		return is_link_plugged(config.first_child().child("domain").child("devices"), nic);
	} catch (const std::exception& error) {
		throw_with_nested(std::runtime_error(fmt::format("Checking if nic {} is plugged from snapshot", nic)));
	}

}

bool QemuVM::is_link_plugged(const std::string& nic) const {
	try {
		auto config = qemu_connect.domain_lookup_by_name(id()).dump_xml();
		return is_link_plugged(config.first_child().child("devices"), nic);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking link status on nic {}", nic)));
	}
}

void QemuVM::set_link(const std::string& nic, bool is_connected) {
	try {
		std::string nic_name = std::string("ua-nic-") + nic;
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");
		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			if (std::string(nic_node.attribute("type").value()) != "network") {
				continue;
			}

			if (std::string(nic_node.child("alias").attribute("name").value()) == nic_name) {
				if (is_connected) { //connect link
					//if we have set link attribute - just change state to up
					if (!nic_node.child("link").empty()) {
						nic_node.child("link").attribute("state").set_value("up");
					}
				} else { //disconnect link
					//if we have set link attribute - set it to down
					if (!nic_node.child("link").empty()) {
						nic_node.child("link").attribute("state").set_value("down");
					} else {
						auto link = nic_node.insert_child_before("link", nic_node.child("alias"));
						link.append_attribute("state") = "down";
					}
				}

				std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

				if (domain.is_active()) {
					flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
				}

				domain.update_device(nic_node, flags);
				break;
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting link status on nic {}", nic)));
	}
}

std::string QemuVM::get_flash_img() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		std::string result = "";

		for (auto disk = devices.child("disk"); disk; disk = disk.next_sibling("disk")) {
			if (std::string(disk.attribute("device").value()) != "disk") {
				continue;
			}

			if (std::string(disk.child("target").attribute("dev").value()) == "sdb") {
				result = disk.child("source").attribute("file").value();
			}
		}

		return result;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Getting flash image"));
	}
}

bool QemuVM::is_flash_plugged(std::shared_ptr<FlashDrive> fd) {
	try {
		return get_flash_img().length();
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking if flash drive {} is pluged", fd->name())));
	}
}

void QemuVM::attach_flash_drive(const std::string& img_path) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		std::string string_config = fmt::format(R"(
			<disk type='file'>
				<driver name='qemu' type='qcow2'/>
				<source file='{}'/>
				<target dev='sdb' bus='usb'/>
			</disk>
			)", img_path);

		//we just need to create new device
		//TODO: check if CURRENT is enough
		std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CONFIG, VIR_DOMAIN_DEVICE_MODIFY_CURRENT};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		pugi::xml_document disk_config;
		disk_config.load_string(string_config.c_str());

		domain.attach_device(disk_config, flags);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Attaching flash drive {}", img_path)));
	}

}

void QemuVM::plug_flash_drive(std::shared_ptr<FlashDrive> fd) {
	try {
		attach_flash_drive(fd->img_path());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Plugging flash drive {}", fd->name())));
	}
}

void QemuVM::detach_flash_drive() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		//TODO: check if CURRENT is enough
		std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		for (auto disk = devices.child("disk"); disk; disk = disk.next_sibling("disk")) {
			if (std::string(disk.attribute("device").value()) != "disk") {
				continue;
			}

			if (std::string(disk.child("target").attribute("dev").value()) == "sdb") {
				domain.detach_device(disk, flags);
				break;
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Detaching flash drive"));
	}
}

//for now it's just only one flash drive possible
void QemuVM::unplug_flash_drive(std::shared_ptr<FlashDrive> fd) {
	try {
		detach_flash_drive();
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Unplugging flash drive {}", fd->name())));
	}
}


bool QemuVM::is_dvd_plugged() const {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");
		return !bool(cdrom.child("source").empty());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Checking if dvd is plugged"));
	}
}

std::string QemuVM::get_dvd_path() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");
		if (cdrom.child("source").empty()) {
			return "";
		}
		return cdrom.child("source").attribute("file").value();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Getting dvd path"));
	}
}


std::string QemuVM::get_dvd_path(vir::Snapshot& snap) {
	try {
		auto config = snap.dump_xml();
		auto cdrom = config.first_child().child("domain").child("devices").find_child_by_attribute("device", "cdrom");
		if (cdrom.child("source").empty()) {
			return "";
		}
		return cdrom.child("source").attribute("file").value();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Getting dvd path from snapshot"));
	}
}

void QemuVM::plug_dvd(fs::path path) {
	try {
		if (!fs::exists(path)) {
			throw std::runtime_error(std::string("specified iso file does not exist: ")
				+ path.generic_string());
		}

		if (!fs::is_regular_file(path)) {
			throw std::runtime_error(std::string("specified iso is not a regular file: ")
				+ path.generic_string());
		}
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");

		if (!cdrom.child("source").empty()) {
			throw std::runtime_error("Some dvd is already plugged in");
		}

		std::string string_config = fmt::format(R"(
			<disk type='file' device='cdrom'>
				<driver name='qemu' type='raw'/>
				<source file='{}'/>
				<backingStore/>
				<target dev='hdb' bus='ide'/>
				<readonly/>
				<alias name='ide0-0-1'/>
				<address type='drive' controller='0' bus='0' target='0' unit='1'/>
			</disk>
		)", path.generic_string().c_str());

		std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CONFIG, VIR_DOMAIN_DEVICE_MODIFY_CURRENT};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		pugi::xml_document dvd_config;
		dvd_config.load_string(string_config.c_str());

		domain.update_device(dvd_config, flags);
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("plugging dvd {}", path.generic_string())));
	}
}

void QemuVM::unplug_dvd() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");

		if (cdrom.child("source").empty()) {
			throw std::runtime_error("Dvd is already unplugged");
		}

		cdrom.remove_child("source");

		std::vector flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		domain.update_device(cdrom, flags);
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error("Unplugging dvd"));
	}

}

void QemuVM::start() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.start();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Starting vm"));
	}
}

void QemuVM::stop() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.stop();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Stopping vm"));
	}
}

void QemuVM::power_button() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.shutdown();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Shutdowning vm"));
	}
}

void QemuVM::suspend() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.suspend();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Suspending vm"));
	}
}

void QemuVM::resume() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.resume();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Resuming vm"));
	}
}

stb::Image QemuVM::screenshot() {
	auto domain = qemu_connect.domain_lookup_by_name(id());
	auto stream = qemu_connect.new_stream();
	auto mime = domain.screenshot(stream);

	if (!screenshot_buffer.size()) {
		screenshot_buffer.resize(10'000'000);
	}

	size_t bytes = stream.recv_all(screenshot_buffer.data(), screenshot_buffer.size());

	stream.finish();

	stb::Image screenshot(screenshot_buffer.data(), bytes);
	return screenshot;
}

int QemuVM::run(const fs::path& exe, std::vector<std::string> args, uint32_t timeout_milliseconds) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		QemuGuestAdditions helper(domain);

		std::string command = exe.generic_string();
		for (auto& arg: args) {
			command += " ";
			command += arg;
		}

		return helper.execute(command, timeout_milliseconds);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Run guest process"));
	}
}

bool QemuVM::has_snapshot(const std::string& snapshot) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto snapshots = domain.snapshots();
		for (auto& snap: snapshots) {
			if (snap.name() == snapshot) {
				return true;
			}
		}
		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking whether vm has snapshot {}", snapshot)));
	}
}

void QemuVM::delete_snapshot(const std::string& snapshot) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto vir_snapshot = domain.snapshot_lookup_by_name(snapshot);
		vir_snapshot.destroy();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Deleting snapshot with children"));
	}
}

bool QemuVM::is_defined() const {
	auto domains = qemu_connect.domains({VIR_CONNECT_LIST_DOMAINS_PERSISTENT});
	for (auto& domain: domains) {
		if (domain.name() == id()) {
			return true;
		}
	}
	return false;
}

VmState QemuVM::state() const {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto state = domain.state();
		if (state == VIR_DOMAIN_SHUTOFF) {
			return VmState::Stopped;
		} else if (state == VIR_DOMAIN_RUNNING) {
			return VmState::Running;
		} else if (state == VIR_DOMAIN_PAUSED) {
			return VmState::Suspended;
		} else {
			return VmState::Other;
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Checking whether vm is running"));
	}
}

bool QemuVM::is_additions_installed() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		QemuGuestAdditions helper(domain);
		return helper.is_avaliable();
	} catch (const std::exception& error) {
		return false;
	}
}

std::string QemuVM::get_tmp_dir() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		QemuGuestAdditions helper(domain);
		return helper.get_tmp_dir();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Getting tmp directory path on guest"));
	}
}

void QemuVM::copy_to_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds) {
	try {
		//1) if there's no src on host - fuck you
		if (!fs::exists(src)) {
			throw std::runtime_error("Source file/folder does not exist on host: " + src.generic_string());
		}

		auto domain = qemu_connect.domain_lookup_by_name(id());
		QemuGuestAdditions helper(domain);

		helper.copy_to_guest(src, dst, timeout_milliseconds);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Copying file(s) to the guest"));
	}
}

void QemuVM::copy_from_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds) {
	try {
		if (src.is_relative()) {
			throw std::runtime_error(fmt::format("Source path on vm must be absolute"));
		}

		auto domain = qemu_connect.domain_lookup_by_name(id());
		QemuGuestAdditions helper(domain);

		helper.copy_from_guest(src, dst, timeout_milliseconds);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Copying file(s) to the guest"));
	}
}

void QemuVM::remove_from_guest(const fs::path& obj) {
	//TODO!!
}

void QemuVM::remove_disk() {
	try {
		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");

		auto vol_name = id() + ".img";

		for (auto& vol: pool.volumes()) {
			if (vol.name() == vol_name) {
				vol.erase();
				break;
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Removing existing disks"));
	}

}

void QemuVM::create_disk() {
	try {
		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");
		pugi::xml_document xml_config;
		xml_config.load_string(fmt::format(R"(
			<volume type='file'>
				<name>{}.img</name>
				<source>
				</source>
				<capacity unit='M'>{}</capacity>
				<target>
					<path>{}/{}.img</path>
					<format type='qcow2'/>
					<permissions>
					</permissions>
					<timestamps>
					</timestamps>
					<compat>1.1</compat>
					<features>
						<lazy_refcounts/>
					</features>
				</target>
			</volume>
		)", id(), config.at("disk_size").get<uint32_t>(), pool.path().generic_string(), id()).c_str());

		auto volume = pool.volume_create_xml(xml_config, {VIR_STORAGE_VOL_CREATE_PREALLOC_METADATA});
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Creating disks"));
	}
}

std::string QemuVM::mouse_button_to_str(MouseButton btn) {
	switch (btn) {
		case Left: return "left";
		case Right: return "right";
		case Middle: return "middle";
		case WheelUp: return "wheel-up";
		case WheelDown: return "wheel-down";
		default: throw std::runtime_error("Unknown button: " + btn);
	}
}
