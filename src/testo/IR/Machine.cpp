
#include "Machine.hpp"
#include "../backends/Environment.hpp"
#include <coro/Timer.h>

namespace IR {

std::shared_ptr<::VM> Machine::vm() const {
	if (!_vm) {
		_vm = env->create_vm(config);
	}
	return _vm;
}

std::string Machine::type() const {
	return "virtual machine";
}

std::string Machine::id() const {
	return vm()->id();
}

bool Machine::is_defined() const {
	return Controller::is_defined() && vm()->is_defined();
}

void Machine::create() {
	try {
		undefine();

		if (fs::exists(get_metadata_dir())) {
			if (!fs::remove_all(get_metadata_dir())) {
				throw std::runtime_error("Error deleting metadata dir " + get_metadata_dir().generic_string());
			}
		}

		vm()->install();

		auto vm_config = config;

		nlohmann::json metadata;

		if (!fs::create_directory(get_metadata_dir())) {
			throw std::runtime_error("Error creating metadata dir " + get_metadata_dir().generic_string());
		}

		vm_config.erase("src_file");
		vm_config.erase("metadata");

		metadata["vm_config"] = vm_config.dump();

		if (vm_config.count("iso")) {
			fs::path iso_file = vm_config.at("iso").get<std::string>();
			metadata["iso_signature"] = file_signature(iso_file);
		}

		if (vm_config.count("loader")) {
			fs::path loader_file = vm_config.at("loader").get<std::string>();
			metadata["loader_signature"] = file_signature(loader_file);
		}

		if (vm_config.count("disk")) {
			for (auto& disk: vm_config.at("disk")) {
				if (disk.count("source")) {
					std::string signature_name = std::string("disk_signature@") + disk.at("name").get<std::string>();
					metadata[signature_name] = file_signature(disk.at("source").get<std::string>());
				}
			}
		}

		write_metadata_file(main_file(), metadata);

		if (config.count("nic")) {
			auto nics = config.at("nic");
			for (auto& nic: nics) {
				plug_nic(nic.at("name").get<std::string>());
			}
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("creating vm"));
	}
}

void Machine::undefine() {
	try {
		if (vm()->is_defined() && vm()->state() != VmState::Stopped) {
			vm()->stop();
		}

		vm()->remove_disks();

		auto metadata_dir = get_metadata_dir();
		if (!vm()->is_defined()) {
			if (fs::exists(metadata_dir)) {
				//The check would be valid only if we have the main file

				if (!fs::remove_all(metadata_dir)) {
					throw std::runtime_error("Error deleting metadata dir " + metadata_dir.generic_string());
				}
			}
			return;
		}

		if (Controller::has_snapshot("_init")) {
			delete_snapshot_with_children("_init");
		}

		vm()->undefine();

		if (!fs::remove_all(metadata_dir)) {
			throw std::runtime_error("Error deleting metadata dir " + metadata_dir.generic_string());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("undefining vm controller"));
	}
}

bool Machine::is_nic_plugged(const std::string& nic) {
	return nic_pci_map[nic].length();
}

void Machine::plug_nic(const std::string& nic) {
	try {
		std::string pci_addr = vm()->attach_nic(nic);
		nic_pci_map[nic] = pci_addr;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("attaching nic " + nic));
	}
}

void Machine::unplug_nic(const std::string& nic) {
	try {
		vm()->detach_nic(nic_pci_map[nic]);
		nic_pci_map[nic] = "";
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("detaching nic " + nic));
	}
}

bool Machine::is_link_plugged(const std::string& nic) {
	try {
		auto pci_addr = nic_pci_map[nic];
		if (!pci_addr.length()) {
			throw std::runtime_error("Internal error: nic " + nic + " is not plugged");
		}
		return vm()->is_link_plugged(pci_addr);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("checking link is plugged: " + nic));
	}

}

void Machine::plug_link(const std::string& nic) {
	try {
		auto pci_addr = nic_pci_map[nic];
		if (!pci_addr.length()) {
			throw std::runtime_error("Internal error: nic " + nic + " is not plugged");
		}
		vm()->set_link(pci_addr, true);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Plugging link: " + nic));
	}
}

void Machine::unplug_link(const std::string& nic) {
	try {
		auto pci_addr = nic_pci_map[nic];
		if (!pci_addr.length()) {
			throw std::runtime_error("Internal error: nic " + nic + " is not plugged");
		}
		return vm()->set_link(pci_addr, false);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Plugging link: " + nic));
	}
}

void Machine::create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed)
{
	try {
		if (hypervisor_snapshot_needed && vm()->is_flash_plugged(nullptr)) {
			throw std::runtime_error("Can't take hypervisor snapshot with a flash drive plugged in. Please unplug the flash drive before the end of the test");
		}

		if (current_held_mouse_button != MouseButton::None) {
			throw std::runtime_error("There is some mouse button held down. Please release it before the end of test");
		}

		if (current_held_keyboard_buttons.size()) {
			throw std::runtime_error("There are some keyboard buttons held down. Please release them before the end of test");
		}

		if (has_snapshot(snapshot)) {
			delete_snapshot_with_children(snapshot);
		}

		//1) Let's try and create the actual snapshot. If we fail then no additional work

		nlohmann::json opaque = nlohmann::json::object();
		if (hypervisor_snapshot_needed) {
			opaque = vm()->make_snapshot(snapshot);
		}

		//Where to store new metadata file?
		fs::path metadata_file = get_metadata_dir();
		metadata_file /= vm()->id() + "_" + snapshot;

		nlohmann::json metadata;
		metadata["cksum"] = cksum;
		metadata["children"] = nlohmann::json::array();
		metadata["parent"] = current_state;
		metadata["opaque"] = opaque;
		metadata["metadata_version"] = "2";

		//nics
		metadata["nics"] = nlohmann::json::object();
		for (auto& nic: nic_pci_map) {
			metadata["nics"][nic.first] = nic.second;
		}

		write_metadata_file(metadata_file, metadata);

		//link parent to a child
		if (current_state.length()) {
			fs::path parent_metadata_file = get_metadata_dir();
			parent_metadata_file /= vm()->id() + "_" + current_state;
			auto parent_metadata = read_metadata_file(parent_metadata_file);
			parent_metadata.at("children").push_back(snapshot);
			write_metadata_file(parent_metadata_file, parent_metadata);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("creating snapshot " + snapshot));
	}
}

void Machine::restore_snapshot(const std::string& snapshot) {
	nic_pci_map.clear();

	fs::path metadata_file = get_metadata_dir();
	metadata_file /= vm()->id() + "_" + snapshot;

	auto metadata = read_metadata_file(metadata_file);
	auto& nics = metadata.at("nics");
	for (auto it = nics.begin(); it != nics.end(); ++it) {
		nic_pci_map[it.key()] = it.value().get<std::string>();
	}

	vm()->rollback(snapshot, metadata.at("opaque"));
	current_state = snapshot;
}

void Machine::delete_snapshot_with_children(const std::string& snapshot)
{
	try {
		//This thins needs to be recursive
		//I guess... go through the children and call recursively on them
		fs::path metadata_file = get_metadata_dir();
		metadata_file /= vm()->id() + "_" + snapshot;

		auto metadata = read_metadata_file(metadata_file);

		for (auto& child: metadata.at("children")) {
			delete_snapshot_with_children(child.get<std::string>());
		}

		//Now we're at the bottom of the hierarchy
		//Delete the hypervisor child if we have one

		if (vm()->has_snapshot(snapshot)) {
			vm()->delete_snapshot(snapshot);
		}

		//Ok, now we need to get our parent
		auto parent = metadata.at("parent").get<std::string>();

		//Unlink the parent
		if (parent.length()) {
			fs::path parent_metadata_file = get_metadata_dir();
			parent_metadata_file /= vm()->id() + "_" + parent;

			auto parent_metadata = read_metadata_file(parent_metadata_file);
			auto& children = parent_metadata.at("children");

			for (auto it = children.begin(); it != children.end(); ++it) {
				if (it.value() == snapshot) {
					children.erase(it);
					break;
				}
			}
			write_metadata_file(parent_metadata_file, parent_metadata);
		}

		//Now we can delete the metadata file
		if (!fs::remove(metadata_file)) {
			throw std::runtime_error("Error deleting metadata file " + metadata_file.generic_string());
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("deleting snapshot"));
	}
}

bool Machine::check_config_relevance() {
	auto old_config = nlohmann::json::parse(get_metadata("vm_config"));
	auto new_config = config;
	//So....

	//1)Check if both have or don't have nics

	auto old_nics = old_config.value("nic", nlohmann::json::array());
	auto new_nics = new_config.value("nic", nlohmann::json::array());

	if (old_nics.size() != new_nics.size()) {
		return false;
	}

	if (!std::is_permutation(old_nics.begin(), old_nics.end(), new_nics.begin())) {
		return false;
	}

	if (new_config.count("loader")) {
		if (!has_key("loader_signature")) {
			return false;
		}

		fs::path loader_file = new_config.at("loader").get<std::string>();
		if (file_signature(loader_file) != get_metadata("loader_signature")) {
			return false;
		}
	}

	if (new_config.count("iso")) {
		if (!has_key("iso_signature")) {
			return false;
		}

		fs::path iso_file = new_config.at("iso").get<std::string>();
		if (file_signature(iso_file) != get_metadata("iso_signature")) {
			return false;
		}
	}

	//So... check the disks...
	//We are not sure about anything...

	//So for every disk in new config
	//check signature if we could

	if (new_config.count("disk")) {
		for (auto& disk: new_config.at("disk")) {
			if (disk.count("source")) {
				//Let's check we even have the metadata
				std::string signature_name = std::string("disk_signature@") + disk.at("name").get<std::string>();
				if (!has_key(signature_name)) {
					return false;
				}

				if (file_signature(disk.at("source").get<std::string>()) != get_metadata(signature_name)) {
					return false;
				}
			}
		}
	}

	new_config.erase("nic");
	old_config.erase("nic");

	new_config.erase("src_file");
	//old_config already doesn't have the src_file

	return old_config == new_config;
}

fs::path Machine::get_metadata_dir() const {
	return env->vm_metadata_dir() / id();
}

void Machine::press(const std::vector<std::string>& buttons) {
	for (auto& button: buttons) {
		if (current_held_keyboard_buttons.find(button) != current_held_keyboard_buttons.end()) {
			throw std::runtime_error("You can't press an already held button: " + button);
		}
	}

	vm()->press(buttons);
}

void Machine::hold(const std::vector<std::string>& buttons) {
	for (auto& button: buttons) {
		if (current_held_keyboard_buttons.find(button) != current_held_keyboard_buttons.end()) {
			throw std::runtime_error("You can't hold an already held button: " + button);
		}
	}

	vm()->hold(buttons);
	std::copy(buttons.begin(), buttons.end(), std::inserter(current_held_keyboard_buttons, current_held_keyboard_buttons.end()));
}

void Machine::release(const std::vector<std::string>& buttons) {
	if (!current_held_keyboard_buttons.size()) {
		throw std::runtime_error("There is no held buttons to release");
	}

	for (auto& button: buttons) {
		if (current_held_keyboard_buttons.find(button) == current_held_keyboard_buttons.end()) {
			throw std::runtime_error("You can't release a button that's not held: " + button);
		}
	}

	vm()->release(buttons);

	for (auto& button: buttons) {
		current_held_keyboard_buttons.erase(button);
	}
}

void Machine::release() {
	if (!current_held_keyboard_buttons.size()) {
		throw std::runtime_error("There is no held buttons to release");
	}

	std::vector<std::string> buttons_to_release(current_held_keyboard_buttons.begin(), current_held_keyboard_buttons.end());
	vm()->release(buttons_to_release);
	current_held_keyboard_buttons.clear();
}

void Machine::mouse_press(const std::vector<MouseButton>& buttons) {
	if (buttons.size() > 1) {
		throw std::runtime_error("Can't press more than 1 mouse button");
	}

	if (current_held_mouse_button != MouseButton::None) {
		throw std::runtime_error("Can't press a mouse button with any already held mouse buttons");
	}

	vm()->mouse_hold(buttons);
	coro::Timer timer;
	timer.waitFor(std::chrono::milliseconds(60));
	vm()->mouse_release(buttons);
}

void Machine::mouse_hold(const std::vector<MouseButton>& buttons) {
	if (buttons.size() > 1) {
		throw std::runtime_error("Can't hold more than 1 mouse button");
	}

	if (current_held_mouse_button != MouseButton::None) {
		throw std::runtime_error("Can't hold a mouse button: there is an already held mouse button");
	}

	vm()->mouse_hold(buttons);
	current_held_mouse_button = buttons[0];
}

void Machine::mouse_release() {
	if (current_held_mouse_button == MouseButton::None) {
		throw std::runtime_error("Can't release any mouse button: there is no held mouse buttons");
	}

	vm()->mouse_release({current_held_mouse_button});
	current_held_mouse_button = MouseButton::None;
}


}
