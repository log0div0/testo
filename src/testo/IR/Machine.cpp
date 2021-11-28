
#include "Machine.hpp"
#include "../Exceptions.hpp"
#include "../backends/Environment.hpp"
#include <fmt/format.h>

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

		if (vm()->is_defined()) {
			vm()->undefine();
		}

		auto metadata_dir = get_metadata_dir();
		if (fs::exists(metadata_dir)) {
			if (!fs::remove_all(metadata_dir)) {
				throw std::runtime_error("Error deleting metadata dir " + metadata_dir.generic_string());
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("undefining vm controller"));
	}
}

bool Machine::is_nic_plugged(const std::string& nic) {
	return vm()->is_nic_plugged(nic);
}

void Machine::plug_nic(const std::string& nic) {
	try {
		vm()->plug_nic(nic);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("plugging nic " + nic));
	}
}

void Machine::unplug_nic(const std::string& nic) {
	try {
		vm()->unplug_nic(nic);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("unplugging nic " + nic));
	}
}

bool Machine::is_link_plugged(const std::string& nic) {
	try {
		if (!vm()->is_nic_plugged(nic)) {
			throw std::runtime_error("Internal error: nic " + nic + " is not plugged");
		}
		return vm()->is_link_plugged(nic);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("checking link is plugged: " + nic));
	}

}

void Machine::plug_link(const std::string& nic) {
	try {
		if (!vm()->is_nic_plugged(nic)) {
			throw std::runtime_error("Internal error: nic " + nic + " is not plugged");
		}
		vm()->set_link(nic, true);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Plugging link: " + nic));
	}
}

void Machine::unplug_link(const std::string& nic) {
	try {
		if (!vm()->is_nic_plugged(nic)) {
			throw std::runtime_error("Internal error: nic " + nic + " is not plugged");
		}
		vm()->set_link(nic, false);
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

		if (hypervisor_snapshot_needed && vm()->is_hostdev_plugged()) {
			throw std::runtime_error("Can't take hypervisor snapshot with a hostdev plugged in. Please unplug all plugged hostdevs before the end of the test");
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
		metadata["metadata_version"] = TESTO_CURRENT_METADATA_VERSION;
		metadata["vars"] = vars;

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
	fs::path metadata_file = get_metadata_dir();
	metadata_file /= vm()->id() + "_" + snapshot;

	auto metadata = read_metadata_file(metadata_file);

	vm()->rollback(snapshot, metadata.at("opaque"));
	if (metadata.count("vars")) {
		vars = metadata.at("vars").get<std::map<std::string, std::string>>();
	}
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

void Machine::hold(KeyboardButton button) {
	auto it = std::find(current_held_keyboard_buttons.begin(), current_held_keyboard_buttons.end(), button);
	if (it != current_held_keyboard_buttons.end()) {
		throw std::runtime_error("You can't hold an already held button: " + ToString(button));
	}
	vm()->hold(button);
	current_held_keyboard_buttons.push_back(button);
}

void Machine::release(KeyboardButton button) {
	auto it = std::find(current_held_keyboard_buttons.begin(), current_held_keyboard_buttons.end(), button);
	if (it == current_held_keyboard_buttons.end()) {
		throw std::runtime_error("You can't release a button that's not held: " + ToString(button));
	}
	vm()->release(button);
	current_held_keyboard_buttons.erase(it);
}

void Machine::release() {
	if (!current_held_keyboard_buttons.size()) {
		throw std::runtime_error("There is no held buttons to release");
	}
	auto copy = current_held_keyboard_buttons;
	for (auto it = copy.rbegin(); it != copy.rend(); ++it) {
		release(*it);
	}
}

void Machine::mouse_hold(MouseButton button) {
	if (current_held_mouse_button != MouseButton::None) {
		throw std::runtime_error("Can't hold a mouse button: there is an already held mouse button");
	}

	vm()->mouse_hold(button);
	current_held_mouse_button = button;
}

void Machine::mouse_release() {
	if (current_held_mouse_button == MouseButton::None) {
		throw std::runtime_error("Can't release any mouse button: there is no held mouse buttons");
	}

	vm()->mouse_release(current_held_mouse_button);
	current_held_mouse_button = MouseButton::None;
}

void Machine::validate_config() {
	// TODO: этот метод должен быть константным
	// сейчас мешает то, что конфиг IR::Machine и конфиг VM - это
	// одно и то же. Наверное, стоит разделить эти понятия

	if (config.count("iso")) {
		fs::path iso_file = config.at("iso").get<std::string>();
		if (iso_file.is_relative()) {
			fs::path src_file(config.at("src_file").get<std::string>());
			iso_file = src_file.parent_path() / iso_file;
		}

		if (!fs::exists(iso_file)) {
			throw std::runtime_error(fmt::format("Target iso file \"{}\" does not exist", iso_file.generic_string()));
		}

		iso_file = fs::canonical(iso_file);

		config["iso"] = iso_file.generic_string();
	}

	if (config.count("loader")) {
		fs::path loader_file = config.at("loader").get<std::string>();
		if (loader_file.is_relative()) {
			fs::path src_file(config.at("src_file").get<std::string>());
			loader_file = src_file.parent_path() / loader_file;
		}

		if (!fs::exists(loader_file)) {
			throw std::runtime_error(fmt::format("Target loader file \"{}\" does not exist", loader_file.generic_string()));
		}

		loader_file = fs::canonical(loader_file);

		config["loader"] = loader_file.generic_string();
	}

	if (config.count("disk")) {
		auto& disks = config.at("disk");

		for (auto& disk: disks) {
			if (disk.count("source")) {
				fs::path source_file = disk.at("source").get<std::string>();
				if (source_file.is_relative()) {
					fs::path src_file(config.at("src_file").get<std::string>());
					source_file = src_file.parent_path() / source_file;
				}

				if (!fs::exists(source_file)) {
					throw std::runtime_error(fmt::format("Source disk image \"{}\" does not exist", source_file.generic_string()));
				}

				source_file = fs::canonical(source_file);
				disk["source"] = source_file;
			}
		}
	}

	if (!config.count("name")) {
		throw std::runtime_error("Field \"name\" is not specified");
	}

	if (!config.count("ram")) {
		throw std::runtime_error("Field \"ram\" is not specified");
	}

	if (!config.count("cpus")) {
		throw std::runtime_error("Field \"cpu\" is not specified");
	}

	{
		int cpus = config.at("cpus");
		if (cpus <= 0) {
			throw std::runtime_error("CPUs number must be a positive interger");
		}
	}

	if (!config.count("disk")) {
		throw std::runtime_error("You must specify at least 1 disk");
	}

	if (config.count("disk")) {
		auto disks = config.at("disk");

		for (auto& disk: disks) {
			if (!(disk.count("size") ^ disk.count("source"))) {
				throw std::runtime_error(fmt::format("Either field \"size\" or \"source\" must be specified for the disk \"{}\"",
					disk.at("name").get<std::string>()));
			}
		}
	}

	if (config.count("shared_folder")) {
		auto shared_folders = config.at("shared_folder");

		for (auto& shared_folder: shared_folders) {
			if (!shared_folder.count("host_path")) {
				throw std::runtime_error(fmt::format("Shared folder {} error: field \"host_path\" is not specified",
					shared_folder.at("name").get<std::string>()));
			}
			fs::path host_path = shared_folder.at("host_path").get<std::string>();
			if (host_path.is_relative()) {
				fs::path src_file(config.at("src_file").get<std::string>());
				host_path = src_file.parent_path() / host_path;
			}
			if (!fs::exists(host_path)) {
				throw std::runtime_error(fmt::format("Shared folder {} error: path \"{}\" does not exist on the host",
					shared_folder.at("name").get<std::string>(), host_path.generic_string()));
			}
			if (!fs::is_directory(host_path)) {
				throw std::runtime_error(fmt::format("Shared folder {} error: path \"{}\" is not a folder",
					shared_folder.at("name").get<std::string>(), host_path.generic_string()));
			}
		}
	}

	if (config.count("nic")) {
		auto nics = config.at("nic");
		for (auto& nic: nics) {
			if (!nic.count("attached_to") && !nic.count("attached_to_dev")) {
				throw std::runtime_error(fmt::format("Neither \"attached_to\" nor \"attached_to_dev\" is specified for the nic \"{}\"",
					nic.at("name").get<std::string>()));
			}

			if (nic.count("attached_to") && nic.count("attached_to_dev")) {
				throw std::runtime_error(fmt::format("Can't specify both \"attached_to\" and \"attached_to_dev\" for the same nic \"{}\"",
					nic.at("name").get<std::string>()));
			}

			if (nic.count("mac")) {
				std::string mac = nic.at("mac").get<std::string>();
				if (!is_mac_correct(mac)) {
					throw std::runtime_error(fmt::format("Incorrect mac address: \"{}\"", mac));
				}
			}
		}
	}

	if (config.count("video")) {
		auto videos = config.at("video");

		if (videos.size() > 1) {
			throw std::runtime_error("Multiple video devices are not supported at the moment");
		}
	}

	env->validate_vm_config(config);
}

const stb::Image<stb::RGB>& Machine::make_new_screenshot() {
	_last_screenshot = vm()->screenshot();
	return _last_screenshot;
}

const stb::Image<stb::RGB>& Machine::get_last_screenshot() const {
	return _last_screenshot;
}

const std::map<std::string, std::string>& Machine::get_vars() const {
	return vars;
}

void Machine::set_var(const std::string& var_name, const std::string& var_value) {
	vars[var_name] = var_value;
}

}
