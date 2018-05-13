
#include "machine.hpp"
#include <stdexcept>
#include "throw_if_failed.hpp"
#include "session.hpp"

namespace vbox {

Machine::Machine(IMachine* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Machine::~Machine() {
	if (handle) {
		IMachine_Release(handle);
	}
}

Machine::Machine(Machine&& other): handle(other.handle) {
	other.handle = nullptr;
}

Machine& Machine::operator=(Machine&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string Machine::name() const {
	try {
		BSTR name = nullptr;
		throw_if_failed(IMachine_get_Name(handle, &name));
		return StringOut(name);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Machine::save_settings() {
	try {
		throw_if_failed(IMachine_SaveSettings(handle));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::string Machine::settings_file_path() const {
	try {
		BSTR name = nullptr;
		throw_if_failed(IMachine_get_SettingsFilePath(handle, &name));
		return StringOut(name);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<StorageController> Machine::storage_controllers() const {
	try {
		SafeArray safe_array;
		throw_if_failed(IMachine_get_StorageControllers(handle, ComSafeArrayAsOutIfaceParam(safe_array.handle, IStorageController*)));
		ArrayOutIface array_out = safe_array.copy_out_iface();
		return {(IStorageController**)array_out.ifaces, (IStorageController**)(array_out.ifaces + array_out.ifaces_count)};
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<MediumAttachment> Machine::medium_attachments() const {
	try {
		SafeArray safe_array;
		throw_if_failed(IMachine_get_MediumAttachments(handle, ComSafeArrayAsOutIfaceParam(safe_array.handle, IMediumAttachment*)));
		ArrayOutIface array_out = safe_array.copy_out_iface();
		return {(IMediumAttachment**)array_out.ifaces, (IMediumAttachment**)(array_out.ifaces + array_out.ifaces_count)};
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

StorageController Machine::add_storage_controller(const std::string& name, StorageBus storage_bus) {
	try {
		IStorageController* result = nullptr;
		throw_if_failed(IMachine_AddStorageController(handle, StringIn(name), storage_bus, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Machine::attach_device_without_medium(const std::string& name, int controller_port, int device, DeviceType device_type) {
	try {
		throw_if_failed(IMachine_AttachDeviceWithoutMedium(handle, StringIn(name), controller_port, device, device_type));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Machine::attach_device(const std::string& name, int controller_port, int device, DeviceType device_type, const Medium& medium) {
	try {
		throw_if_failed(IMachine_AttachDevice(handle, StringIn(name), controller_port, device, device_type, medium.handle));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

SafeArray Machine::unregister(CleanupMode cleanup_mode) {
	try {
		SafeArray safe_array;
		throw_if_failed(IMachine_Unregister(handle, cleanup_mode, ComSafeArrayAsOutIfaceParam(safe_array.handle, IMedium*)));
		return safe_array;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Progress Machine::delete_config(SafeArray mediums) {
	try {
		IProgress* result = nullptr;
		throw_if_failed(IMachine_DeleteConfig(handle,
			ComSafeArrayAsInParam(mediums.handle, IMedium*),
			&result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Progress Machine::launch_vm_process(Session& session, const std::string& name, const std::string& environment) {
	try {
		IProgress* result = nullptr;
		throw_if_failed(IMachine_LaunchVMProcess(handle,
			session.handle,
			StringIn(name),
			StringIn(environment),
			&result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Machine::memory_size(ULONG size) {
	try {
		throw_if_failed(IMachine_put_MemorySize(handle, size));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Machine::vram_size(ULONG size) {
	try {
		throw_if_failed(IMachine_put_VRAMSize(handle, size));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

MachineState Machine::state() const {
	try {
		MachineState_T result = MachineState_Null;
		throw_if_failed(IMachine_get_State(handle, &result));
		return (MachineState)result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

SessionState Machine::session_state() const {
	try {
		SessionState_T result = SessionState_Null;
		throw_if_failed(IMachine_get_SessionState(handle, &result));
		return (SessionState)result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Machine::lock_machine(Session& session, LockType lock_type) {
	try {
		throw_if_failed(IMachine_LockMachine(handle, session.handle, lock_type));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::ostream& operator<<(std::ostream& stream, const Machine& machine) {
	stream << machine.name() << std::endl;
	stream << "Storage Controllers:" << std::endl;
	for (auto& storage_controller: machine.storage_controllers()) {
		stream << storage_controller << std::endl;
	}
	stream << "Medium Attachments:" << std::endl;
	for (auto& medium_attachment: machine.medium_attachments()) {
		stream << medium_attachment << std::endl;
	}
	return stream;
}

}
