
#include <vbox/machine.hpp>
#include <stdexcept>
#include <vbox/throw_if_failed.hpp>
#include <vbox/session.hpp>

#include <iostream>

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
		throw_if_failed(IMachine_get_StorageControllers(handle, SAFEARRAY_AS_OUT_PARAM(IStorageController*, safe_array)));
		ArrayOutIface array_out = safe_array.copy_out_iface();
		return {(IStorageController**)array_out.ifaces, (IStorageController**)(array_out.ifaces + array_out.ifaces_count)};
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

StorageController Machine::storage_controller_by_name(const std::string& name) const {
	try {
		IStorageController* result = nullptr;
		throw_if_failed(IMachine_GetStorageControllerByName(handle, StringIn(name), &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<USBController> Machine::usb_controllers() const {
	try {
		SafeArray safe_array;
		throw_if_failed(IMachine_get_USBControllers(handle, SAFEARRAY_AS_OUT_PARAM(IUSBController*, safe_array)));
		ArrayOutIface array_out = safe_array.copy_out_iface();
		return {(IUSBController**)array_out.ifaces, (IUSBController**)(array_out.ifaces + array_out.ifaces_count)};
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

USBController Machine::usb_controller_by_name(const std::string& name) const {
	try {
		IUSBController* result = nullptr;
		throw_if_failed(IMachine_GetUSBControllerByName(handle, StringIn(name), &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<MediumAttachment> Machine::medium_attachments() const {
	try {
		SafeArray safe_array;
		throw_if_failed(IMachine_get_MediumAttachments(handle, SAFEARRAY_AS_OUT_PARAM(IMediumAttachment*, safe_array)));
		ArrayOutIface array_out = safe_array.copy_out_iface();
		return {(IMediumAttachment**)array_out.ifaces, (IMediumAttachment**)(array_out.ifaces + array_out.ifaces_count)};
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<MediumAttachment> Machine::medium_attachments_of_controller(const std::string& name) const {
	try {
		SafeArray safe_array;
		throw_if_failed(IMachine_GetMediumAttachmentsOfController(handle, StringIn(name), SAFEARRAY_AS_OUT_PARAM(IMediumAttachment*, safe_array)));
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

void Machine::mount_medium(const std::string& controller, int controller_port, int device, Medium& medium, bool force) {
	try {
		throw_if_failed(IMachine_MountMedium(handle, StringIn(controller), controller_port, device, medium.handle, force));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Machine::unmount_medium(const std::string& controller, int controller_port, int device, bool force) {
	try {
		throw_if_failed(IMachine_UnmountMedium(handle, StringIn(controller), controller_port, device, force));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

USBController Machine::add_usb_controller(const std::string& name, USBControllerType type) {
	try {
		IUSBController* result = nullptr;
		throw_if_failed(IMachine_AddUSBController(handle, StringIn(name), type, &result));
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

void Machine::detach_device(const std::string& name, int controller_port, int device) {
	try {
		throw_if_failed(IMachine_DetachDevice(handle, StringIn(name), controller_port, device));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::string Machine::getExtraData(const std::string& key) const {
	try {
		BSTR result = nullptr;
		throw_if_failed(IMachine_GetExtraData(handle, StringIn(key), &result));
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Machine::setExtraData(const std::string& key, const std::string& value) const {
	try {
		throw_if_failed(IMachine_SetExtraData(handle, StringIn(key), StringIn(value)));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

NetworkAdapter Machine::getNetworkAdapter(ULONG slot) const {
	try {
		INetworkAdapter* result;
		throw_if_failed(IMachine_GetNetworkAdapter(handle, slot, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool Machine::hasSnapshot(const std::string& name) const {
	try {
		ISnapshot* result = nullptr;
		throw_if_failed(IMachine_FindSnapshot(handle, StringIn(name), &result));
		return true;
	}
	catch (const std::exception&) {
		return false;
	}
}

Snapshot Machine::findSnapshot(const std::string& name) const {
	try {
		ISnapshot* result = nullptr;
		throw_if_failed(IMachine_FindSnapshot(handle, StringIn(name), &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Progress Machine::takeSnapshot(const std::string& name, const std::string& description, bool pause) {
	try {
		IProgress* result = nullptr;
		BSTR id = nullptr;
		throw_if_failed(IMachine_TakeSnapshot(handle,
			StringIn(name),
			StringIn(description),
			pause,
			&id,
			&result));
		StringOut tmp(id);
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Progress Machine::restoreSnapshot(Snapshot& snapshot) {
	try {
		IProgress* result;
		throw_if_failed(IMachine_RestoreSnapshot(handle, snapshot.handle, &result));
		return result;
	}
	catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Progress Machine::deleteSnapshot(Snapshot& snapshot) {
	try {
		IProgress* result = nullptr;
		throw_if_failed(IMachine_DeleteSnapshot(handle, StringIn(snapshot.id()), &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Progress Machine::saveState() const {
	try {
		IProgress* result = nullptr;
		throw_if_failed(IMachine_SaveState(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

SafeArray Machine::unregister(CleanupMode cleanup_mode) {
	try {
		SafeArray safe_array;
		throw_if_failed(IMachine_Unregister(handle, cleanup_mode, SAFEARRAY_AS_OUT_PARAM(IMedium*, safe_array)));
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
			SAFEARRAY_AS_IN_PARAM(IMedium*, mediums),
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

void Machine::cpus(ULONG num) {
	try {
		throw_if_failed(IMachine_SetCPUCount(handle, num));
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
